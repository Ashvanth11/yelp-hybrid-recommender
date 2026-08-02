"""
Feature-rich XGBoost recommender for Yelp star-rating prediction.

Single-stage gradient-boosted model over ~120 engineered features, built for a
strict single-core runtime budget: the expensive item-CF stack of the
companion model (cf_stacked_recommender.py) is replaced with a fast, non-leaky
user/business bias proxy (user_avg + business_avg - global_avg) in the CF
feature slots, and the saved runtime is spent on a much wider feature set.

Core ideas:
  - Bayesian-smoothed user and business rating estimates using JSON priors.
  - Leave-one-out user/business training statistics for each training row to
    avoid target leakage from raw averages, variance, min/max, and rating-rate
    features (a user with 3 ratings has an average that is 1/3 the target
    itself).
  - Final calibration expands predictions around the global mean because the
    raw model is slightly conservative on 1-star and 5-star examples.

Feature groups:
  - Smoothed user/business averages at multiple shrinkage constants.
  - User/business confidence, count, variance, range, min, max, and cold flags.
  - JSON user signals: average_stars, review_count, fans, votes, compliments,
    friends, elite count, yelping age.
  - JSON business signals: stars, review_count, location, open flag, price,
    category count.
  - Business attributes: alcohol, kids, outdoor seating, noise, wifi, TV, attire.
  - Hours/checkin signals: weekly hours, weekend-open flag, evening/weekend
    checkin ratios, total checkins.
  - Tip/photo engagement counts.
  - Rating-distribution rates: low/high rates plus exact 1-star and 5-star
    rates for users and businesses.
  - Top business category indicators from business.json.
  - Cross features combining user/business bias, confidence, variance, price,
    and rating-distribution rates.

Validation RMSE: 0.9775 on ~142k held-out pairs (~100s single-core runtime).

Usage:
    spark-submit feature_rich_recommender.py <folder_path> <test_file> <output_file>
"""

import sys
import os
import csv
import math
import json
from pyspark import SparkContext, SparkConf


# Helpers
def parse_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default


def parse_int(val, default=0):
    try:
        return int(val)
    except:
        return default


def clamp_rating(x):
    return max(1.0, min(5.0, x))


def write_output(output_file, test_pairs, preds):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for (uid, bid), pred in zip(test_pairs, preds):
            writer.writerow([uid, bid, pred])


def compute_stats(ratings_list):
    n   = len(ratings_list)
    avg = sum(ratings_list) / n
    var = sum((r - avg) ** 2 for r in ratings_list) / n
    return (avg, n, var, min(ratings_list), max(ratings_list))


def build_rating_aggregates(rows, key_pos):
    aggs = {}
    for row in rows:
        key = row[key_pos]
        r   = row[2]
        if key not in aggs:
            aggs[key] = {
                "cnt": 0, "sum": 0.0, "sumsq": 0.0,
                "min": 6.0, "min_count": 0, "second_min": 6.0,
                "max": 0.0, "max_count": 0, "second_max": 0.0,
                "low": 0, "high": 0, "one": 0, "five": 0,
            }
        a = aggs[key]
        a["cnt"] += 1
        a["sum"] += r
        a["sumsq"] += r * r
        if r <= 2.0:
            a["low"] += 1
        if r >= 4.0:
            a["high"] += 1
        if r == 1.0:
            a["one"] += 1
        if r == 5.0:
            a["five"] += 1

        if r < a["min"]:
            a["second_min"] = a["min"]
            a["min"] = r
            a["min_count"] = 1
        elif r == a["min"]:
            a["min_count"] += 1
        elif r < a["second_min"]:
            a["second_min"] = r

        if r > a["max"]:
            a["second_max"] = a["max"]
            a["max"] = r
            a["max_count"] = 1
        elif r == a["max"]:
            a["max_count"] += 1
        elif r > a["second_max"]:
            a["second_max"] = r
    return aggs


def stats_from_parts(
    cnt, total, sumsq, min_val, max_val,
    low_cnt, high_cnt, one_cnt, five_cnt, global_avg
):
    if cnt <= 0:
        return (global_avg, 0, 0.0, global_avg, global_avg, 0.0, 0.0, 0.0, 0.0)
    avg = total / cnt
    var = max(0.0, (sumsq / cnt) - avg * avg)
    return (
        avg, cnt, var, min_val, max_val,
        low_cnt / cnt, high_cnt / cnt, one_cnt / cnt, five_cnt / cnt
    )


def full_stats_from_agg(agg, global_avg):
    return stats_from_parts(
        agg["cnt"], agg["sum"], agg["sumsq"], agg["min"], agg["max"],
        agg["low"], agg["high"], agg["one"], agg["five"], global_avg
    )


def loo_stats_from_agg(agg, rating, global_avg):
    cnt = agg["cnt"] - 1
    if cnt <= 0:
        return (global_avg, 0, 0.0, global_avg, global_avg, 0.0, 0.0, 0.0, 0.0)

    min_val = agg["min"]
    if rating == agg["min"] and agg["min_count"] == 1:
        min_val = agg["second_min"]

    max_val = agg["max"]
    if rating == agg["max"] and agg["max_count"] == 1:
        max_val = agg["second_max"]

    low_cnt  = agg["low"]  - (1 if rating <= 2.0 else 0)
    high_cnt = agg["high"] - (1 if rating >= 4.0 else 0)
    one_cnt  = agg["one"]  - (1 if rating == 1.0 else 0)
    five_cnt = agg["five"] - (1 if rating == 5.0 else 0)
    return stats_from_parts(
        cnt, agg["sum"] - rating, agg["sumsq"] - rating * rating,
        min_val, max_val, low_cnt, high_cnt, one_cnt, five_cnt, global_avg
    )


# Attribute parsing helpers (verified against actual schema)
# Attributes are stored as strings: "True"/"False", or words like "average"
def parse_bool_attr(val, default=0):
    if val == 'True':  return 1
    if val == 'False': return 0
    return default


def parse_noise(val):
    # "quiet"=0, "average"=1, "loud"=2, "very_loud"=3, missing=-1
    return {'quiet': 0, 'average': 1, 'loud': 2, 'very_loud': 3}.get(val, -1)


def parse_attire(val):
    # "casual"=0, "dressy"=1, "formal"=2, missing=0
    return {'casual': 0, 'dressy': 1, 'formal': 2}.get(val, 0)


def parse_alcohol(val):
    # "none" or "False" or missing = 0, anything else (beer_and_wine, full_bar) = 1
    if not val or val in ('none', 'False', 'None'):
        return 0
    return 1


def parse_wifi(val):
    # "no" or "False" or missing = 0, otherwise 1
    if not val or val in ('no', 'False', 'None'):
        return 0
    return 1


TOP_CATEGORIES = (
    "restaurants", "food", "nightlife", "bars", "shopping",
    "beauty & spas", "health & medical", "home services",
    "local services", "automotive", "active life", "event planning & services",
    "coffee & tea", "sandwiches", "american (traditional)",
    "american (new)", "pizza", "breakfast & brunch", "mexican",
    "italian", "chinese", "japanese", "seafood", "fast food",
)


def parse_hours_features(hours_dict):
    """
    Returns (hours_per_week, is_open_weekend).
    hours_dict keys: "Monday", "Tuesday", etc.
    values: "HH:MM-HH:MM" e.g. "11:0-21:0"
    """
    if not hours_dict or not isinstance(hours_dict, dict):
        return 0.0, 0
    weekend_days = {'Saturday', 'Sunday'}
    total_hours  = 0.0
    is_weekend   = 0
    for day, timestr in hours_dict.items():
        if day in weekend_days:
            is_weekend = 1
        try:
            start, end = timestr.split('-')
            sh, sm = map(int, start.split(':'))
            eh, em = map(int, end.split(':'))
            hrs = (eh + em / 60.0) - (sh + sm / 60.0)
            if hrs < 0:
                hrs += 24  # overnight hours
            total_hours += hrs
        except:
            pass
    return total_hours, is_weekend


# CF: Pearson item-based (shared with cf_stacked_recommender.py)
def pearson_similarity(item1_users, item2_users):
    common = set(item1_users) & set(item2_users)
    n = len(common)
    if n < 2:
        return 0.0, n
    avg1 = sum(item1_users[u] for u in common) / n
    avg2 = sum(item2_users[u] for u in common) / n
    num = d1sq = d2sq = 0.0
    for u in common:
        d1 = item1_users[u] - avg1
        d2 = item2_users[u] - avg2
        num  += d1 * d2
        d1sq += d1 * d1
        d2sq += d2 * d2
    if d1sq == 0 or d2sq == 0:
        return 0.0, n
    return num / (math.sqrt(d1sq) * math.sqrt(d2sq)), n


def build_cf_structures(rows):
    from collections import defaultdict
    ubr = defaultdict(dict)
    bur = defaultdict(dict)
    for uid, bid, r in rows:
        ubr[uid][bid] = r
        bur[bid][uid] = r
    u_avg = {u: sum(v.values()) / len(v) for u, v in ubr.items()}
    b_avg = {b: sum(v.values()) / len(v) for b, v in bur.items()}
    g_avg = sum(r for _, _, r in rows) / len(rows)
    return dict(ubr), dict(bur), u_avg, b_avg, g_avg


def predict_cf_single(uid, bid, ubr, bur, u_avg, b_avg, g_avg, sim_cache):
    u_ex = uid in ubr
    b_ex = bid in bur
    if not u_ex and not b_ex:
        return g_avg, 0
    if not u_ex:
        return b_avg.get(bid, g_avg), 0
    if not b_ex:
        return u_avg.get(uid, g_avg), 0
    rated = ubr[uid]
    if bid in rated:
        return rated[bid], 50
    target_users = bur[bid]
    target_avg   = b_avg.get(bid, g_avg)
    neighbors    = []
    for other_bid, rating in rated.items():
        if other_bid == bid:
            continue
        pair     = (bid, other_bid)
        rev_pair = (other_bid, bid)
        if pair in sim_cache:
            sim = sim_cache[pair]
        elif rev_pair in sim_cache:
            sim = sim_cache[rev_pair]
        else:
            sim, nc = pearson_similarity(target_users, bur.get(other_bid, {}))
            if nc > 0:
                sim *= min(nc, 50) / 50.0
            if sim != 0.0:
                sim = sim * (abs(sim) ** 1.5)
            sim_cache[pair] = sim
        if abs(sim) > 0.01:
            neighbors.append((sim, rating, b_avg.get(other_bid, g_avg)))
    if not neighbors:
        return clamp_rating(u_avg.get(uid, g_avg) + b_avg.get(bid, g_avg) - g_avg), 0
    neighbors.sort(key=lambda x: -abs(x[0]))
    top = neighbors[:50]
    n   = len(top)
    num = den = 0.0
    for sim, r, oa in top:
        num += sim * (r - oa)
        den += abs(sim)
    if den == 0:
        return clamp_rating(u_avg.get(uid, g_avg) + b_avg.get(bid, g_avg) - g_avg), 0
    pred = target_avg + num / den
    if n < 15:
        conf     = n / 15.0
        fallback = u_avg.get(uid, g_avg) + b_avg.get(bid, g_avg) - g_avg
        pred     = conf * pred + (1.0 - conf) * fallback
    return clamp_rating(pred), n


# Feature builder (79 features: 68 original + 11 new business features)
def make_feature_builder(
    user_features, business_features, business_extra,
    checkin_features, tip_biz_stats, tip_user_stats, photo_counts,
    user_avg_train, user_count_train, user_var_train,
    user_min_train, user_max_train,
    biz_avg_train, biz_count_train, biz_var_train,
    biz_min_train, biz_max_train,
    user_low_rate_train, user_high_rate_train,
    biz_low_rate_train, biz_high_rate_train,
    user_one_rate_train, user_five_rate_train,
    biz_one_rate_train, biz_five_rate_train,
    global_avg
):
    C_USER = 12
    C_BIZ  = 32
    C_JSON = 8

    def bs(raw, cnt, prior, C):
        return (raw * cnt + prior * C) / (cnt + C)

    def build(uid, bid, cf_pred, n_neighbors, stat_override=None):
        g = global_avg

        if stat_override:
            (
                u_avg_raw, u_cnt, u_var, u_min, u_max,
                u_low_rate, u_high_rate, u_one_rate, u_five_rate,
                b_avg_raw, b_cnt, b_var, b_min, b_max,
                b_low_rate, b_high_rate, b_one_rate, b_five_rate
            ) = stat_override
        else:
            u_cnt       = user_count_train.get(uid, 0)
            u_avg_raw   = user_avg_train.get(uid, g)
            u_var       = user_var_train.get(uid, 0.0)
            u_min       = user_min_train.get(uid, g)
            u_max       = user_max_train.get(uid, g)
            u_low_rate  = user_low_rate_train.get(uid, 0.0)
            u_high_rate = user_high_rate_train.get(uid, 0.0)
            u_one_rate  = user_one_rate_train.get(uid, 0.0)
            u_five_rate = user_five_rate_train.get(uid, 0.0)

            b_cnt       = biz_count_train.get(bid, 0)
            b_avg_raw   = biz_avg_train.get(bid, g)
            b_var       = biz_var_train.get(bid, 0.0)
            b_min       = biz_min_train.get(bid, g)
            b_max       = biz_max_train.get(bid, g)
            b_low_rate  = biz_low_rate_train.get(bid, 0.0)
            b_high_rate = biz_high_rate_train.get(bid, 0.0)
            b_one_rate  = biz_one_rate_train.get(bid, 0.0)
            b_five_rate = biz_five_rate_train.get(bid, 0.0)

        uf = user_features.get(uid)
        if uf:
            (u_json_avg, u_json_rc, u_fans, u_useful, u_funny, u_cool,
             u_since, u_comp, u_friends, u_elite, u_fpr, u_upr) = uf
        else:
            u_json_avg = g; u_json_rc = 0; u_fans = 0
            u_useful = 0; u_funny = 0; u_cool = 0; u_since = 2015
            u_comp = 0; u_friends = 0; u_elite = 0
            u_fpr = 0.0; u_upr = 0.0

        bf = business_features.get(bid)
        if bf:
            b_json_stars, b_json_rc, b_lat, b_lon, b_is_open, b_price, b_num_cats = bf
        else:
            b_json_stars = g; b_json_rc = 0
            b_lat = 0.0; b_lon = 0.0; b_is_open = 1; b_price = 0; b_num_cats = 0

        # New business extra features
        bx = business_extra.get(bid)
        if bx:
            (b_alcohol, b_kids, b_outdoor, b_noise, b_wifi,
             b_tv, b_attire, b_hours_pw, b_is_weekend) = bx[:9]
            b_cat_flags = bx[9:]
        else:
            b_alcohol = 0; b_kids = 0; b_outdoor = 0; b_noise = -1
            b_wifi = 0; b_tv = 0; b_attire = 0; b_hours_pw = 0.0; b_is_weekend = 0
            b_cat_flags = (0,) * len(TOP_CATEGORIES)

        # Checkin distribution features
        cf_dist = checkin_features.get(bid)
        if cf_dist:
            b_evening_ratio, b_weekend_ratio, b_total_checkins = cf_dist
        else:
            b_evening_ratio = 0.0; b_weekend_ratio = 0.0; b_total_checkins = 0

        tip_b      = tip_biz_stats.get(bid,  (0, 0))
        tip_u      = tip_user_stats.get(uid,  (0, 0))
        b_photos   = photo_counts.get(bid, 0)

        u_prior = u_json_avg   if uf else g
        b_prior = b_json_stars if bf else g

        u_smooth = bs(u_avg_raw, u_cnt, u_prior, C_USER)
        b_smooth = bs(b_avg_raw, b_cnt, b_prior, C_BIZ)
        u_js     = bs(u_json_avg,   u_json_rc, g, C_JSON)
        b_js     = bs(b_json_stars, b_json_rc, g, C_JSON)

        u_final = u_smooth if u_cnt > 0 else u_js
        b_final = b_smooth if b_cnt > 0 else b_js

        u_conf = u_cnt / (u_cnt + C_USER)
        b_conf = b_cnt / (b_cnt + C_BIZ)

        u_c5  = bs(u_avg_raw, u_cnt, u_prior, 5)
        u_c15 = bs(u_avg_raw, u_cnt, u_prior, 15)
        b_c10 = bs(b_avg_raw, b_cnt, b_prior, 10)
        b_c35 = bs(b_avg_raw, b_cnt, b_prior, 35)

        avg_pair = (u_final + b_final) / 2.0
        diff     = u_final - b_final
        u_bias   = u_final - g
        b_bias   = b_final - g
        u_range  = u_max - u_min
        b_range  = b_max - b_min
        u_years  = 2019 - u_since

        cf_nb_conf  = min(n_neighbors, 50) / 50.0
        cf_residual = cf_pred - u_final
        cf_vs_biz   = cf_pred - b_final
        cf_vs_avg   = cf_pred - avg_pair
        cf_weighted = cf_pred * cf_nb_conf
        cf_x_bconf  = cf_pred * b_conf
        cf_x_uconf  = cf_pred * u_conf
        cf_disagree = abs(cf_pred - u_final)

        features = [
            # ---- Base 60 features (shared with cf_stacked_recommender.py) ----
            u_final, b_final, avg_pair, avg_pair ** 2,
            u_js, b_js,
            u_c5, u_c15, b_c10, b_c35,
            u_conf, b_conf,
            math.log1p(u_cnt),    math.log1p(b_cnt),
            math.log1p(u_json_rc),math.log1p(b_json_rc),
            diff, abs(diff), u_bias, b_bias,
            u_var, b_var, u_range, b_range,
            1 if u_cnt == 0 else 0, 1 if b_cnt == 0 else 0,
            math.log1p(u_fans),   math.log1p(u_useful),
            math.log1p(u_comp),   math.log1p(u_friends),
            u_elite, u_fpr, u_upr, u_years,
            b_json_stars, b_lat, b_lon, b_is_open, b_price, b_num_cats,
            math.log1p(b_total_checkins), math.log1p(b_photos),
            math.log1p(tip_b[0]),   math.log1p(tip_u[0]),
            tip_b[1], tip_u[1],
            u_final * b_final,    u_bias * b_bias,
            diff * u_var,         diff * b_var,
            u_var * b_var,
            math.log1p(u_cnt) * math.log1p(b_cnt),
            u_final * b_price,    b_var * math.log1p(b_cnt),
            u_final * u_conf,     b_final * b_conf,
            diff * (u_conf + b_conf), avg_pair * b_conf,
            # ---- CF stack features (61-70) ----
            cf_pred,
            float(n_neighbors),
            cf_nb_conf,
            cf_residual,
            cf_vs_biz,
            cf_vs_avg,
            cf_weighted,
            cf_x_bconf,
            cf_x_uconf,
            cf_disagree,
            # ---- NEW: Business attribute features (71-77) ----
            float(b_alcohol),
            float(b_kids),
            float(b_outdoor),
            float(b_noise),       # -1 if unknown, 0-3 otherwise
            float(b_wifi),
            float(b_tv),
            float(b_attire),      # 0=casual, 1=dressy, 2=formal
            # ---- NEW: Business hours features (78-79) ----
            b_hours_pw,           # total hours open per week
            float(b_is_weekend),  # 1 if open on weekend
            # ---- NEW: Checkin distribution features (80-81) ----
            b_evening_ratio,      # fraction of checkins in evening (17-23)
            b_weekend_ratio,      # fraction of checkins on Sat/Sun
            # ---- NEW: Rating-distribution features (OOF for train) ----
            u_low_rate,
            u_high_rate,
            b_low_rate,
            b_high_rate,
            u_low_rate * b_low_rate,
            u_high_rate * b_high_rate,
            u_one_rate,
            u_five_rate,
            b_one_rate,
            b_five_rate,
            u_one_rate * b_one_rate,
            u_five_rate * b_five_rate,
            u_one_rate * b_low_rate,
            u_five_rate * b_high_rate,
        ]
        features.extend(float(x) for x in b_cat_flags)
        return features

    return build


# Main
def main():
    if len(sys.argv) != 4:
        print("Usage: spark-submit feature_rich_recommender.py <folder_path> <test_file> <output_file>")
        sys.exit(1)

    folder_path = sys.argv[1]
    test_file   = sys.argv[2]
    output_file = sys.argv[3]

    conf = SparkConf().setAppName("feature_rich_recommender").setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # Training data
    train_path = os.path.join(folder_path, "yelp_train.csv")
    train_rdd  = sc.textFile(train_path)
    train_hdr  = train_rdd.first()
    train_rdd  = (
        train_rdd
        .filter(lambda x: x != train_hdr)
        .map(lambda x: x.split(","))
        .map(lambda x: (x[0], x[1], parse_float(x[2])))
        .cache()
    )

    # user.json
    def parse_user(line):
        try:
            d   = json.loads(line)
            uid = d.get("user_id", "")
            avg_stars    = parse_float(d.get("average_stars", 0))
            review_count = parse_int(d.get("review_count", 0))
            fans         = parse_int(d.get("fans", 0))
            useful       = parse_int(d.get("useful", 0))
            funny        = parse_int(d.get("funny", 0))
            cool         = parse_int(d.get("cool", 0))
            # yelping_since is "YYYY-MM-DD" — parse year only
            since_year   = parse_int(str(d.get("yelping_since", "2000-01-01"))[:4], 2000)
            cf_fields = [
                "compliment_hot", "compliment_more", "compliment_profile",
                "compliment_cute", "compliment_list", "compliment_note",
                "compliment_plain", "compliment_cool", "compliment_funny",
                "compliment_writer", "compliment_photos"
            ]
            total_comp  = sum(parse_int(d.get(f, 0)) for f in cf_fields)
            fs          = d.get("friends", "None")
            num_friends = len(str(fs).split(",")) if fs and str(fs).strip() not in ("None", "") else 0
            es          = d.get("elite", "None")
            num_elite   = len(str(es).split(",")) if es and str(es).strip() not in ("None", "") else 0
            fpr = fans   / review_count if review_count > 0 else 0.0
            upr = useful / review_count if review_count > 0 else 0.0
            return (uid, (avg_stars, review_count, fans, useful, funny, cool,
                          since_year, total_comp, num_friends, num_elite, fpr, upr))
        except:
            return None

    user_features = (
        sc.textFile(os.path.join(folder_path, "user.json"))
        .map(parse_user).filter(lambda x: x is not None).collectAsMap()
    )

    # business.json — split into original features + new extra features
    def parse_business(line):
        try:
            d   = json.loads(line)
            bid = d.get("business_id", "")

            # Original features
            stars   = parse_float(d.get("stars", 0))
            rc      = parse_int(d.get("review_count", 0))
            lat     = parse_float(d.get("latitude", 0))
            lon     = parse_float(d.get("longitude", 0))
            is_open = parse_int(d.get("is_open", 1))
            cats    = d.get("categories", "") or ""
            ncat    = len(cats.split(",")) if cats else 0
            cat_set = set(c.strip().lower() for c in cats.split(",") if c.strip())
            cat_flags = tuple(1 if c in cat_set else 0 for c in TOP_CATEGORIES)

            attrs = d.get("attributes", None)
            if not isinstance(attrs, dict):
                attrs = {}
            price = parse_int(attrs.get("RestaurantsPriceRange2", 0), 0)

            # New attribute features (verified against actual schema)
            alcohol  = parse_alcohol(attrs.get("Alcohol", "none"))
            kids     = parse_bool_attr(attrs.get("GoodForKids", "False"))
            outdoor  = parse_bool_attr(attrs.get("OutdoorSeating", "False"))
            noise    = parse_noise(attrs.get("NoiseLevel", ""))
            wifi     = parse_wifi(attrs.get("WiFi", "no"))
            tv       = parse_bool_attr(attrs.get("HasTV", "False"))
            attire   = parse_attire(attrs.get("RestaurantsAttire", "casual"))

            # Hours features
            hours = d.get("hours", None)
            hours_pw, is_weekend = parse_hours_features(hours)

            return (
                bid,
                (stars, rc, lat, lon, is_open, price, ncat),
                (alcohol, kids, outdoor, noise, wifi, tv, attire, hours_pw, is_weekend) + cat_flags
            )
        except:
            return None

    biz_rdd = (
        sc.textFile(os.path.join(folder_path, "business.json"))
        .map(parse_business).filter(lambda x: x is not None).cache()
    )
    business_features = biz_rdd.map(lambda x: (x[0], x[1])).collectAsMap()
    business_extra    = biz_rdd.map(lambda x: (x[0], x[2])).collectAsMap()
    biz_rdd.unpersist()

    # checkin.json — total count + evening/weekend ratios
    def parse_checkin(line):
        try:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            td  = d.get("time", {})
            if not isinstance(td, dict) or not td:
                return (bid, (0.0, 0.0, 0))
            weekend_days = {'Sat', 'Sun'}
            total    = 0
            evening  = 0
            weekend  = 0
            for key, cnt in td.items():
                parts = key.split('-')
                if len(parts) != 2:
                    continue
                day, hour_str = parts
                try:
                    hour = int(hour_str)
                except:
                    continue
                total   += cnt
                if day in weekend_days:
                    weekend += cnt
                if hour >= 17:
                    evening += cnt
            if total == 0:
                return (bid, (0.0, 0.0, 0))
            return (bid, (evening / total, weekend / total, total))
        except:
            return None

    checkin_features = (
        sc.textFile(os.path.join(folder_path, "checkin.json"))
        .map(parse_checkin).filter(lambda x: x is not None).collectAsMap()
    )

    # tip.json
    def parse_tip(line):
        try:
            d = json.loads(line)
            return (d.get("user_id", ""), d.get("business_id", ""),
                    parse_int(d.get("likes", 0)))
        except:
            return None

    tip_parsed = (
        sc.textFile(os.path.join(folder_path, "tip.json"))
        .map(parse_tip).filter(lambda x: x is not None).cache()
    )
    tip_biz_stats = (
        tip_parsed.map(lambda x: (x[1], (1, x[2])))
        .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])).collectAsMap()
    )
    tip_user_stats = (
        tip_parsed.map(lambda x: (x[0], (1, x[2])))
        .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])).collectAsMap()
    )
    tip_parsed.unpersist()

    # photo.json — total count per business (label breakdown collinear, skipped)
    def parse_photo(line):
        try:
            d = json.loads(line)
            return (d.get("business_id", ""), 1)
        except:
            return None

    photo_counts = (
        sc.textFile(os.path.join(folder_path, "photo.json"))
        .map(parse_photo).filter(lambda x: x is not None)
        .reduceByKey(lambda a, b: a + b).collectAsMap()
    )

    # Per-user / per-business training stats
    user_ratings_rdd = (
        train_rdd.map(lambda x: (x[0], x[2]))
        .groupByKey().mapValues(list).mapValues(compute_stats).cache()
    )
    u_stats_map      = user_ratings_rdd.collectAsMap()
    user_avg_train   = {k: v[0] for k, v in u_stats_map.items()}
    user_count_train = {k: v[1] for k, v in u_stats_map.items()}
    user_var_train   = {k: v[2] for k, v in u_stats_map.items()}
    user_min_train   = {k: v[3] for k, v in u_stats_map.items()}
    user_max_train   = {k: v[4] for k, v in u_stats_map.items()}
    user_ratings_rdd.unpersist()

    biz_ratings_rdd = (
        train_rdd.map(lambda x: (x[1], x[2]))
        .groupByKey().mapValues(list).mapValues(compute_stats).cache()
    )
    b_stats_map     = biz_ratings_rdd.collectAsMap()
    biz_avg_train   = {k: v[0] for k, v in b_stats_map.items()}
    biz_count_train = {k: v[1] for k, v in b_stats_map.items()}
    biz_var_train   = {k: v[2] for k, v in b_stats_map.items()}
    biz_min_train   = {k: v[3] for k, v in b_stats_map.items()}
    biz_max_train   = {k: v[4] for k, v in b_stats_map.items()}
    biz_ratings_rdd.unpersist()

    global_avg = train_rdd.map(lambda x: x[2]).mean()
    raw_train  = train_rdd.collect()
    n_train    = len(raw_train)

    user_aggs = build_rating_aggregates(raw_train, 0)
    biz_aggs  = build_rating_aggregates(raw_train, 1)

    user_full_stats = {k: full_stats_from_agg(v, global_avg) for k, v in user_aggs.items()}
    biz_full_stats  = {k: full_stats_from_agg(v, global_avg) for k, v in biz_aggs.items()}
    user_low_rate_train  = {k: v[5] for k, v in user_full_stats.items()}
    user_high_rate_train = {k: v[6] for k, v in user_full_stats.items()}
    user_one_rate_train  = {k: v[7] for k, v in user_full_stats.items()}
    user_five_rate_train = {k: v[8] for k, v in user_full_stats.items()}
    biz_low_rate_train   = {k: v[5] for k, v in biz_full_stats.items()}
    biz_high_rate_train  = {k: v[6] for k, v in biz_full_stats.items()}
    biz_one_rate_train   = {k: v[7] for k, v in biz_full_stats.items()}
    biz_five_rate_train  = {k: v[8] for k, v in biz_full_stats.items()}

    loo_stat_overrides = []
    for uid, bid, rating in raw_train:
        u_loo = loo_stats_from_agg(user_aggs[uid], rating, global_avg)
        b_loo = loo_stats_from_agg(biz_aggs[bid], rating, global_avg)
        loo_stat_overrides.append((
            u_loo[0], u_loo[1], u_loo[2], u_loo[3], u_loo[4],
            u_loo[5], u_loo[6], u_loo[7], u_loo[8],
            b_loo[0], b_loo[1], b_loo[2], b_loo[3], b_loo[4],
            b_loo[5], b_loo[6], b_loo[7], b_loo[8],
        ))

    build_features = make_feature_builder(
        user_features, business_features, business_extra,
        checkin_features, tip_biz_stats, tip_user_stats, photo_counts,
        user_avg_train, user_count_train, user_var_train,
        user_min_train, user_max_train,
        biz_avg_train, biz_count_train, biz_var_train,
        biz_min_train, biz_max_train,
        user_low_rate_train, user_high_rate_train,
        biz_low_rate_train, biz_high_rate_train,
        user_one_rate_train, user_five_rate_train,
        biz_one_rate_train, biz_five_rate_train,
        global_avg
    )

    test_rdd = sc.textFile(test_file)
    test_hdr = test_rdd.first()
    test_rdd = (
        test_rdd
        .filter(lambda x: x != test_hdr)
        .map(lambda x: x.split(","))
        .map(lambda x: (x[0], x[1]))
    )
    test_pairs = test_rdd.collect()

    # The single-core runtime budget rules out the expensive item-CF OOF/test
    # loops (see cf_stacked_recommender.py). Use a cheap non-leaky bias proxy
    # in the CF slots so the leave-one-out target-stat model fits the budget.
    oof_cf = []
    oof_nb = []
    for st in loo_stat_overrides:
        u_avg_loo = st[0]
        u_cnt_loo = st[1]
        b_avg_loo = st[9]
        b_cnt_loo = st[10]
        if u_cnt_loo > 0 and b_cnt_loo > 0:
            cf_p = u_avg_loo + b_avg_loo - global_avg
        elif u_cnt_loo > 0:
            cf_p = u_avg_loo
        elif b_cnt_loo > 0:
            cf_p = b_avg_loo
        else:
            cf_p = global_avg
        oof_cf.append(clamp_rating(cf_p))
        oof_nb.append(0)

    X_train = [
        build_features(uid, bid, oof_cf[i], oof_nb[i], loo_stat_overrides[i])
        for i, (uid, bid, _) in enumerate(raw_train)
    ]
    y_train = [stars for _, _, stars in raw_train]

    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.02,
        objective="reg:squarederror",
        base_score=global_avg,
        subsample=0.5983273008793066,
        colsample_bytree=0.567536501065443,
        colsample_bylevel=0.5330728248161238,
        min_child_weight=17,
        gamma=0.10988871719602569,
        reg_lambda=3.0255954163146557,
        reg_alpha=0.6052439683929954,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    test_cf_preds  = []
    test_nb        = []
    for uid, bid in test_pairs:
        u = user_avg_train.get(uid)
        b = biz_avg_train.get(bid)
        if u is not None and b is not None:
            cf_p = u + b - global_avg
        elif u is not None:
            cf_p = u
        elif b is not None:
            cf_p = b
        else:
            cf_p = global_avg
        test_cf_preds.append(clamp_rating(cf_p))
        test_nb.append(0)

    X_test = [
        build_features(uid, bid, cf_p, n_nb)
        for (uid, bid), cf_p, n_nb in zip(test_pairs, test_cf_preds, test_nb)
    ]
    preds = model.predict(X_test)
    # Validation-calibrated expansion around the train global mean. The model
    # is slightly too conservative on 1-star and 5-star cases.
    preds = [
        max(1.0, min(5.0, global_avg + 1.063 * (float(p) - global_avg) - 0.0015))
        for p in preds
    ]
    write_output(output_file, test_pairs, preds)

    train_rdd.unpersist()
    sc.stop()


if __name__ == "__main__":
    main()
