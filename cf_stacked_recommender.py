"""
Two-stage stacked hybrid recommender for Yelp star-rating prediction.

Stage 1 — item-based collaborative filtering: Pearson similarity with
significance weighting (shrinks similarities computed from few co-raters) and
case amplification (suppresses weak similarities). Predictions use the top-50
neighbors' mean-centered ratings, blended toward a user/business bias fallback
when the neighborhood is thin.

Stage 2 — XGBoost regressor over 68 features. The CF prediction enters as a
feature along with 9 derivatives (neighbor count, confidence, residuals vs.
user/business baselines, confidence-weighted variants), so the tree model
learns *when* to trust CF rather than applying one global blend weight.
To keep the stack leak-free, training rows receive out-of-fold CF predictions
via a 5-fold scheme: each row is predicted by a CF model fit on the other four
folds. Test rows use CF fit on the full training set.

Remaining features: Bayesian-smoothed user/business rating estimates at
multiple shrinkage constants (with Yelp JSON profile stars as priors),
count/variance/range statistics, cold-start flags, user profile signals
(fans, votes, compliments, friends, elite years, account age), business
signals (stars, location, price, category count), checkin/tip/photo
engagement counts, and cross features.

Usage:
    spark-submit cf_stacked_recommender.py <folder_path> <test_file> <output_file>
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


def compute_stats(ratings_list):
    n   = len(ratings_list)
    avg = sum(ratings_list) / n
    var = sum((r - avg) ** 2 for r in ratings_list) / n
    return (avg, n, var, min(ratings_list), max(ratings_list))


# CF: Pearson item-based
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
    """Returns (cf_pred, n_neighbors)."""
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


# Feature builder (60 model features + 8 CF features = 68 total)
def make_feature_builder(
    user_features, business_features,
    checkin_counts, tip_biz_stats, tip_user_stats, photo_counts,
    user_avg_train, user_count_train, user_var_train,
    user_min_train, user_max_train,
    biz_avg_train, biz_count_train, biz_var_train,
    biz_min_train, biz_max_train,
    global_avg
):
    C_USER = 12
    C_BIZ  = 32
    C_JSON = 8

    def bs(raw, cnt, prior, C):
        return (raw * cnt + prior * C) / (cnt + C)

    def build(uid, bid, cf_pred, n_neighbors):
        g = global_avg

        u_cnt     = user_count_train.get(uid, 0)
        u_avg_raw = user_avg_train.get(uid, g)
        u_var     = user_var_train.get(uid, 0.0)
        u_min     = user_min_train.get(uid, g)
        u_max     = user_max_train.get(uid, g)

        b_cnt     = biz_count_train.get(bid, 0)
        b_avg_raw = biz_avg_train.get(bid, g)
        b_var     = biz_var_train.get(bid, 0.0)
        b_min     = biz_min_train.get(bid, g)
        b_max     = biz_max_train.get(bid, g)

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

        tip_b      = tip_biz_stats.get(bid,  (0, 0))
        tip_u      = tip_user_stats.get(uid,  (0, 0))
        b_checkins = checkin_counts.get(bid,  0)
        b_photos   = photo_counts.get(bid,    0)

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

        # CF-derived features
        cf_nb_conf  = min(n_neighbors, 50) / 50.0
        cf_residual = cf_pred - u_final
        cf_vs_biz   = cf_pred - b_final
        cf_vs_avg   = cf_pred - avg_pair
        cf_weighted = cf_pred * cf_nb_conf
        cf_x_bconf  = cf_pred * b_conf
        cf_x_uconf  = cf_pred * u_conf
        cf_disagree = abs(cf_pred - u_final)

        return [
            # Original 60 features
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
            math.log1p(b_checkins), math.log1p(b_photos),
            math.log1p(tip_b[0]),   math.log1p(tip_u[0]),
            tip_b[1], tip_u[1],
            u_final * b_final,    u_bias * b_bias,
            diff * u_var,         diff * b_var,
            u_var * b_var,
            math.log1p(u_cnt) * math.log1p(b_cnt),
            u_final * b_price,    b_var * math.log1p(b_cnt),
            u_final * u_conf,     b_final * b_conf,
            diff * (u_conf + b_conf), avg_pair * b_conf,
            # CF features (61–68)
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
        ]

    return build


# Main
def main():
    if len(sys.argv) != 4:
        print("Usage: spark-submit cf_stacked_recommender.py <folder_path> <test_file> <output_file>")
        sys.exit(1)

    folder_path = sys.argv[1]
    test_file   = sys.argv[2]
    output_file = sys.argv[3]

    conf = SparkConf().setAppName("cf_stacked_recommender").setMaster("local[*]")
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
            since_year   = parse_int(str(d.get("yelping_since", "2000"))[:4], 2000)
            cf_fields = [
                "compliment_hot", "compliment_more", "compliment_profile",
                "compliment_cute", "compliment_list", "compliment_note",
                "compliment_plain", "compliment_cool", "compliment_funny",
                "compliment_writer", "compliment_photos"
            ]
            total_comp  = sum(parse_int(d.get(f, 0)) for f in cf_fields)
            fs          = d.get("friends", "None")
            num_friends = len(str(fs).split(",")) if fs and str(fs).strip() not in ("None","") else 0
            es          = d.get("elite", "None")
            num_elite   = len(str(es).split(",")) if es and str(es).strip() not in ("None","") else 0
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

    # business.json
    def parse_business(line):
        try:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            stars   = parse_float(d.get("stars", 0))
            rc      = parse_int(d.get("review_count", 0))
            lat     = parse_float(d.get("latitude", 0))
            lon     = parse_float(d.get("longitude", 0))
            is_open = parse_int(d.get("is_open", 1))
            attrs   = d.get("attributes", None)
            if not isinstance(attrs, dict): attrs = {}
            price = parse_int(attrs.get("RestaurantsPriceRange2", 0), 0)
            cats  = d.get("categories", "")
            ncat  = len(cats.split(",")) if cats and isinstance(cats, str) else 0
            return (bid, (stars, rc, lat, lon, is_open, price, ncat))
        except:
            return None

    business_features = (
        sc.textFile(os.path.join(folder_path, "business.json"))
        .map(parse_business).filter(lambda x: x is not None).collectAsMap()
    )

    # checkin.json
    def parse_checkin(line):
        try:
            d   = json.loads(line)
            bid = d.get("business_id", "")
            td  = d.get("time", {})
            return (bid, sum(td.values()) if isinstance(td, dict) else 0)
        except:
            return None

    checkin_counts = (
        sc.textFile(os.path.join(folder_path, "checkin.json"))
        .map(parse_checkin).filter(lambda x: x is not None).collectAsMap()
    )

    # tip.json
    def parse_tip(line):
        try:
            d = json.loads(line)
            return (d.get("user_id",""), d.get("business_id",""), parse_int(d.get("likes",0)))
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

    # photo.json
    def parse_photo(line):
        try:
            d = json.loads(line)
            return (d.get("business_id",""), 1)
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

    # Feature builder
    build_features = make_feature_builder(
        user_features, business_features,
        checkin_counts, tip_biz_stats, tip_user_stats, photo_counts,
        user_avg_train, user_count_train, user_var_train,
        user_min_train, user_max_train,
        biz_avg_train, biz_count_train, biz_var_train,
        biz_min_train, biz_max_train,
        global_avg
    )

    # Collect all training rows
    raw_train = train_rdd.collect()
    n_train   = len(raw_train)

    # 5-fold OOF CF predictions for training rows
    K         = 5
    oof_cf    = [0.0] * n_train
    oof_nb    = [0]   * n_train
    fold_of   = [i % K for i in range(n_train)]

    for fold in range(K):
        cf_train   = [raw_train[i] for i in range(n_train) if fold_of[i] != fold]
        cf_val_idx = [i             for i in range(n_train) if fold_of[i] == fold]
        ubr, bur, u_avg_cf, b_avg_cf, g_avg_cf = build_cf_structures(cf_train)
        sim_cache = {}
        for idx in cf_val_idx:
            uid, bid, _ = raw_train[idx]
            cf_p, n_nb  = predict_cf_single(
                uid, bid, ubr, bur, u_avg_cf, b_avg_cf, g_avg_cf, sim_cache
            )
            oof_cf[idx] = cf_p
            oof_nb[idx] = n_nb

    # Build training feature matrix with OOF CF features
    X_train = [
        build_features(uid, bid, oof_cf[i], oof_nb[i])
        for i, (uid, bid, _) in enumerate(raw_train)
    ]
    y_train = [stars for _, _, stars in raw_train]

    # Train XGBoost (Bayesian-optimized with 68-feature vector)
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.007080805948990546,
        subsample=0.6,
        colsample_bytree=0.45765817440007944,
        colsample_bylevel=0.4,
        colsample_bynode=0.4,
        min_child_weight=10,
        gamma=0.0,
        reg_lambda=2.0,
        reg_alpha=0.49999999999999994,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Full-training CF structures for test predictions
    ubr_full, bur_full, u_avg_full, b_avg_full, _ = build_cf_structures(raw_train)

    # Test data
    test_rdd   = sc.textFile(test_file)
    test_hdr   = test_rdd.first()
    test_rdd   = (
        test_rdd
        .filter(lambda x: x != test_hdr)
        .map(lambda x: x.split(","))
        .map(lambda x: (x[0], x[1]))
    )
    test_pairs = test_rdd.collect()

    # CF predictions on test set (full training CF)
    sim_cache_test = {}
    test_cf_preds  = []
    test_nb        = []
    for uid, bid in test_pairs:
        cf_p, n_nb = predict_cf_single(
            uid, bid, ubr_full, bur_full,
            u_avg_full, b_avg_full, global_avg, sim_cache_test
        )
        test_cf_preds.append(cf_p)
        test_nb.append(n_nb)

    # Build test features & predict
    X_test = [
        build_features(uid, bid, cf_p, n_nb)
        for (uid, bid), cf_p, n_nb in zip(test_pairs, test_cf_preds, test_nb)
    ]
    preds = model.predict(X_test)
    preds = [max(1.0, min(5.0, float(p))) for p in preds]

    # Write output
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for (uid, bid), pred in zip(test_pairs, preds):
            writer.writerow([uid, bid, pred])

    train_rdd.unpersist()
    sc.stop()


if __name__ == "__main__":
    main()
