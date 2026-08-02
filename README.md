# Yelp Hybrid Recommender

Predicting the star rating a user will give a business on the Yelp dataset.
Two complementary models, both PySpark + XGBoost, exploring opposite ends of
the same trade-off: **spend the compute on a smarter collaborative-filtering
stage, or on a richer feature set?**

| | `cf_stacked_recommender.py` | `feature_rich_recommender.py` |
|---|---|---|
| Architecture | Two-stage stack: item-based CF → XGBoost | Single-stage XGBoost |
| CF signal | Real item-based Pearson CF, fed in via 5-fold out-of-fold stacking | Fast bias proxy (`user_avg + biz_avg − global_avg`) |
| Features | 68 | ~120 |
| Leakage control | OOF CF predictions | Leave-one-out per-row training statistics |
| Calibration | Clamp to [1, 5] | Validation-tuned expansion around the global mean, then clamp |
| Runtime (single core) | Minutes (dominated by pairwise similarities) | ~100 s |
| Validation RMSE | ~0.98 | **0.9775** |

## Model 1: CF-stacked (`cf_stacked_recommender.py`)

The architecturally interesting one. Its strength is **how it combines two
fundamentally different signals without leaking the target**:

- **Stage 1** is classic item-based collaborative filtering — Pearson
  similarity over co-raters, with *significance weighting* (similarities from
  few co-raters are shrunk by `min(n, 50)/50`) and *case amplification*
  (`sim·|sim|^1.5`) to suppress noisy neighbors. Predictions come from the
  top-50 neighbors' mean-centered ratings, blended toward a bias fallback when
  the neighborhood is thin.
- **Stage 2** feeds the CF prediction into XGBoost as a feature, along with 9
  derivatives (neighbor count, confidence, residual vs. user and business
  baselines, confidence-weighted variants). The tree model learns *when* to
  trust CF — many neighbors, low disagreement with baselines — instead of a
  single global blend weight.
- **The stacking is out-of-fold:** training rows get CF predictions from a
  5-fold scheme where each row is predicted by a CF model that never saw it.
  Without this, the second stage would learn to trust CF far more than it
  deserves at test time.

## Model 2: Feature-rich (`feature_rich_recommender.py`)

The best-scoring one, built for a strict single-core runtime budget. It drops
the expensive CF stage (keeping a cheap bias proxy in the CF feature slots)
and spends the saved runtime on **feature engineering and leakage discipline**:

- **Leave-one-out training statistics** — for every training row, that row's
  own rating is subtracted from its user's and business's average, variance,
  min/max, and rating-distribution features. A user with three ratings has an
  average that is one-third the target itself; removing it gave a measurable
  RMSE gain.
- **Bayesian-smoothed rating estimates** at multiple shrinkage constants, with
  Yelp JSON profile stars as priors, so sparse users and businesses degrade
  gracefully toward informative priors instead of the global mean.
- **Rating-distribution features** — per user and business: fraction of
  ratings that are ≤2, ≥4, exactly 1, exactly 5, plus cross terms (a
  harsh-rater × polarizing-business interaction is very predictive of 1-star
  outcomes).
- **~50 extra content features** over Model 1: business attributes (alcohol,
  noise level, attire, wifi…), weekly opening hours, evening/weekend checkin
  ratios, tip/photo engagement, top-24 category indicators, and bias ×
  confidence × price cross features.
- **Output calibration** — the raw model is conservative at the extremes, so
  predictions are linearly expanded around the global mean (factor tuned on
  validation) before clamping.
- Hyperparameters were tuned with Bayesian optimization for this
  configuration.

Error distribution at RMSE 0.9775 (~142k validation pairs): 102,919 within 1
star, 32,034 within 1–2, 6,169 within 2–3, 922 above 3.

**Natural next step:** the two strengths are orthogonal — restoring the OOF CF
stack inside the feature-rich model (where runtime isn't capped) and re-tuning
should beat both.

## Usage

Requires the [Yelp Open Dataset](https://www.yelp.com/dataset) files
(`yelp_train.csv` with `user_id,business_id,stars`, plus `user.json`,
`business.json`, `checkin.json`, `tip.json`, `photo.json`) in one folder.

```bash
pip install -r requirements.txt

spark-submit feature_rich_recommender.py ./data ./data/yelp_val_in.csv output.csv
spark-submit cf_stacked_recommender.py   ./data ./data/yelp_val_in.csv output.csv
```

Output is a CSV with `user_id,business_id,prediction`.
