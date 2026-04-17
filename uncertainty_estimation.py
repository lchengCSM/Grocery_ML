"""
=============================================================================
Uncertainty / Prediction Error Estimation — Four Methods
=============================================================================
Applied to grocery store income prediction under domain shift.
Source: Kroger + Safeway + Walmart  (median tier, $30k–$105k)
Target: all 5 stores in turn (LOSO-CV)

Four methods implemented, each answering a different question:

  M1 — Source CV residuals
       "What error do we typically get on data we have seen?"
       → Point MAE estimate, no domain awareness

  M2 — Shift-adjusted interval
       "How much should we inflate M1 given the feature shift magnitude?"
       → Point MAE estimate, KS-based inflation factor

  M3 — Conformal prediction
       "What ±interval contains the true value X% of the time?"
       → Coverage interval with empirical guarantee (degrades under shift)

  M4 — Density-ratio weighted residuals
       "What error would we get if source data were reweighted to look
        like the target distribution?"
       → Point MAE estimate, importance-weighted

Key finding:
  Medium-shift stores (Kroger, Safeway, Walmart): M4 within $3–5k of actual
  High-shift stores (Whole Foods, Thrift): only M3 is useful — gives ±interval
  M2 (shift-adjusted) over-inflates badly for high KS values

Requirements:
    pip install pandas scikit-learn numpy scipy
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr, ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# =============================================================================
# 1. LOAD DATA AND SHARED SETUP
# =============================================================================

df = pd.read_csv("grocery_all_stores.csv")

le = LabelEncoder()
le.fit(df["segment"])
df["segment_enc"] = le.transform(df["segment"])

FEAT_COLS = [
    "age", "visits_per_month", "avg_basket_usd", "monthly_spend_usd",
    "grocery_pct", "electronics_pct", "apparel_pct", "home_pct",
    "private_label_pct", "online_orders_pct", "coupon_usage_pct",
    "loyalty_score", "segment_enc",
]
STORES = ["whole_foods", "kroger", "safeway", "walmart", "thrift_store"]
TRUE_RANGES = {
    "whole_foods":  (85_000, 160_000),
    "kroger":       (55_000, 105_000),
    "safeway":      (45_000,  95_000),
    "walmart":      (30_000,  80_000),
    "thrift_store": (10_000,  45_000),
}
PROXY_IDX = [2, 9]   # avg_basket_usd, online_orders_pct
BEST_ALPHAS = [0.1, 0.3, 0.5, 0.7]   # tuned in previous session

# Build shared PCA reference library (used for range estimation)
sc_all    = StandardScaler()
X_all_sc  = sc_all.fit_transform(df[FEAT_COLS].values.astype(float))
pca_ref   = PCA(n_components=6, random_state=42)
X_all_pca = pca_ref.fit_transform(X_all_sc)

reference_library = {}
for s in STORES:
    mask = df["store"] == s
    reference_library[s] = {
        "centroid":    X_all_pca[mask].mean(axis=0),
        "income_mean": np.mean(TRUE_RANGES[s]),
    }


# =============================================================================
# 2. SHARED HELPER FUNCTIONS
# =============================================================================

def make_bridge(X_src_sc, X_tgt_sc, y_src_rank,
                alphas=BEST_ALPHAS, n_total=400, random_state=42):
    """Generate bridge samples between source and target (no target labels)."""
    rng = np.random.RandomState(random_state)
    X_bridge, y_bridge = [], []
    n_each = max(1, n_total // len(alphas))
    for alpha in alphas:
        for _ in range(n_each):
            i = rng.randint(len(X_src_sc))
            j = rng.randint(len(X_tgt_sc))
            X_bridge.append((1 - alpha) * X_src_sc[i] + alpha * X_tgt_sc[j])
            y_bridge.append((1 - alpha) * y_src_rank[i] + alpha * 0.5)
    return np.array(X_bridge), np.array(y_bridge)


def estimate_income_range(X_src_sc, X_tgt_sc, y_src, target_store, ref_stores):
    """Method B range estimate (proxy regression + nearest-retailer blend)."""
    proxy_reg = LinearRegression().fit(X_src_sc[:, PROXY_IDX], y_src)
    src_mean  = proxy_reg.predict(X_src_sc[:, PROXY_IDX]).mean()
    tgt_mean  = proxy_reg.predict(X_tgt_sc[:, PROXY_IDX]).mean()
    gap       = tgt_mean - src_mean
    span      = (y_src.max() - y_src.min()) / 2 * 1.1

    tgt_centroid = X_all_pca[df["store"] == target_store].mean(axis=0).reshape(1, -1)
    distances    = {s: cdist(tgt_centroid,
                              reference_library[s]["centroid"].reshape(1, -1),
                              "euclidean")[0, 0]
                    for s in ref_stores}
    nn           = min(distances, key=distances.get)
    blended_mean = 0.4 * reference_library[nn]["income_mean"] + 0.6 * (src_mean + gap)
    est_lo       = max(0, round((blended_mean - span) / 1_000) * 1_000)
    est_hi       = round((blended_mean + span) / 1_000) * 1_000
    return est_lo, est_hi


def build_rank_model(X_train, y_train_rank, X_bridge, y_bridge):
    """Train Ridge regression on rank labels + bridge samples."""
    X_univ = np.vstack([X_train, X_bridge])
    y_univ = np.concatenate([y_train_rank, y_bridge])
    model  = Ridge(alpha=10.0)
    model.fit(X_univ, y_univ)
    return model


def predict_income(model, X_test, lo, hi):
    """Predict income from rank model and rescale to target range."""
    ranks = np.clip(model.predict(X_test), 0.0, 1.0)
    return lo + ranks * (hi - lo)


# =============================================================================
# 3. METHOD 1 — SOURCE CROSS-VALIDATION RESIDUALS
#
# Core idea:
#   Run K-fold CV on the source (labeled) data.
#   Collect |predicted - actual| residuals on held-out folds.
#   The mean of these residuals is the predicted MAE for any target.
#
# Why it works for median-tier stores:
#   If source and target have similar distributions, source CV error is a
#   reasonable proxy for target error. The model's ability to predict
#   income within the source range transfers when target is in that range.
#
# Why it fails for high-shift targets:
#   Source CV residuals measure performance within the source income range
#   ($30k–$105k). For Whole Foods ($85k–$160k), the model is predicting
#   in a partly new range. For Thrift ($10k–$45k), the range is almost
#   entirely outside training. Source CV does not capture these errors.
#
# Parameters:
#   k_folds  : number of CV folds (5 is standard)
#   lo, hi   : income range used for rescaling (same as final model)
#              using the same range ensures CV error is on the same scale
# =============================================================================

def method1_source_cv(X_src_sc, y_src, lo, hi, k_folds=5, random_state=42):
    """
    Estimate prediction error via K-fold CV on source data.

    Returns
    -------
    est_mae  : float — mean of absolute residuals across all folds
    est_std  : float — standard deviation of absolute residuals
    residuals: array — all fold-level absolute residuals
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    residuals = []

    for train_idx, val_idx in kf.split(X_src_sc):
        X_tr, X_val = X_src_sc[train_idx], X_src_sc[val_idx]
        y_tr, y_val = y_src[train_idx], y_src[val_idx]

        # Convert to ranks within this fold's training set
        y_rank_tr = (rankdata(y_tr) - 1) / (len(y_tr) - 1)

        # Train rank model (no bridge — no target available in CV)
        m = Ridge(alpha=10.0)
        m.fit(X_tr, y_rank_tr)

        # Predict on validation fold
        preds_val = lo + np.clip(m.predict(X_val), 0, 1) * (hi - lo)

        # Collect absolute residuals
        residuals.extend(np.abs(y_val - preds_val))

    residuals = np.array(residuals)
    return residuals.mean(), residuals.std(), residuals


# =============================================================================
# 4. METHOD 2 — SHIFT-ADJUSTED INTERVAL
#
# Core idea:
#   The source CV MAE underestimates target error when there is domain shift.
#   Adjust upward by a factor proportional to the feature shift magnitude.
#
# The adjustment factor:
#   factor = 1 + avg_KS × k
#   where avg_KS is the mean KS statistic across all features
#   and k is a scaling constant (empirically set to 2.0 here)
#
#   Intuition: a KS of 0 means no shift (factor=1, no inflation).
#              a KS of 0.5 means substantial shift (factor=2, double the error).
#              a KS of 1.0 means complete non-overlap (factor=3).
#
# Caveat:
#   The k=2.0 scaling was not tuned on this dataset — it is a heuristic.
#   In practice, this method over-inflates at high KS (WF: $71k vs actual $18k).
#   It is useful as an upper bound on expected error, not as a precise estimate.
#   A better version would tune k on source leave-one-store-out CV.
# =============================================================================

def method2_shift_adjusted(est_mae_m1, X_src_sc, X_tgt_sc, k=2.0):
    """
    Inflate source CV MAE by feature shift magnitude.

    Parameters
    ----------
    est_mae_m1 : float  — source CV MAE from Method 1
    X_src_sc   : array  — standardized source features
    X_tgt_sc   : array  — standardized target features
    k          : float  — scaling constant (default 2.0, treat as heuristic)

    Returns
    -------
    adjusted_mae : float — inflated MAE estimate
    avg_ks       : float — mean KS statistic (shift indicator)
    factor       : float — the inflation factor applied
    """
    # Compute KS statistic per feature, then average
    ks_stats = [ks_2samp(X_src_sc[:, i], X_tgt_sc[:, i])[0]
                for i in range(X_src_sc.shape[1])]
    avg_ks   = np.mean(ks_stats)

    # Inflation factor: 1 (no shift) to 1+k (complete non-overlap)
    factor = 1 + avg_ks * k

    return est_mae_m1 * factor, avg_ks, factor


# =============================================================================
# 5. METHOD 3 — CONFORMAL PREDICTION
#
# Core idea:
#   Conformal prediction gives a formal coverage guarantee:
#   "The true value will fall in the interval [pred - q, pred + q]
#    at least (1-alpha)% of the time."
#
#   This guarantee holds EXACTLY when calibration and test data are
#   exchangeable (IID from the same distribution). Under domain shift,
#   the guarantee is approximate — but it is still the most principled
#   uncertainty estimate available without target labels.
#
# Algorithm (split conformal prediction):
#   1. Split source into training (80%) and calibration (20%)
#   2. Train the rank model on the training split + bridge samples
#      (using the target features for bridge generation)
#   3. For each calibration sample, compute the nonconformity score:
#        score_i = |y_cal_i - pred_cal_i|
#      This measures how "nonconforming" each prediction is
#   4. Find the empirical (1-alpha) quantile of calibration scores:
#        q = quantile(scores, (1-alpha) * (1 + 1/n_cal))
#      The (1+1/n_cal) correction gives finite-sample validity
#   5. For each target prediction:
#        interval = [pred - q, pred + q]
#
# Coverage guarantee:
#   If the calibration distribution = test distribution (IID):
#   P(y_test in [pred - q, pred + q]) >= 1 - alpha
#
#   Under covariate shift (our case), the actual coverage may differ.
#   Large shift → coverage may be lower (interval too narrow)
#                 OR higher (interval too wide from large source residuals)
#   The Thrift Store example (64.8% vs target 90%) illustrates this.
#
# Parameters:
#   coverage_target : desired coverage level, e.g. 0.90 for 90% intervals
#   cal_fraction    : fraction of source data used for calibration (default 0.2)
# =============================================================================

def method3_conformal(X_src_sc, y_src, X_tgt_sc,
                       lo, hi,
                       coverage_target=0.90,
                       cal_fraction=0.20,
                       random_state=42):
    """
    Compute conformal prediction intervals for target predictions.

    Parameters
    ----------
    X_src_sc        : standardized source features
    y_src           : source income labels
    X_tgt_sc        : standardized target features
    lo, hi          : income range for rank rescaling
    coverage_target : desired coverage level (e.g. 0.90)
    cal_fraction    : fraction of source used for calibration

    Returns
    -------
    pred_intervals  : array shape (n_tgt, 2) — [lower, upper] for each target
    q               : float — conformal quantile (half-width of all intervals)
    cal_residuals   : array — calibration nonconformity scores
    model           : the trained Ridge model
    """
    rng     = np.random.RandomState(random_state)
    n       = len(X_src_sc)
    n_cal   = int(n * cal_fraction)
    n_train = n - n_cal

    # Split source: first n_train for training, last n_cal for calibration
    # (in practice shuffle first)
    idx = rng.permutation(n)
    tr_idx, cal_idx = idx[:n_train], idx[n_train:]

    X_tr,  X_cal  = X_src_sc[tr_idx],  X_src_sc[cal_idx]
    y_tr,  y_cal  = y_src[tr_idx],     y_src[cal_idx]

    # Train rank model on training split
    y_rank_tr = (rankdata(y_tr) - 1) / (len(y_tr) - 1)
    X_b, y_b  = make_bridge(X_tr, X_tgt_sc, y_rank_tr)
    model     = build_rank_model(X_tr, y_rank_tr, X_b, y_b)

    # Compute nonconformity scores on calibration set
    pred_cal       = predict_income(model, X_cal, lo, hi)
    cal_residuals  = np.abs(y_cal - pred_cal)

    # Conformal quantile with finite-sample correction
    # ceil((1-alpha) * (1 + 1/n_cal)) ensures conservative coverage
    level  = np.ceil((1 - coverage_target) * 0 + coverage_target *
                     (1 + 1 / n_cal)) / 1
    # Standard formula: quantile at level = (1-alpha)(1+1/n)
    q_level = (1 - (1 - coverage_target)) * (1 + 1 / n_cal)
    q_level = min(q_level, 1.0)
    q       = float(np.quantile(cal_residuals, q_level))

    # Build prediction intervals for each target sample
    pred_tgt       = predict_income(model, X_tgt_sc, lo, hi)
    pred_intervals = np.column_stack([pred_tgt - q, pred_tgt + q])

    return pred_intervals, q, cal_residuals, model, pred_tgt


# =============================================================================
# 6. METHOD 4 — DENSITY-RATIO WEIGHTED RESIDUALS
#
# Core idea:
#   Source CV residuals give equal weight to all source samples.
#   But some source samples look like the target (small shift) while
#   others look nothing like the target (large shift).
#   We should weight source samples by how representative they are
#   of the target distribution.
#
# The importance weight for each source sample x:
#   w(x) = p_target(x) / p_source(x)
#
#   If p_target(x) >> p_source(x): this x is more likely in target → high weight
#   If p_target(x) << p_source(x): this x is rare in target → low weight
#
# Estimation:
#   We cannot compute p_target and p_source analytically, but we can
#   estimate them using Kernel Density Estimation (KDE) on each dataset.
#   log w(x) = log p_target(x) - log p_source(x)
#            = kde_target.score_samples(x) - kde_source.score_samples(x)
#
# Weighted MAE estimate:
#   The weighted mean of calibration residuals is then:
#   est_MAE = Σ w(x_i) × |y_i - pred_i| / Σ w(x_i)
#   which up-weights residuals from source samples that look like target.
#
# Why it fails for large-shift targets:
#   When p_target and p_source barely overlap (KS ≈ 1.0), almost all
#   source samples have near-zero weight. The weighted mean becomes
#   numerically unstable — we are averaging only the extreme edge of
#   the source distribution, which may not represent the target at all.
#
# Parameters:
#   bandwidth : KDE bandwidth. Larger = smoother density estimate.
#               1.5 works well for standardized data with 13 features.
#               Too small → noisy weights. Too large → uniform weights.
# =============================================================================

def method4_density_ratio(X_src_sc, y_src, X_tgt_sc, X_cal, y_cal,
                           lo, hi, model, bandwidth=1.5):
    """
    Estimate prediction MAE using density-ratio reweighted source residuals.

    Parameters
    ----------
    X_src_sc  : standardized source features (full source)
    y_src     : source income labels
    X_tgt_sc  : standardized target features
    X_cal     : calibration subset of source features
    y_cal     : calibration income labels
    lo, hi    : income range for rescaling
    model     : trained rank model
    bandwidth : KDE bandwidth

    Returns
    -------
    weighted_mae : float — density-ratio weighted MAE estimate
    weights      : array — importance weights for each calibration sample
    """
    # Fit KDE on target distribution (features only — no labels)
    kde_tgt = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde_tgt.fit(X_tgt_sc)

    # Fit KDE on source distribution
    kde_src = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde_src.fit(X_src_sc)

    # Compute log importance weights for each calibration sample
    # Use log-space for numerical stability, then exponentiate
    log_w_cal = (kde_tgt.score_samples(X_cal) -
                 kde_src.score_samples(X_cal))

    # Shift by max for numerical stability before exponentiating
    log_w_cal -= log_w_cal.max()
    weights    = np.exp(log_w_cal)

    # Clip very small weights to avoid numerical collapse
    # and normalize so weights sum to n_cal (preserve scale)
    weights = np.clip(weights, 0.01, None)
    weights = weights / weights.mean()

    # Compute calibration residuals under current model
    pred_cal     = predict_income(model, X_cal, lo, hi)
    cal_residuals = np.abs(y_cal - pred_cal)

    # Weighted MAE
    weighted_mae = float(np.average(cal_residuals, weights=weights))

    return weighted_mae, weights


# =============================================================================
# 7. MAIN LOOP — LEAVE-ONE-STORE-OUT CV
# =============================================================================

print("=" * 72)
print("UNCERTAINTY ESTIMATION — LEAVE-ONE-STORE-OUT CV")
print("=" * 72)

all_results = {}

for target_store in STORES:

    # ── Split source / target ─────────────────────────────────────────────
    source_df = df[df["store"] != target_store].copy()
    target_df = df[df["store"] == target_store].copy()

    X_src_raw = source_df[FEAT_COLS].values.astype(float)
    X_tgt_raw = target_df[FEAT_COLS].values.astype(float)
    y_src     = source_df["income_usd"].values
    y_tgt     = target_df["income_usd"].values   # held out for evaluation only

    # Scale on source only
    scaler   = StandardScaler()
    X_src_sc = scaler.fit_transform(X_src_raw)
    X_tgt_sc = scaler.transform(X_tgt_raw)

    # Estimate income range
    ref_stores = [s for s in STORES if s != target_store]
    est_lo, est_hi = estimate_income_range(
        X_src_sc, X_tgt_sc, y_src, target_store, ref_stores
    )

    # Build final prediction model
    y_src_rank  = (rankdata(y_src) - 1) / (len(y_src) - 1)
    X_b, y_b    = make_bridge(X_src_sc, X_tgt_sc, y_src_rank)
    final_model = build_rank_model(X_src_sc, y_src_rank, X_b, y_b)
    final_preds = predict_income(final_model, X_tgt_sc, est_lo, est_hi)
    actual_mae  = mean_absolute_error(y_tgt, final_preds)
    actual_errs = np.abs(y_tgt - final_preds)

    print(f"\n{'─'*65}")
    print(f"  Target: {target_store}  |  actual MAE = ${actual_mae:,.0f}  |  "
          f"range est: ${est_lo:,}–${est_hi:,}")
    print(f"{'─'*65}")

    # ── Method 1: Source CV ───────────────────────────────────────────────
    m1_mae, m1_std, m1_residuals = method1_source_cv(
        X_src_sc, y_src, est_lo, est_hi, k_folds=5
    )
    m1_err = abs(m1_mae - actual_mae)
    print(f"  M1 source CV:        est=${m1_mae:>8,.0f}  actual=${actual_mae:>8,.0f}  "
          f"err=${m1_err:>7,.0f}")

    # ── Method 2: Shift-adjusted ──────────────────────────────────────────
    m2_mae, avg_ks, factor = method2_shift_adjusted(
        m1_mae, X_src_sc, X_tgt_sc, k=2.0
    )
    m2_err = abs(m2_mae - actual_mae)
    print(f"  M2 shift-adjusted:   est=${m2_mae:>8,.0f}  actual=${actual_mae:>8,.0f}  "
          f"err=${m2_err:>7,.0f}  (KS={avg_ks:.3f}, ×{factor:.2f})")

    # ── Method 3: Conformal prediction ───────────────────────────────────
    (intervals, q, cal_residuals,
     conf_model, conf_preds) = method3_conformal(
        X_src_sc, y_src, X_tgt_sc,
        est_lo, est_hi, coverage_target=0.90, cal_fraction=0.20
    )
    actual_coverage = float((actual_errs <= q).mean())
    print(f"  M3 conformal (90%):  ±${q:>8,.0f}  "
          f"actual coverage={actual_coverage:.1%}  "
          f"(target=90.0%)")

    # ── Method 4: Density-ratio ───────────────────────────────────────────
    # Use the same calibration split as Method 3 for consistency
    rng   = np.random.RandomState(42)
    n_cal = int(0.2 * len(X_src_sc))
    idx   = rng.permutation(len(X_src_sc))
    X_cal, y_cal = X_src_sc[idx[len(X_src_sc)-n_cal:]], y_src[idx[len(X_src_sc)-n_cal:]]

    m4_mae, m4_weights = method4_density_ratio(
        X_src_sc, y_src, X_tgt_sc, X_cal, y_cal,
        est_lo, est_hi, conf_model, bandwidth=1.5
    )
    m4_err = abs(m4_mae - actual_mae)
    print(f"  M4 density-ratio:    est=${m4_mae:>8,.0f}  actual=${actual_mae:>8,.0f}  "
          f"err=${m4_err:>7,.0f}  "
          f"(eff. n={1/np.sum((m4_weights/m4_weights.sum())**2):.0f}/{n_cal})")

    all_results[target_store] = {
        "actual_mae":       round(actual_mae),
        "m1_est":           round(m1_mae),
        "m1_std":           round(m1_std),
        "m1_err":           round(m1_err),
        "m2_est":           round(m2_mae),
        "m2_err":           round(m2_err),
        "avg_ks":           round(avg_ks, 4),
        "shift_factor":     round(factor, 3),
        "m3_q":             round(q),
        "m3_coverage":      round(actual_coverage, 3),
        "m4_est":           round(m4_mae),
        "m4_err":           round(m4_err),
    }


# =============================================================================
# 8. SUMMARY AND DECISION GUIDE
# =============================================================================

print(f"\n\n{'='*72}")
print("SUMMARY")
print(f"{'='*72}")
print(f"\n  {'Store':<15}  {'Actual':>8}  {'M1 err':>8}  {'M2 err':>8}  "
      f"{'M4 err':>8}  {'M3 ±width':>10}  {'M3 cov':>8}")
print("  " + "─" * 68)
for s in STORES:
    r = all_results[s]
    print(f"  {s:<15}  ${r['actual_mae']:>7,}  ${r['m1_err']:>7,}  "
          f"${r['m2_err']:>7,}  ${r['m4_err']:>7,}  "
          f"±${r['m3_q']:>7,}  {r['m3_coverage']:>7.1%}")

print(f"""
DECISION GUIDE — which method to use for your next project
──────────────────────────────────────────────────────────
Step 1: Measure feature shift (avg KS statistic or classifier discrepancy)

  avg_KS < 0.2  (small shift):
    Use M1 (source CV) for a quick point estimate.
    Use M3 (conformal) if you need a coverage interval.

  avg_KS 0.2–0.5 (medium shift):
    Use M4 (density-ratio) for best point estimate.
    Use M3 (conformal) for coverage interval.
    M2 gives an upper bound — treat it as worst-case.

  avg_KS > 0.5  (large shift):
    Do not use M1 or M4 as point estimates — unreliable.
    Use M3 (conformal ±interval) and report uncertainty explicitly.
    Communicate: "we estimate error of ±$X with Y% empirical coverage"
    rather than "we estimate MAE of $X."

Step 2: Check conformal coverage against target
  If actual coverage << 90%: the shift is causing the interval to be
    too narrow — inflate q by (0.90 / actual_coverage) as a heuristic.
  If actual coverage >> 90%: the interval is conservative — you can
    tighten it by using a higher coverage_target (e.g. 0.95).
""")


# =============================================================================
# 9. SAVE RESULTS
# =============================================================================

import json
with open("uncertainty_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("  Results saved to: uncertainty_results.json")
print("  Done.")
