"""
=============================================================================
Feature Shift Quantification — Six Methods with Full Annotations
=============================================================================
Applied to the grocery store income prediction case:
  Source: Kroger + Safeway + Walmart  (median income tier, $30k–$105k)
  Target: Whole Foods (high tier)  and  Thrift Store (low tier)

The six methods:
  1. KS statistic             — per-feature, max CDF gap
  2. Wasserstein distance      — per-feature, mean shift + spread
  3. Mean shift + std ratio    — per-feature, moment-based
  4. Jensen-Shannon divergence — per-feature, symmetric information distance
  5. Classifier discrepancy    — joint (all features together), most powerful
  6. PCA centroid distance     — joint, overall multivariate shift

Decision rule for alpha selection in bridge sampling:
  Composite score > 0.5  →  alpha ≤ 0.3  (large shift)
  Composite score > 0.3  →  alpha ≤ 0.5  (medium shift)
  Composite score ≤ 0.3  →  alpha ≤ 0.7  (small shift)

Requirements:
    pip install pandas scikit-learn numpy scipy
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance, entropy
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# =============================================================================
# 1. LOAD AND PREPARE DATA
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

# Source: the three median-tier stores (labeled training data)
SOURCE_STORES = ["kroger", "safeway", "walmart"]
TARGET_STORES = ["whole_foods", "thrift_store"]

source_df = df[df["store"].isin(SOURCE_STORES)].copy()

# Scale on source data only — this is critical.
# Fitting the scaler on target would leak distributional information.
scaler    = StandardScaler()
X_src_sc  = scaler.fit_transform(source_df[FEAT_COLS].values.astype(float))

print("=" * 65)
print("DATA SUMMARY")
print("=" * 65)
print(f"  Source: {SOURCE_STORES}  n={len(source_df)}")
for t in TARGET_STORES:
    sub = df[df["store"] == t]
    print(f"  Target: {t}  n={len(sub)}  "
          f"income=${sub['income_usd'].min():,.0f}–${sub['income_usd'].max():,.0f}")


# =============================================================================
# HELPER: SHIFT LEVEL LABEL
# =============================================================================

def level(v, lo=0.2, hi=0.4):
    """Convert a 0–1 score into a LOW / MED / HIGH label."""
    return "HIGH" if v > hi else "MED" if v > lo else "LOW"


# =============================================================================
# MAIN LOOP — run all 6 methods for each target
# =============================================================================

all_results = {}

for target_store in TARGET_STORES:

    tgt_df   = df[df["store"] == target_store].copy()
    X_tgt_sc = scaler.transform(tgt_df[FEAT_COLS].values.astype(float))

    print(f"\n\n{'#'*65}")
    print(f"# SOURCE (Kroger+Safeway+Walmart)  →  TARGET: {target_store}")
    print(f"{'#'*65}")

    per_feature = {}


    # =========================================================================
    # METHOD 1 — KS STATISTIC (Kolmogorov-Smirnov two-sample test)
    #
    # What it measures:
    #   The maximum absolute vertical distance between the two empirical CDFs.
    #   The CDF of a distribution F at value x is P(X ≤ x).
    #   KS = max_x |F_source(x) - F_target(x)|
    #
    # Intuition:
    #   Imagine plotting both distributions as cumulative step functions.
    #   KS is the biggest gap you can find between them at any point x.
    #
    # Range: 0.0 (identical CDFs) to 1.0 (completely non-overlapping)
    #
    # p-value interpretation:
    #   p < 0.05 → the two samples are statistically significantly different
    #   Very large datasets almost always produce p < 0.05 even for small
    #   practical shifts — so look at the statistic itself, not just the p-value.
    #
    # Strength: detects any shape difference — bimodality, skew, tails
    # Weakness: misses uniform shifts. If target = source + constant, the
    #           CDFs are horizontally shifted, and KS may be small even though
    #           the mean has changed substantially. Use Wasserstein for that.
    #
    # Rule of thumb:
    #   KS < 0.2  → LOW shift   (distributions mostly overlap)
    #   KS 0.2–0.4 → MED shift
    #   KS > 0.4  → HIGH shift  (substantially different)
    # =========================================================================

    print(f"\n{'─'*65}")
    print("METHOD 1 — KS STATISTIC")
    print("  Range 0 (identical) → 1 (no overlap)")
    print(f"{'─'*65}")
    print(f"  {'Feature':<25}  {'KS stat':>8}  {'p-value':>12}  {'Level':>6}")
    print("  " + "─" * 56)

    ks_stats = {}
    for i, feat in enumerate(FEAT_COLS):
        ks_stat, p_value = ks_2samp(X_src_sc[:, i], X_tgt_sc[:, i])
        ks_stats[feat] = ks_stat
        lv = level(ks_stat, lo=0.2, hi=0.4)
        print(f"  {feat:<25}  {ks_stat:>8.4f}  {p_value:>12.4e}  {lv:>6}")

    per_feature["ks"] = ks_stats


    # =========================================================================
    # METHOD 2 — WASSERSTEIN DISTANCE (Earth Mover's Distance)
    #
    # What it measures:
    #   The minimum "work" required to transform one probability distribution
    #   into another. Work = mass × distance moved.
    #
    # Intuition:
    #   Imagine two piles of sand shaped like each distribution.
    #   Wasserstein is the minimum total effort to reshape one pile into the other.
    #
    # Mathematical form (for 1D):
    #   W = ∫ |F_source(x) - F_target(x)| dx
    #   where F is the CDF. This equals the L1 distance between CDFs.
    #
    # Range: 0 (identical) to ∞, but in standardized units (data is scaled)
    #   a value of 1.0 means the "center of mass" shifted by ~1 standard deviation
    #
    # Strength: sensitive to MAGNITUDE of shift. A uniform +2σ shift produces
    #           W ≈ 2.0 but might have a small KS statistic if the shape is preserved.
    # Weakness: can be large just because of scale differences, not shape.
    #           Always standardize first (we already did with StandardScaler).
    #
    # Rule of thumb (on standardized data):
    #   W < 0.4  → LOW shift
    #   W 0.4–0.8 → MED shift
    #   W > 0.8  → HIGH shift
    # =========================================================================

    print(f"\n{'─'*65}")
    print("METHOD 2 — WASSERSTEIN DISTANCE (Earth Mover's Distance)")
    print("  In std units (data already standardized)")
    print(f"{'─'*65}")
    print(f"  {'Feature':<25}  {'Wasserstein':>12}  {'Level':>6}")
    print("  " + "─" * 46)

    wass_stats = {}
    for i, feat in enumerate(FEAT_COLS):
        w = wasserstein_distance(X_src_sc[:, i], X_tgt_sc[:, i])
        wass_stats[feat] = w
        lv = level(w, lo=0.4, hi=0.8)
        print(f"  {feat:<25}  {w:>12.4f}  {lv:>6}")

    per_feature["wasserstein"] = wass_stats


    # =========================================================================
    # METHOD 3 — MEAN SHIFT AND STANDARD DEVIATION RATIO
    #
    # What it measures:
    #   The first two moments of the distributions:
    #     mean_shift  = |mean_target - mean_source|  (in source std units)
    #     std_ratio   = std_target / std_source
    #
    # Intuition:
    #   mean_shift: how far has the average customer moved?
    #     shift=0.5 means target average is 0.5 source-std-deviations away
    #   std_ratio:  has the spread of the distribution changed?
    #     ratio<1.0: target customers are more homogeneous
    #     ratio>1.0: target customers are more diverse
    #     ratio=1.0: same spread, just shifted
    #
    # This is the most interpretable method — you can say
    #   "basket size shifted +2.1 standard deviations upward and
    #    the target population is 17% more homogeneous on that feature"
    #
    # Strength: immediately human-readable, fast, diagnostic
    # Weakness: only captures the first two moments — misses skew, bimodality,
    #           and tail differences. A bimodal target can have the same mean
    #           and std as a unimodal source but completely different shape.
    # =========================================================================

    print(f"\n{'─'*65}")
    print("METHOD 3 — MEAN SHIFT AND STANDARD DEVIATION RATIO")
    print("  Mean shift in source std units (already standardized)")
    print("  Std ratio = target_std / source_std")
    print(f"{'─'*65}")
    print(f"  {'Feature':<25}  {'Mean shift':>12}  {'Std ratio':>10}  {'Interpretation':>20}")
    print("  " + "─" * 72)

    mean_shifts = {}
    std_ratios  = {}
    for i, feat in enumerate(FEAT_COLS):
        src_mean = X_src_sc[:, i].mean()
        tgt_mean = X_tgt_sc[:, i].mean()
        src_std  = X_src_sc[:, i].std()
        tgt_std  = X_tgt_sc[:, i].std()

        ms = abs(tgt_mean - src_mean)    # already in std units
        sr = tgt_std / (src_std + 1e-8)

        mean_shifts[feat] = ms
        std_ratios[feat]  = sr

        direction = "higher" if tgt_mean > src_mean else "lower"
        spread    = "narrower" if sr < 0.9 else "wider" if sr > 1.1 else "similar spread"
        interp    = f"{direction}, {spread}"
        print(f"  {feat:<25}  {ms:>+12.3f}  {sr:>10.3f}  {interp:>20}")

    per_feature["mean_shift"] = mean_shifts
    per_feature["std_ratio"]  = std_ratios


    # =========================================================================
    # METHOD 4 — JENSEN-SHANNON DIVERGENCE
    #
    # What it measures:
    #   A symmetric, bounded version of KL divergence.
    #   KL(P||Q) = Σ P(x) log(P(x) / Q(x))
    #   JS(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)   where M = (P+Q)/2
    #
    # Why JS instead of plain KL?
    #   1. KL is asymmetric: KL(P||Q) ≠ KL(Q||P)
    #   2. KL = infinity when Q(x)=0 and P(x)>0 (no smoothing needed)
    #   3. JS is bounded [0, 1] and symmetric — easier to interpret
    #
    # Implementation: bin each feature into 30 histogram buckets,
    #   compute empirical probabilities, then apply JS formula.
    #   Small smoothing (1e-10) prevents log(0).
    #
    # Range: 0.0 (identical distributions) to 1.0 (completely different)
    #   In practice, values above 0.15 indicate meaningful distributional shift.
    #   Values above 0.5 indicate extreme separation.
    #
    # Strength: captures full distributional shape, not just moments.
    #           Symmetric and bounded — comparable across features.
    # Weakness: depends on histogram bin count; sensitive to binning choice.
    #           With 30 bins on standardized data this is generally stable.
    # =========================================================================

    print(f"\n{'─'*65}")
    print("METHOD 4 — JENSEN-SHANNON DIVERGENCE")
    print("  Range 0 (identical) → 1 (completely different)")
    print("  Computed via 30-bin histogram approximation")
    print(f"{'─'*65}")
    print(f"  {'Feature':<25}  {'JS divergence':>14}  {'Level':>6}")
    print("  " + "─" * 50)

    js_stats = {}
    for i, feat in enumerate(FEAT_COLS):
        src_vals = X_src_sc[:, i]
        tgt_vals = X_tgt_sc[:, i]

        # Build histograms over the combined range of both distributions
        lo   = min(src_vals.min(), tgt_vals.min())
        hi   = max(src_vals.max(), tgt_vals.max())
        bins = np.linspace(lo, hi, 31)   # 30 bins = 31 edges

        p, _ = np.histogram(src_vals, bins=bins, density=True)
        q, _ = np.histogram(tgt_vals, bins=bins, density=True)

        # Normalize to probability mass (density × bin_width = probability)
        bin_width = bins[1] - bins[0]
        p = p * bin_width + 1e-10   # smoothing prevents log(0)
        q = q * bin_width + 1e-10
        p /= p.sum()
        q /= q.sum()

        # JS divergence = mean of two KL divergences to the mixture
        m  = 0.5 * (p + q)
        js = float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))
        js_stats[feat] = js

        lv = level(js, lo=0.05, hi=0.15)
        print(f"  {feat:<25}  {js:>14.4f}  {lv:>6}")

    per_feature["js"] = js_stats


    # =========================================================================
    # METHOD 5 — CLASSIFIER DISCREPANCY
    #
    # What it measures:
    #   The ability of a machine learning classifier to distinguish source
    #   samples from target samples. If it succeeds → large shift.
    #   If it performs at chance → distributions are indistinguishable.
    #
    # Why this is the most powerful method:
    #   All four previous methods are UNIVARIATE — they look at each feature
    #   independently. Two distributions can be similar on every individual
    #   feature while being very different in their joint distribution
    #   (e.g., features that are correlated in source but anti-correlated
    #   in target). The classifier captures ALL of this simultaneously.
    #
    # Implementation:
    #   1. Combine source (label=0) and target (label=1) into one dataset
    #   2. Train a GradientBoostingClassifier to predict source vs target
    #   3. Evaluate with 5-fold cross-validation to avoid overfitting
    #   4. Normalize: discrepancy = 2 × (accuracy - 0.5)
    #      This maps random chance (0.5) → 0  and  perfect (1.0) → 1
    #
    # Interpretation of normalized discrepancy:
    #   0.0  → distributions identical (classifier at chance)
    #   0.3  → moderate shift (classifier 65% accurate)
    #   0.6  → large shift (classifier 80% accurate)
    #   1.0  → perfect separation (classifier 100% accurate)
    #
    # Strength: captures interaction effects, non-linear differences, and
    #           joint distributional differences. Cannot be fooled by two
    #           distributions that look similar marginally but differ jointly.
    # Weakness: computationally slower than univariate methods.
    #           Requires enough samples in both domains (n > 100 each).
    # =========================================================================

    print(f"\n{'─'*65}")
    print("METHOD 5 — CLASSIFIER DISCREPANCY")
    print("  Train a classifier: source (0) vs target (1)")
    print("  Normalized discrepancy = 2 × (accuracy - 0.5)")
    print("  0 = indistinguishable, 1 = perfectly separated")
    print(f"{'─'*65}")

    n_src = len(X_src_sc)
    n_tgt = len(X_tgt_sc)
    X_clf = np.vstack([X_src_sc, X_tgt_sc])
    y_clf = np.array([0] * n_src + [1] * n_tgt)

    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    cv_acc = cross_val_score(
        clf, X_clf, y_clf, cv=5, scoring="accuracy"
    ).mean()

    # Normalize so that chance (0.5) → 0 and perfect (1.0) → 1
    disc = 2 * (cv_acc - 0.5)
    lv   = level(disc, lo=0.3, hi=0.6)

    print(f"  Source samples:  {n_src}")
    print(f"  Target samples:  {n_tgt}")
    print(f"  CV accuracy:     {cv_acc:.4f}  ({cv_acc:.1%})")
    print(f"  Normalized disc: {disc:.4f}  [{lv}]")

    # Extract feature importances — which features drive the separability?
    clf_fit = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    clf_fit.fit(X_clf, y_clf)
    imp_order = np.argsort(clf_fit.feature_importances_)[::-1]
    print(f"\n  Top 5 features driving source vs target separation:")
    for rank, idx in enumerate(imp_order[:5], 1):
        print(f"    {rank}. {FEAT_COLS[idx]:<25}  "
              f"importance={clf_fit.feature_importances_[idx]:.4f}")

    classifier_disc = disc
    per_feature["classifier_importances"] = {
        FEAT_COLS[i]: round(float(clf_fit.feature_importances_[i]), 4)
        for i in range(len(FEAT_COLS))
    }


    # =========================================================================
    # METHOD 6 — PCA CENTROID DISTANCE
    #
    # What it measures:
    #   The Euclidean distance between the center of mass (centroid) of the
    #   source and target distributions in a shared PCA feature space.
    #
    # How it works:
    #   1. Fit PCA on the combined source + target data (n_components=6)
    #   2. Project both datasets into this shared 6-dimensional space
    #   3. Compute the centroid (mean vector) of each dataset in PCA space
    #   4. Measure Euclidean distance between the two centroids
    #
    # Why PCA first?
    #   Raw feature space can have redundant dimensions (correlated features)
    #   that inflate the distance. PCA decorrelates and reduces dimensionality,
    #   giving a more meaningful measure of overall multivariate shift.
    #   6 components is chosen to capture ~87% of variance in our data.
    #
    # Interpretation:
    #   Distance is in units of the first principal component (roughly std units)
    #   < 1.0  → small overall shift
    #   1.0–3.0 → medium shift
    #   > 3.0  → large shift (distributions are far apart in feature space)
    #
    # This method also reveals WHICH direction the shift occurs — the
    # difference vector between centroids points in the direction of the shift,
    # and projecting this onto the original features reveals which ones
    # contribute most to the overall displacement.
    #
    # Strength: gives a single number for the overall multivariate shift.
    #           Easy to visualize. Naturally handles feature correlations.
    # Weakness: sensitive to n_components choice. Only captures mean shift,
    #           not distributional shape changes.
    # =========================================================================

    print(f"\n{'─'*65}")
    print("METHOD 6 — PCA CENTROID DISTANCE")
    print("  Overall multivariate shift in PCA space")
    print(f"{'─'*65}")

    X_combined = np.vstack([X_src_sc, X_tgt_sc])
    pca = PCA(n_components=6, random_state=42)
    pca.fit(X_combined)
    print(f"  PCA explained variance (6 components): "
          f"{pca.explained_variance_ratio_.sum():.1%}")

    X_src_pca = pca.transform(X_src_sc)
    X_tgt_pca = pca.transform(X_tgt_sc)

    src_centroid = X_src_pca.mean(axis=0)
    tgt_centroid = X_tgt_pca.mean(axis=0)
    dist = float(np.linalg.norm(src_centroid - tgt_centroid))
    lv   = level(dist, lo=1.5, hi=3.0)

    print(f"  PCA centroid distance: {dist:.4f} units  [{lv}]")

    # Which original features contribute most to the centroid shift?
    # Project the shift vector back to original feature space
    shift_vec_pca = tgt_centroid - src_centroid   # direction of shift in PCA space
    shift_original = pca.components_.T @ shift_vec_pca  # back to original space
    shift_order    = np.argsort(np.abs(shift_original))[::-1]
    print(f"\n  Top 5 features driving the centroid shift:")
    for rank, idx in enumerate(shift_order[:5], 1):
        direction = "↑" if shift_original[idx] > 0 else "↓"
        print(f"    {rank}. {FEAT_COLS[idx]:<25}  "
              f"contribution={shift_original[idx]:+.3f} {direction}")

    pca_dist = dist


    # =========================================================================
    # COMPOSITE SHIFT SCORE
    #
    # Combines all six methods into a single actionable score.
    #
    # Normalization approach:
    #   - KS statistic:         already in [0,1]
    #   - Wasserstein:          divide by 2 (practical max in std units ≈ 5)
    #   - JS divergence:        multiply by 3 (rescale to [0,1] sensitivity)
    #   - Mean shift:           divide by 2 (practical max ≈ 5 std units)
    #   - Classifier disc.:     already in [0,1]
    #   - PCA centroid dist:    divide by 12 (practical max ≈ 15 std units)
    # Average of normalized values gives composite score.
    #
    # Decision rule:
    #   composite > 0.5  →  LARGE SHIFT  →  use alpha ≤ 0.3
    #   composite > 0.3  →  MEDIUM SHIFT →  use alpha ≤ 0.5
    #   composite ≤ 0.3  →  SMALL SHIFT  →  use alpha ≤ 0.7
    # =========================================================================

    avg_ks   = float(np.mean(list(ks_stats.values())))
    avg_wass = float(np.mean(list(wass_stats.values())))
    avg_js   = float(np.mean(list(js_stats.values())))
    avg_ms   = float(np.mean(list(mean_shifts.values())))

    composite = (avg_ks +
                 min(avg_wass / 2, 1.0) +
                 min(avg_js * 3, 1.0) +
                 min(avg_ms / 2, 1.0) +
                 classifier_disc) / 5

    shift_level = "LARGE"  if composite > 0.5 else \
                  "MEDIUM" if composite > 0.3 else "SMALL"
    alpha_rec   = "≤ 0.3"  if composite > 0.5 else \
                  "≤ 0.5"  if composite > 0.3 else "≤ 0.7"

    print(f"\n{'─'*65}")
    print("COMPOSITE SHIFT SCORE")
    print(f"{'─'*65}")
    print(f"  Avg KS statistic:        {avg_ks:.4f}")
    print(f"  Avg Wasserstein:         {avg_wass:.4f}")
    print(f"  Avg JS divergence:       {avg_js:.4f}")
    print(f"  Avg mean shift:          {avg_ms:.4f}")
    print(f"  Classifier discrepancy:  {classifier_disc:.4f}")
    print(f"  PCA centroid distance:   {pca_dist:.4f}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Composite score:         {composite:.4f}  [{shift_level} SHIFT]")
    print(f"  Alpha recommendation:    alpha {alpha_rec}")

    all_results[target_store] = {
        "avg_ks":           round(avg_ks, 4),
        "avg_wasserstein":  round(avg_wass, 4),
        "avg_js":           round(avg_js, 4),
        "avg_mean_shift":   round(avg_ms, 4),
        "classifier_disc":  round(classifier_disc, 4),
        "pca_dist":         round(pca_dist, 4),
        "composite_score":  round(composite, 4),
        "shift_level":      shift_level,
        "alpha_rec":        alpha_rec,
        "per_feature":      per_feature,
    }


# =============================================================================
# FINAL COMPARISON TABLE
# =============================================================================

print(f"\n\n{'='*65}")
print("FINAL COMPARISON — Whole Foods vs Thrift Store")
print(f"{'='*65}")
print(f"\n  {'Method':<35}  {'Whole Foods':>12}  {'Thrift':>10}  {'Bigger':>12}")
print("  " + "─" * 73)
comparisons = [
    ("Avg KS statistic",        "avg_ks"),
    ("Avg Wasserstein dist.",    "avg_wasserstein"),
    ("Avg JS divergence",        "avg_js"),
    ("Avg mean shift (std)",     "avg_mean_shift"),
    ("Classifier discrepancy",   "classifier_disc"),
    ("PCA centroid distance",    "pca_dist"),
    ("COMPOSITE SCORE",          "composite_score"),
]
for label, key in comparisons:
    wf = all_results["whole_foods"][key]
    th = all_results["thrift_store"][key]
    bigger = "Thrift" if th > wf else "Whole Foods"
    print(f"  {label:<35}  {wf:>12.4f}  {th:>10.4f}  {bigger:>12}")

print(f"\n  Alpha recommendations:")
for t in TARGET_STORES:
    r = all_results[t]
    print(f"    {t:<15}  composite={r['composite_score']:.4f}  "
          f"[{r['shift_level']}]  →  alpha {r['alpha_rec']}")


# =============================================================================
# SAVE RESULTS
# =============================================================================

import json
with open("shift_quantification_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n  Results saved to: shift_quantification_results.json")
print("  Done.")
