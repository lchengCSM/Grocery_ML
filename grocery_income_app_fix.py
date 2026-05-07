"""
Grocery Store Income Prediction — Streamlit ML Experiment App
Runs the full S10 rank model pipeline interactively.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import rankdata, spearmanr, ks_2samp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grocery Income Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
STORE_COLORS = {
    "whole_foods":  "#185FA5",
    "kroger":       "#1D9E75",
    "safeway":      "#0F6E56",
    "walmart":      "#854F0B",
    "thrift_store": "#A32D2D",
}
STORE_LABELS = {
    "whole_foods":  "Whole Foods",
    "kroger":       "Kroger",
    "safeway":      "Safeway",
    "walmart":      "Walmart",
    "thrift_store": "Thrift Store",
}
TIER_LABELS = {
    "whole_foods": "High", "kroger": "Median",
    "safeway": "Median", "walmart": "Median", "thrift_store": "Low",
}
FEAT_COLS = ["age","visits_per_month","avg_basket_usd","monthly_spend_usd",
             "grocery_pct","electronics_pct","apparel_pct","home_pct",
             "private_label_pct","online_orders_pct","coupon_usage_pct",
             "loyalty_score","segment_enc"]
FEAT_NICE = ["Age","Visits/mo","Basket $","Monthly spend","Grocery %",
             "Electronics %","Apparel %","Home %","Private label %",
             "Online orders %","Coupon use %","Loyalty score","Segment"]
PROXY_IDX  = [2, 9]
STORES     = list(STORE_LABELS.keys())

# Custom (6th) store — label and colour for the user-defined store
CUSTOM_KEY   = "custom_store"
CUSTOM_LABEL = "My Store (custom)"
CUSTOM_COLOR = "#6B2D8B"   # purple

def rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert 6-digit hex + alpha float to Plotly-safe rgba() string.
    Use this everywhere instead of appending hex digits (#RRGGBBAA).
    rgba('#185FA5', 0.15)  →  'rgba(24,95,165,0.15)'
    """
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

TRUE_RANGES = {
    "whole_foods":  (85_000, 160_000),
    "kroger":       (55_000, 105_000),
    "safeway":      (45_000,  95_000),
    "walmart":      (30_000,  80_000),
    "thrift_store": (10_000,  45_000),
}

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("grocery_all_stores.csv")
    le = LabelEncoder()
    le.fit(df["segment"])
    df["segment_enc"] = le.transform(df["segment"])
    return df

@st.cache_resource
def build_infrastructure(df):
    sc_all    = StandardScaler()
    X_all_sc  = sc_all.fit_transform(df[FEAT_COLS].values.astype(float))
    pca_ref   = PCA(n_components=6, random_state=42)
    X_all_pca = pca_ref.fit_transform(X_all_sc)
    ref_lib   = {
        s: {"centroid": X_all_pca[df["store"]==s].mean(axis=0),
            "income_mean": np.mean(TRUE_RANGES[s])}
        for s in STORES
    }
    return sc_all, pca_ref, X_all_pca, ref_lib

# ── Model helpers ──────────────────────────────────────────────────────────────
def make_bridge(Xs, Xt, yr, alphas, n=400, seed=42):
    rng = np.random.RandomState(seed); bX, by = [], []
    for a in alphas:
        for _ in range(max(1, n // len(alphas))):
            i, j = rng.randint(len(Xs)), rng.randint(len(Xt))
            bX.append((1-a)*Xs[i] + a*Xt[j])
            by.append((1-a)*yr[i]  + a*0.5)
    return np.array(bX), np.array(by)

def get_range(Xs, Xt, ys, ts, refs, X_all_pca, df, ref_lib):
    preg = LinearRegression().fit(Xs[:,PROXY_IDX], ys)
    gap  = preg.predict(Xt[:,PROXY_IDX]).mean() - preg.predict(Xs[:,PROXY_IDX]).mean()
    cen  = X_all_pca[df["store"]==ts].mean(axis=0).reshape(1,-1)
    dists = {s: float(cdist(cen, ref_lib[s]["centroid"].reshape(1,-1), "euclidean")[0,0])
             for s in refs}
    nn    = min(dists, key=dists.get)
    bl    = 0.4*ref_lib[nn]["income_mean"] + 0.6*(preg.predict(Xs[:,PROXY_IDX]).mean()+gap)
    span  = (ys.max()-ys.min())/2*1.1
    return max(0, round((bl-span)/1000)*1000), round((bl+span)/1000)*1000, dists

def run_rank_model(Xs, Xt, ys, lo, hi, alphas):
    yr = (rankdata(ys)-1)/(len(ys)-1)
    bX, by = make_bridge(Xs, Xt, yr, alphas)
    m = Ridge(alpha=10.0)
    m.fit(np.vstack([Xs,bX]), np.concatenate([yr,by]))
    return lo + np.clip(m.predict(Xt), 0, 1)*(hi-lo), m

def run_log_model(Xs, Xt, ys):
    y_log = np.log(ys)
    log_prior = np.median(y_log)
    rng = np.random.RandomState(42); bX, by = [], []
    for a in [0.1,0.3,0.5,0.7]:
        for _ in range(100):
            i,j = rng.randint(len(Xs)), rng.randint(len(Xt))
            bX.append((1-a)*Xs[i]+a*Xt[j]); by.append((1-a)*y_log[i]+a*log_prior)
    m = Ridge(alpha=10.0)
    m.fit(np.vstack([Xs,bX]), np.concatenate([y_log,by]))
    return np.exp(m.predict(Xt)), m

# ── Run full pipeline (cached by params) ──────────────────────────────────────
@st.cache_data
def run_pipeline(alphas_tuple, model_type, _df_hash):
    df        = load_data()
    sc_all, pca_ref, X_all_pca, ref_lib = build_infrastructure(df)
    alphas    = list(alphas_tuple)
    results   = {}

    for ts in STORES:
        src = df[df["store"]!=ts].copy()
        tgt = df[df["store"]==ts].copy()
        sc  = StandardScaler()
        Xs  = sc.fit_transform(src[FEAT_COLS].values.astype(float))
        Xt  = sc.transform(tgt[FEAT_COLS].values.astype(float))
        ys  = src["income_usd"].values
        yt  = tgt["income_usd"].values
        refs = [s for s in STORES if s!=ts]

        lo, hi, dists = get_range(Xs, Xt, ys, ts, refs, X_all_pca, df, ref_lib)

        if model_type == "rank":
            preds, model = run_rank_model(Xs, Xt, ys, lo, hi, alphas)
        elif model_type == "log":
            preds, model = run_log_model(Xs, Xt, ys)
        else:  # hybrid
            pr, _  = run_rank_model(Xs, Xt, ys, lo, hi, alphas)
            pl, _  = run_log_model(Xs, Xt, ys)
            preds  = 0.7*pr + 0.3*pl
            model  = None

        sp, _   = spearmanr(yt, preds)
        avg_ks  = float(np.mean([ks_2samp(Xs[:,i], Xt[:,i]).statistic
                                  for i in range(Xs.shape[1])]))

        # Linear attribution (rank model only)
        contribs = None
        if model is not None and model_type == "rank":
            contribs = (Xt * model.coef_[np.newaxis,:]) * (hi - lo)

        results[ts] = {
            "yt": yt.tolist(), "preds": preds.tolist(),
            "mae":  float(mean_absolute_error(yt, preds)),
            "r2":   float(r2_score(yt, preds)),
            "sp":   float(sp),
            "bias": float(preds.mean() - yt.mean()),
            "lo":   lo, "hi": hi, "avg_ks": avg_ks,
            "nearest": min(dists, key=dists.get),
            "contribs": contribs.tolist() if contribs is not None else None,
            "tgt_names": tgt["name"].tolist(),
        }
    return results



# ── Single-customer prediction ────────────────────────────────────────────────
def predict_single_customer(feature_dict, train_stores, model_type, alphas,
                             df, sc_all, X_all_pca, ref_lib,
                             custom_range=None):
    """
    Predict income for one customer defined by feature_dict.
    Trains on train_stores (up to 5 known stores).
    custom_range: optional (lo, hi) tuple; if None, auto-estimated.
    Returns: predicted income, estimated range, nearest store, ranked contributions.
    """
    # Build training data from selected source stores
    src_df = df[df["store"].isin(train_stores)].copy()
    sc     = StandardScaler()
    X_src  = sc.fit_transform(src_df[FEAT_COLS].values.astype(float))
    y_src  = src_df["income_usd"].values

    # Encode the single customer as a 1-row feature matrix
    seg_map = {"Family": 0, "Professional": 1, "Senior": 2, "Single": 3, "Student": 4}
    raw_row = [
        feature_dict["age"],
        feature_dict["visits_per_month"],
        feature_dict["avg_basket_usd"],
        feature_dict["monthly_spend_usd"],
        feature_dict["grocery_pct"],
        feature_dict["electronics_pct"],
        feature_dict["apparel_pct"],
        feature_dict["home_pct"],
        feature_dict["private_label_pct"],
        feature_dict["online_orders_pct"],
        feature_dict["coupon_usage_pct"],
        feature_dict["loyalty_score"],
        seg_map[feature_dict["segment"]],
    ]
    X_cust = sc.transform(np.array([raw_row], dtype=float))  # (1, 13)

    # Estimate income range for this customer's store
    if custom_range is not None:
        lo, hi = custom_range
        dists  = {}
        nearest = train_stores[0]
    else:
        preg  = LinearRegression().fit(X_src[:, PROXY_IDX], y_src)
        gap   = float(preg.predict(X_cust[:, PROXY_IDX])[0]
                      - preg.predict(X_src[:, PROXY_IDX]).mean())
        cust_pca = sc_all.transform(np.array([raw_row], dtype=float))
        cust_pca = PCA(n_components=6, random_state=42).fit(
            sc_all.transform(df[FEAT_COLS].values.astype(float))
        ).transform(cust_pca)
        # Use X_all_pca centroid distances
        dists = {s: float(np.linalg.norm(X_all_pca[df["store"]==s].mean(0) - cust_pca[0]))
                 for s in train_stores}
        nearest = min(dists, key=dists.get)
        bl    = 0.4*ref_lib[nearest]["income_mean"] + 0.6*(
                preg.predict(X_src[:, PROXY_IDX]).mean() + gap)
        span  = (y_src.max()-y_src.min())/2*1.1
        lo    = max(0, round((bl-span)/1000)*1000)
        hi    = round((bl+span)/1000)*1000

    # Train model and predict
    if model_type == "rank":
        yr   = (rankdata(y_src)-1)/(len(y_src)-1)
        bX, by = make_bridge(X_src, X_cust, yr, alphas, n=200)
        m = Ridge(alpha=10.0)
        m.fit(np.vstack([X_src, bX]), np.concatenate([yr, by]))
        pred = float(lo + np.clip(m.predict(X_cust)[0], 0, 1)*(hi-lo))
        # Feature attributions
        contribs = (X_cust[0] * m.coef_) * (hi - lo)
    else:
        y_log = np.log(y_src)
        log_prior = float(np.median(y_log))
        rng2 = np.random.RandomState(42); bX, by = [], []
        for a in alphas:
            for _ in range(50):
                i = rng2.randint(len(X_src))
                bX.append((1-a)*X_src[i]+a*X_cust[0])
                by.append((1-a)*y_log[i]+a*log_prior)
        m = Ridge(alpha=10.0)
        m.fit(np.vstack([X_src,bX]), np.concatenate([y_log,by]))
        pred     = float(np.exp(m.predict(X_cust)[0]))
        contribs = np.zeros(len(FEAT_COLS))   # no linear attribution for log

    return pred, lo, hi, nearest, dists, contribs

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🛒 Income Prediction")
    st.caption("Cross-retailer domain adaptation")
    st.divider()

    page = st.radio("Page", ["📊 LOSO Pipeline", "🧍 Predict New Customer"],
                    label_visibility="collapsed")
    st.divider()

    st.subheader("Model settings")
    model_type = st.selectbox(
        "Transformation",
        ["rank", "log", "hybrid"],
        format_func=lambda x: {"rank":"Rank (S10)","log":"Log","hybrid":"Hybrid (70% rank + 30% log)"}[x],
    )

    st.markdown("**Bridge sampling alphas**")
    alpha_min = st.slider("Min alpha", 0.0, 0.5, 0.1, 0.05)
    alpha_max = st.slider("Max alpha", 0.5, 1.0, 0.7, 0.05)
    n_alphas  = st.slider("Number of alpha levels", 2, 6, 4)
    alphas    = tuple(np.round(np.linspace(alpha_min, alpha_max, n_alphas), 2).tolist())
    st.caption(f"Alphas: {list(alphas)}")

    st.divider()
    st.subheader("View settings")
    target_store = st.selectbox(
        "Focus store",
        STORES,
        format_func=lambda s: f"{STORE_LABELS[s]} ({TIER_LABELS[s]})",
    )
    show_feat_idx = st.selectbox(
        "Feature for distribution plot",
        range(len(FEAT_NICE)),
        format_func=lambda i: FEAT_NICE[i],
        index=2,
    )

    st.divider()
    st.caption("Tip: change alphas and click 'Run pipeline' to compare model configurations.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
if page == "🧍 Predict New Customer":
# ─────────────────────────────────────────────────────────────────────────────

    st.title("🧍 Predict a New Customer's Income")
    st.markdown(
        "Adjust the sliders to define a customer's behavioral profile. "
        "The model will predict their income using the S10 rank model trained "
        "on your chosen source stores."
    )

    # Load data and infrastructure
    df_p  = load_data()
    sc_all_p, pca_ref_p, X_all_pca_p, ref_lib_p = build_infrastructure(df_p)

    col_sliders, col_result = st.columns([2, 3])

    with col_sliders:
        st.subheader("Customer features")

        seg_choice = st.selectbox("Customer segment",
                                  ["Family","Professional","Senior","Single","Student"])

        st.markdown("**Shopping behaviour**")
        age_val   = st.slider("Age",              18, 75,  40)
        vis_val   = st.slider("Visits per month",  3, 14,   8)
        bask_val  = st.slider("Avg basket ($)",   12, 320, 100)

        st.markdown("**Spend breakdown (%)**")
        groc_val  = st.slider("Grocery %",    18, 70, 50)
        elec_val  = st.slider("Electronics %", 1, 16,  6)
        app_val   = st.slider("Apparel %",     5, 55, 15)
        home_val  = st.slider("Home %",        5, 46, 25)

        st.markdown("**Income-predictive signals**")
        pl_val    = st.slider("Private label %",  5, 89, 35)
        onl_val   = st.slider("Online orders %",  2, 77, 25)
        coup_val  = st.slider("Coupon use %",     3, 87, 30)
        loy_val   = st.slider("Loyalty score",   14, 87, 50)

        # Auto-compute monthly spend from visits × basket
        monthly_val = vis_val * bask_val

        st.caption(f"Monthly spend auto-computed: **${monthly_val:,}**  "
                   f"(visits × basket)")

        st.markdown("---")
        st.markdown("**Source stores (training data)**")
        source_opts = st.multiselect(
            "Train on",
            options=STORES,
            default=["kroger","safeway","walmart"],
            format_func=lambda s: STORE_LABELS[s],
        )

        st.markdown("**Custom income range (optional)**")
        use_custom_range = st.checkbox("Manually set income range")
        if use_custom_range:
            cr_lo = st.slider("Range low ($k)",  5, 100,  40) * 1000
            cr_hi = st.slider("Range high ($k)", 50, 200, 100) * 1000
            custom_range = (cr_lo, cr_hi)
        else:
            custom_range = None

    with col_result:
        if len(source_opts) == 0:
            st.warning("Select at least one source store to train on.")
        else:
            feature_dict = {
                "age":               age_val,
                "visits_per_month":  vis_val,
                "avg_basket_usd":    bask_val,
                "monthly_spend_usd": monthly_val,
                "grocery_pct":       groc_val,
                "electronics_pct":   elec_val,
                "apparel_pct":       app_val,
                "home_pct":          home_val,
                "private_label_pct": pl_val,
                "online_orders_pct": onl_val,
                "coupon_usage_pct":  coup_val,
                "loyalty_score":     loy_val,
                "segment":           seg_choice,
            }

            with st.spinner("Running prediction…"):
                pred, est_lo, est_hi, nearest, dists, contribs = predict_single_customer(
                    feature_dict, source_opts, model_type, list(alphas),
                    df_p, sc_all_p, X_all_pca_p, ref_lib_p, custom_range,
                )

            # ── Big prediction result ──────────────────────────────────────
            st.subheader("Predicted income")
            tier = "High" if pred > 80000 else "Median" if pred > 35000 else "Low"
            tier_color = {"High":"#185FA5","Median":"#1D9E75","Low":"#A32D2D"}[tier]

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted income", f"${pred:,.0f}")
            c2.metric("Estimated range",  f"${est_lo//1000}k – ${est_hi//1000}k")
            c3.metric("Income tier", tier)

            st.caption(f"Nearest training store: **{STORE_LABELS.get(nearest, nearest)}**")

            # ── Gauge chart ────────────────────────────────────────────────
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred/1000,
                number={"prefix":"$","suffix":"k","font":{"size":28}},
                gauge={
                    "axis": {"range":[0, 175], "ticksuffix":"k"},
                    "bar":  {"color": tier_color, "thickness": 0.25},
                    "steps": [
                        {"range":[0,   45],  "color":"#FDECEA"},
                        {"range":[45,  85],  "color":"#FDF5E6"},
                        {"range":[85,  175], "color":"#E8F0F8"},
                    ],
                    "threshold": {
                        "line": {"color": tier_color, "width": 4},
                        "thickness": 0.75, "value": pred/1000,
                    },
                },
                title={"text": "Predicted income ($k)"},
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Feature attribution waterfall ──────────────────────────────
            if model_type == "rank" and np.any(contribs != 0):
                st.subheader("Which features drove this prediction?")
                order = np.argsort(np.abs(contribs))[::-1][:10]
                vals  = contribs[order] / 1000
                names = [FEAT_NICE[i] for i in order]
                bar_cols = ["#27500A" if v > 0 else "#791F1F" for v in vals]

                fig_wf = go.Figure(go.Bar(
                    x=names, y=vals,
                    marker_color=bar_cols,
                    hovertemplate="%{x}: %{y:+.2f}k<extra></extra>",
                ))
                fig_wf.add_hline(y=0, line_color="black", line_width=1)
                fig_wf.update_layout(
                    height=300,
                    title="Feature contributions to prediction ($k above/below baseline)",
                    yaxis_title="Contribution ($k)",
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_wf, use_container_width=True)

            # ── How does this customer compare to each store? ──────────────
            st.subheader("How does this customer compare to each store's average?")
            radar_feats = ["avg_basket_usd","coupon_usage_pct","private_label_pct",
                           "online_orders_pct","loyalty_score","visits_per_month"]
            radar_nice  = ["Basket $","Coupon %","Priv. label","Online %","Loyalty","Visits"]

            df_p2 = df_p  # already loaded
            store_means = {s: [df_p2[df_p2["store"]==s][f].mean() for f in radar_feats]
                           for s in STORES}
            cust_vals   = [feature_dict[f] for f in radar_feats]

            # Normalise to [0, 1] per feature
            feat_max = [max(store_means[s][i] for s in STORES) for i in range(len(radar_feats))]
            feat_min = [min(store_means[s][i] for s in STORES) for i in range(len(radar_feats))]

            def norm_feat(vals):
                return [(v - feat_min[i]) / max(feat_max[i]-feat_min[i], 1e-6)
                        for i, v in enumerate(vals)]

            angles = list(range(len(radar_nice))) + [0]
            fig_radar = go.Figure()
            for s in STORES:
                nv = norm_feat(store_means[s]) + [norm_feat(store_means[s])[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=nv, theta=radar_nice+[radar_nice[0]],
                    name=STORE_LABELS[s],
                    line=dict(color=STORE_COLORS[s], width=1.5),
                    fill="none", opacity=0.5,
                ))
            # Customer
            cv = norm_feat(cust_vals) + [norm_feat(cust_vals)[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=cv, theta=radar_nice+[radar_nice[0]],
                name="Your customer",
                line=dict(color=CUSTOM_COLOR, width=3, dash="dot"),
                fill="toself",
                fillcolor=rgba(CUSTOM_COLOR, 0.13),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1.1])),
                showlegend=True, height=380,
                title="Customer profile vs store averages (normalised)",
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
else:  # LOSO Pipeline page
# ─────────────────────────────────────────────────────────────────────────────

    st.title("Grocery Store Customer Income Prediction")
st.markdown(
    "**Train:** Kroger + Safeway + Walmart (median tier) | "
    "**Method:** Leave-one-store-out CV | "
    "**No target labels used**"
)

# Load data and run pipeline
try:
    df = load_data()
    sc_all, pca_ref, X_all_pca, ref_lib = build_infrastructure(df)
except FileNotFoundError:
    st.error("grocery_all_stores.csv not found. Place it in the same directory as app.py.")
    st.stop()

with st.spinner(f"Running {model_type} model pipeline…"):
    results = run_pipeline(alphas, model_type, id(df))

# ── Top KPI strip ─────────────────────────────────────────────────────────────
st.subheader("Pipeline results — all 5 stores")
cols = st.columns(5)
for col, ts in zip(cols, STORES):
    r   = results[ts]
    lbl = STORE_LABELS[ts]
    col.metric(
        label=f"**{lbl}**",
        value=f"${r['mae']/1000:.1f}k MAE",
        delta=f"Sp={r['sp']:.3f}  R²={r['r2']:.3f}",
        delta_color="off",
    )

st.divider()

# ── Two column layout ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    # ── Tab bar ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Predictions", "🗺️ PCA space", "📉 Feature shift",
        "🎯 Attribution", "🔢 Summary table"
    ])

    # ── TAB 1: Predicted vs actual ────────────────────────────────────────────
    with tab1:
        r   = results[target_store]
        yt  = np.array(r["yt"]); preds = np.array(r["preds"])
        col = STORE_COLORS[target_store]
        lbl = STORE_LABELS[target_store]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Actual vs Predicted", "Residual distribution"],
        )
        # Scatter
        fig.add_trace(go.Scatter(
            x=yt/1000, y=preds/1000, mode="markers",
            marker=dict(color=col, size=5, opacity=0.5),
            name="Customers",
            hovertemplate="Actual: $%{x:.0f}k<br>Pred: $%{y:.0f}k<extra></extra>",
        ), row=1, col=1)
        lo_k = TRUE_RANGES[target_store][0]/1000
        hi_k = TRUE_RANGES[target_store][1]/1000
        fig.add_trace(go.Scatter(
            x=[lo_k, hi_k], y=[lo_k, hi_k], mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="Perfect prediction", showlegend=False,
        ), row=1, col=1)
        # Residuals
        resid = (preds - yt) / 1000
        fig.add_trace(go.Histogram(
            x=resid, nbinsx=25, marker_color=col, opacity=0.7,
            name="Residuals",
        ), row=1, col=2)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)
        fig.add_vline(x=float(resid.mean()), line_color=col, line_width=2,
                      annotation_text=f"bias={resid.mean():+.1f}k", row=1, col=2)

        fig.update_xaxes(title_text="Actual ($k)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted ($k)", row=1, col=1)
        fig.update_xaxes(title_text="Error ($k)", row=1, col=2)
        fig.update_layout(height=380, showlegend=False,
                          title_text=f"{lbl} — MAE ${r['mae']/1000:.1f}k  "
                                     f"R²={r['r2']:.3f}  Spearman={r['sp']:.3f}")
        st.plotly_chart(fig, use_container_width=True)

        # Per-customer table (top errors)
        df_pred = pd.DataFrame({
            "Customer":    r["tgt_names"],
            "Actual ($)":  [f"${v:,.0f}" for v in yt],
            "Predicted ($)":[f"${v:,.0f}" for v in preds],
            "Error ($)":   [f"${v:,.0f}" for v in np.abs(yt-preds)],
        })
        df_pred["_err"] = np.abs(yt-preds)
        df_pred = df_pred.sort_values("_err", ascending=False).drop("_err", axis=1)
        st.caption(f"Largest prediction errors for {lbl} (top 10)")
        st.dataframe(df_pred.head(10), use_container_width=True, hide_index=True)

    # ── TAB 2: PCA scatter ────────────────────────────────────────────────────
    with tab2:
        fig = go.Figure()
        for ts in STORES:
            mask  = df["store"] == ts
            X_pca = X_all_pca[mask.values]
            inc   = df[mask]["income_usd"].values
            fig.add_trace(go.Scatter(
                x=X_pca[:,0], y=X_pca[:,1],
                mode="markers",
                marker=dict(color=STORE_COLORS[ts], size=5, opacity=0.45),
                name=STORE_LABELS[ts],
                hovertemplate=f"{STORE_LABELS[ts]}<br>PC1=%{{x:.2f}} PC2=%{{y:.2f}}<extra></extra>",
            ))
        # Centroids
        for ts in STORES:
            cen = ref_lib[ts]["centroid"]
            fig.add_trace(go.Scatter(
                x=[cen[0]], y=[cen[1]], mode="markers+text",
                marker=dict(color=STORE_COLORS[ts], size=16, symbol="star",
                            line=dict(color="white", width=1.5)),
                text=[STORE_LABELS[ts].split()[0]], textposition="top center",
                textfont=dict(size=9, color=STORE_COLORS[ts]),
                name=f"{STORE_LABELS[ts]} centroid", showlegend=False,
            ))
        ev = pca_ref.explained_variance_ratio_
        fig.update_layout(
            height=400,
            title=f"PCA feature space — PC1 {ev[0]*100:.1f}% + PC2 {ev[1]*100:.1f}% variance",
            xaxis_title=f"PC1 ({ev[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({ev[1]*100:.1f}%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Feature shift ──────────────────────────────────────────────────
    with tab3:
        # KS heatmap
        src_df = df[df["store"].isin(["kroger","safeway","walmart"])]
        ks_mat = np.zeros((len(STORES), len(FEAT_COLS)))
        for i, ts in enumerate(STORES):
            tgt_sub = df[df["store"]==ts]
            for j, feat in enumerate(FEAT_COLS):
                ks_mat[i, j] = ks_2samp(src_df[feat].values, tgt_sub[feat].values).statistic

        fig = go.Figure(data=go.Heatmap(
            z=ks_mat,
            x=FEAT_NICE, y=[STORE_LABELS[s] for s in STORES],
            colorscale="YlOrRd", zmin=0, zmax=1,
            text=np.round(ks_mat, 3),
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="Store: %{y}<br>Feature: %{x}<br>KS: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(height=320,
                          title="KS statistic: source (Kroger+Safeway+Walmart) vs each store",
                          xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)

        # Feature distribution for selected feature
        feat_col = FEAT_COLS[show_feat_idx]
        feat_lbl = FEAT_NICE[show_feat_idx]
        fig2 = go.Figure()
        for ts in STORES:
            vals = df[df["store"]==ts][feat_col].values
            fig2.add_trace(go.Violin(
                y=vals, name=STORE_LABELS[ts],
                box_visible=True, meanline_visible=True,
                fillcolor=rgba(STORE_COLORS[ts], 0.45),
                line_color=STORE_COLORS[ts],
            ))
        fig2.update_layout(height=320, title=f"Distribution: {feat_lbl} by store",
                           yaxis_title=feat_lbl, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 4: Attribution ────────────────────────────────────────────────────
    with tab4:
        r = results[target_store]
        if r["contribs"] is None:
            st.info("Attribution only available for the rank model. Switch to 'Rank (S10)'.")
        else:
            contribs = np.array(r["contribs"])
            mean_abs = np.abs(contribs).mean(axis=0)
            sorted_idx = np.argsort(mean_abs)[::-1]

            fig = go.Figure(go.Bar(
                x=[FEAT_NICE[i] for i in sorted_idx],
                y=mean_abs[sorted_idx]/1000,
                marker_color=[STORE_COLORS[target_store]] * len(sorted_idx),
                hovertemplate="%{x}: mean |contrib| $%{y:.1f}k<extra></extra>",
            ))
            fig.update_layout(
                height=320,
                title=f"Mean |attribution| per feature — {STORE_LABELS[target_store]}",
                xaxis_tickangle=-40,
                yaxis_title="Mean |contribution| ($k)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Waterfall for worst prediction
            yt    = np.array(r["yt"])
            preds = np.array(r["preds"])
            errs  = np.abs(yt - preds)
            worst_idx = int(np.argmax(errs))

            wf_contribs = contribs[worst_idx]
            order = np.argsort(np.abs(wf_contribs))[::-1][:10]
            vals  = wf_contribs[order] / 1000

            fig2 = go.Figure(go.Bar(
                x=[FEAT_NICE[i] for i in order],
                y=vals,
                marker_color=["#27500A" if v > 0 else "#791F1F" for v in vals],
                hovertemplate="%{x}: %{y:+.1f}k<extra></extra>",
            ))
            fig2.add_hline(y=0, line_color="black", line_width=1)
            fig2.add_hline(y=preds[worst_idx]/1000, line_dash="dash",
                           line_color=STORE_COLORS[target_store],
                           annotation_text=f"Pred ${preds[worst_idx]/1000:.0f}k")
            fig2.add_hline(y=yt[worst_idx]/1000, line_color="black",
                           annotation_text=f"Actual ${yt[worst_idx]/1000:.0f}k")
            fig2.update_layout(
                height=320,
                title=f"Worst prediction waterfall — error ${errs[worst_idx]/1000:.1f}k  "
                      f"({r['tgt_names'][worst_idx]})",
                xaxis_tickangle=-40,
                yaxis_title="Contribution to prediction ($k)",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 5: Summary table ──────────────────────────────────────────────────
    with tab5:
        rows = []
        for ts in STORES:
            r   = results[ts]
            lo  = TRUE_RANGES[ts][0]; hi = TRUE_RANGES[ts][1]
            rows.append({
                "Store":       STORE_LABELS[ts],
                "Tier":        TIER_LABELS[ts],
                "True range":  f"${lo//1000}k–${hi//1000}k",
                "Est. range":  f"${r['lo']//1000}k–${r['hi']//1000}k",
                "Nearest ref": STORE_LABELS.get(r["nearest"], r["nearest"]),
                "Avg KS":      f"{r['avg_ks']:.3f}",
                "MAE ($k)":    f"{r['mae']/1000:.2f}",
                "R²":          f"{r['r2']:.4f}",
                "Spearman r":  f"{r['sp']:.4f}",
                "Bias ($k)":   f"{r['bias']/1000:+.2f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Store"), use_container_width=True)

        # Alpha sensitivity
        st.markdown("**Alpha influence on MAE**")
        alpha_results = {}
        for a_set in [(0.1,0.3,0.5,0.7),(0.3,0.5,0.7,0.9),(0.7,0.9),(0.9,)]:
            avg = np.mean([run_pipeline(a_set, model_type, id(df))[s]["mae"]
                           for s in STORES]) / 1000
            alpha_results[str(list(a_set))] = avg

        fig_a = go.Figure(go.Bar(
            x=list(alpha_results.keys()),
            y=list(alpha_results.values()),
            marker_color=[
                rgba("#185FA5", 1.0) if k==str(list(alphas))
                else rgba("#B4B2A9", 0.6)
                for k in alpha_results
            ],
        ))
        fig_a.update_layout(height=250, yaxis_title="Avg MAE ($k)", showlegend=False,
                            title="Avg MAE for 4 common alpha sets (current highlighted)")
        st.plotly_chart(fig_a, use_container_width=True)

# ── Right sidebar panel ───────────────────────────────────────────────────────
with col_right:
    st.subheader("Store details")
    ts  = target_store
    r   = results[ts]
    col = STORE_COLORS[ts]
    lbl = STORE_LABELS[ts]

    # KPI cards
    c1, c2 = st.columns(2)
    c1.metric("MAE", f"${r['mae']/1000:.1f}k")
    c2.metric("Spearman r", f"{r['sp']:.3f}")
    c1.metric("R²", f"{r['r2']:.4f}")
    c2.metric("Bias", f"${r['bias']/1000:+.1f}k")

    st.caption(f"**Tier:** {TIER_LABELS[ts]} | **KS shift:** {r['avg_ks']:.3f}")
    st.caption(f"**Nearest ref:** {STORE_LABELS.get(r['nearest'], r['nearest'])}")
    st.caption(f"**Est. range:** ${r['lo']//1000}k – ${r['hi']//1000}k  "
               f"(true: ${TRUE_RANGES[ts][0]//1000}k – ${TRUE_RANGES[ts][1]//1000}k)")

    st.divider()

    # MAE comparison bar
    st.markdown("**MAE across all stores**")
    maes = [results[s]["mae"]/1000 for s in STORES]
    fig_bar = go.Figure(go.Bar(
        x=[STORE_LABELS[s] for s in STORES],
        y=maes,
        marker_color=[
            rgba(STORE_COLORS[s], 1.0 if s==target_store else 0.35)
            for s in STORES
        ],
        marker_line_color=[STORE_COLORS[s] for s in STORES],
        marker_line_width=[3 if s==target_store else 0 for s in STORES],
        hovertemplate="%{x}: $%{y:.1f}k<extra></extra>",
    ))
    fig_bar.update_layout(height=200, showlegend=False, margin=dict(l=0,r=0,t=0,b=30),
                          yaxis_title="MAE ($k)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Spearman comparison
    st.markdown("**Spearman rank correlation**")
    sps = [results[s]["sp"] for s in STORES]
    fig_sp = go.Figure(go.Bar(
        x=[STORE_LABELS[s] for s in STORES], y=sps,
        marker_color=[
            rgba(STORE_COLORS[s], 1.0 if s==target_store else 0.35)
            for s in STORES
        ],
        marker_line_color=[STORE_COLORS[s] for s in STORES],
        marker_line_width=[3 if s==target_store else 0 for s in STORES],
    ))
    fig_sp.update_layout(height=200, showlegend=False, margin=dict(l=0,r=0,t=0,b=30),
                         yaxis_title="Spearman r")
    st.plotly_chart(fig_sp, use_container_width=True)

    st.divider()
    st.subheader("How it works")
    st.markdown(f"""\
**Step 1** — Range estimation (Method B):
- Proxy regression on basket & online orders
- Nearest store in PCA space: **{STORE_LABELS.get(r['nearest'], r['nearest'])}**
- Estimated: **${r['lo']//1000}k – ${r['hi']//1000}k**

**Step 2** — {model_type.capitalize()} model:
- Alphas: `{list(alphas)}`
- Bridge samples: {len(alphas)*100} (source ↔ target)
- Ridge(alpha=10.0) in {"rank" if model_type!="log" else "log"} space

**Key invariant:** Spearman rank correlation is
identical regardless of range estimate quality.
All MAE differences are scaling artifacts.
""")
