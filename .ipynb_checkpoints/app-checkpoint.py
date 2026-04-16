import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime #only for greeting

import os, json, joblib
from mlxtend.frequent_patterns import fpgrowth, association_rules

st.set_page_config(
    page_title="Bank Customer Attrition Insights", 
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main content styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
    
    h2 {
        color: #374151;
        font-weight: 600;
    }
    
    h3 {
        color: #4b5563;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Risk indicator */
    .risk-low {
        color: #10b981;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    .risk-medium {
        color: #f59e0b;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    .risk-high {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Paths
# =============================
DATA_PATH = "BankChurners_clean.csv"
MODEL_DIR = "models"
ART_DIR = "artifacts"

MODEL_FILES = {
    "LR": "lr_model.joblib",
    "SVM": "svm_model.joblib",
    "RF": "rf_model.joblib",
    "NB": "nb_model.joblib",
}

METRIC_FILES = {
    "LR": "lr_metrics.json",
    "SVM": "svm_metrics.json",
    "RF": "rf_metrics.json",
    "NB": "nb_metrics.json",
}

# =============================
# Loaders
# =============================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target"] = df["Attrition_Flag"].map({"Existing Customer": 0, "Attrited Customer": 1}).astype(int)
    return df

@st.cache_data
def load_preprocess_spec() -> dict:
    spec_path = os.path.join(ART_DIR, "preprocess_columns.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_frequent_itemsets() -> pd.DataFrame:
    p = os.path.join(ART_DIR, "frequent_itemsets.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


@st.cache_resource
def load_models() -> dict:
    models = {}
    for name, fn in MODEL_FILES.items():
        models[name] = joblib.load(os.path.join(MODEL_DIR, fn))  # {"model":..., "scaler":...}
    return models

@st.cache_data
def load_metrics_table() -> pd.DataFrame:
    rows = []
    for name, fn in METRIC_FILES.items():
        p = os.path.join(ART_DIR, fn)
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "roc_auc": m.get("roc_auc"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)

# =============================
# Preprocess + prediction helpers
# =============================
def preprocess_X_raw(X_raw: pd.DataFrame, spec: dict, scaler) -> pd.DataFrame:
    """
    Transforms a raw feature row(s) into the training-aligned matrix:
    - one-hot encode categorical columns
    - scale numerical columns
    - align to the exact column set used at training
    """
    cat_cols = spec["cat_cols"]
    num_cols = spec["num_cols"]
    full_cols = spec["columns"]

    X = X_raw.copy()

    for c in cat_cols:
        if c not in X.columns:
            X[c] = ""
    for c in num_cols:
        if c not in X.columns:
            X[c] = 0.0

    X_oh = pd.get_dummies(X, columns=cat_cols)
    X_oh = X_oh.reindex(columns=full_cols, fill_value=0)

    for c in num_cols:
        if c in X_oh.columns:
            X_oh[c] = X_oh[c].astype(float)

    X_oh[num_cols] = scaler.transform(X_oh[num_cols])
    return X_oh

def prob_to_band(p: float, bands: dict) -> tuple[str, str]:
    """
    Map probability to a banker-friendly band.
    bands: {"low":0.30, "high":0.60} meaning:
      - < low  => Low risk
      - low..high => Medium risk
      - >= high => High risk
    """
    low = float(bands.get("low", 0.30))
    high = float(bands.get("high", 0.60))
    if p < low:
        return "Low", "Likely to stay"
    if p < high:
        return "Medium", "Needs attention"
    return "High", "High likelihood to leave"

def risk_bar(prob: float):
    # Color based on risk level
    if prob < 0.30:
        color = "#10b981"  # Green
    elif prob < 0.60:
        color = "#f59e0b"  # Orange
    else:
        color = "#ef4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Risk Score", 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60}}))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def compute_portfolio_risk(df: pd.DataFrame, spec: dict, _model_pack: dict, model_key: str) -> pd.DataFrame:
    """
    Adds model probability to each customer row (for watchlist + portfolio KPIs).
    Cached because it is used in multiple pages.
    """
    X = df.drop(columns=["target", "Attrition_Flag"], errors="ignore")
    if "CLIENTNUM" in X.columns:
        X = X.drop(columns=["CLIENTNUM"])
    scaler = _model_pack["scaler"]
    model = _model_pack["model"]

    X_m = preprocess_X_raw(X, spec, scaler)
    prob = model.predict_proba(X_m)[:, 1]
    out = df.copy()
    out["churn_prob"] = prob
    return out

def highlight_signals(row: pd.Series, churn_profile: dict) -> list[str]:
    """
    Lightweight, banker-readable 'signals' (not a technical explanation).
    Uses dataset percentiles for a few well-known churn drivers in this dataset.
    """
    signals = []
    # Each entry: (feature, direction, threshold, message)
    for feat, rule in churn_profile.items():
        if feat not in row:
            continue
        v = row[feat]
        thr = rule["thr"]
        if pd.isna(v):
            continue
        if rule["dir"] == "low" and v <= thr:
            signals.append(rule["msg"])
        if rule["dir"] == "high" and v >= thr:
            signals.append(rule["msg"])
    return signals[:6]

@st.cache_data
def build_churn_signal_profile(df: pd.DataFrame) -> dict:
    """
    Create a simple 'risk signal' profile using quantiles from churned vs stayed customers.
    This is intentionally non-technical: it produces thresholds and plain-English messages.
    """
    churned = df[df["target"] == 1]
    stayed = df[df["target"] == 0]

    profile = {}

    def q(s, qq):
        return float(np.nanquantile(s.astype(float), qq))

    # Frequently strong churn indicators for this dataset (transaction activity, engagement)
    candidates = [
        ("Total_Trans_Ct", "low", 0.25, "Low transaction count (reduced activity)"),
        ("Total_Trans_Amt", "low", 0.25, "Low total transaction amount (reduced spending)"),
        ("Total_Ct_Chng_Q4_Q1", "low", 0.25, "Transaction count dropped vs prior period"),
        ("Total_Amt_Chng_Q4_Q1", "low", 0.25, "Spending dropped vs prior period"),
        ("Contacts_Count_12_mon", "high", 0.75, "Higher service contact frequency (potential friction)"),
        ("Months_Inactive_12_mon", "high", 0.75, "More months inactive in the past year"),
    ]

    for feat, direction, qq, msg in candidates:
        if feat not in df.columns:
            continue
        # Choose threshold from churned distribution to capture meaningful signals
        thr = q(churned[feat], qq)
        profile[feat] = {"dir": direction, "thr": thr, "msg": msg}

    return profile

# =============================
# ARM helpers (banker-friendly) - OPTIMIZED
# =============================
@st.cache_data(show_spinner="Mining churn patterns... This may take a moment.")
def run_arm_analysis(df: pd.DataFrame, min_support: float, min_confidence: float, max_len: int) -> pd.DataFrame:
    """
    OPTIMIZED: Cache ARM results to avoid recomputation.
    Run association rule mining on discretized customer features.
    Returns a DataFrame of rules suitable for banker interpretation.
    """
    def safe_bins(s: pd.Series, bins: list, labels: list):
        try:
            return pd.cut(s, bins=bins, labels=labels, duplicates="drop")
        except:
            return pd.Series(["Unknown"] * len(s), index=s.index)

    arm_df = df.copy()
    
    # Discretization
    if "Total_Trans_Ct" in arm_df.columns:
        arm_df["TxnCount_Level"] = safe_bins(arm_df["Total_Trans_Ct"], 
                                              [0, 40, 80, 999], 
                                              ["Low", "Medium", "High"])
    
    if "Total_Trans_Amt" in arm_df.columns:
        arm_df["TxnAmt_Level"] = safe_bins(arm_df["Total_Trans_Amt"], 
                                           [0, 3000, 8000, 99999], 
                                           ["Low", "Medium", "High"])
    
    if "Total_Revolving_Bal" in arm_df.columns:
        arm_df["RevBal_Level"] = safe_bins(arm_df["Total_Revolving_Bal"], 
                                           [0, 500, 1500, 99999], 
                                           ["Low", "Medium", "High"])
    
    if "Contacts_Count_12_mon" in arm_df.columns:
        arm_df["Contacts_Level"] = safe_bins(arm_df["Contacts_Count_12_mon"], 
                                             [0, 2, 4, 99], 
                                             ["Low", "Medium", "High"])
    
    if "Months_Inactive_12_mon" in arm_df.columns:
        arm_df["Inactive_Level"] = safe_bins(arm_df["Months_Inactive_12_mon"], 
                                             [0, 2, 3, 99], 
                                             ["Low", "Medium", "High"])

    arm_df["Outcome"] = arm_df["Attrition_Flag"].map({"Attrited Customer": "Churned", "Existing Customer": "Stayed"})

    cat_cols = ["Card_Category", "Gender", "Education_Level", "Marital_Status", "Income_Category",
                "TxnCount_Level", "TxnAmt_Level", "RevBal_Level", "Contacts_Level", "Inactive_Level", "Outcome"]
    cat_cols = [c for c in cat_cols if c in arm_df.columns]

    basket = arm_df[cat_cols].copy()
    for c in basket.columns:
        basket[c] = basket[c].astype(str)

    basket_oh = pd.get_dummies(basket)
    basket_oh = basket_oh.astype(bool)

    # FP-Growth
    freq = fpgrowth(basket_oh, min_support=min_support, use_colnames=True, max_len=max_len)
    
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence, num_itemsets=len(freq))
    
    if rules.empty:
        return pd.DataFrame()

    rules = rules.sort_values("lift", ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    
    return rules

def humanize_rule(row: pd.Series) -> str:
    ant = row["antecedents"].replace("_Level", "").replace("_", " ")
    cons = row["consequents"].replace("_Level", "").replace("_", " ")
    return f"IF {ant} THEN {cons}"

# =============================
# Load data + models
# =============================
df = load_data(DATA_PATH)
spec = load_preprocess_spec()
models_pack = load_models()
metrics = load_metrics_table()
freq_itemsets = load_frequent_itemsets()


# Default to best-performing model
best_model_key = metrics.index[0] if not metrics.empty else "LR"
best_pack = models_pack[best_model_key]

# Compute portfolio-level risk (cached)
df_scored = compute_portfolio_risk(df, spec, best_pack, best_model_key)

# Build churn signal profile
signal_profile = build_churn_signal_profile(df)

# Risk band definitions
risk_bands = {"low": 0.30, "high": 0.60}

# =============================
# Custom CSS 
# =============================
st.markdown("""
<style>
    /* Main content styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* deep baby blue background */
    [data-testid="stSidebar"] {
        background-color: #5f1e3a !important;  /* 深宝宝蓝 */
    }
    
    /* white font */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* font of selection button */s
    [data-testid="stSidebar"] .stRadio label {
        color: #e0e7ff !important;  /* blue */
        font-size: 1.05em !important;
        font-weight: 500 !important;
        padding: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    /* */
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #fbbf24 !important;  /* yellow */
        transform: translateX(5px);
    }
    
    /* choosen page */
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
        
        font-weight: 700 !important;
        background-color: rgba(251, 191, 36, 0.1);
        border-radius: 8px;
        padding: 0.5rem 1rem !important;
    }
    
    /* selected circle */
    [data-testid="stSidebar"] .stRadio input:checked ~ div {
        background-color: #fbbf24 !important;
        border-color: #fbbf24 !important;
    }
    
    /* not selected circle */
    [data-testid="stSidebar"] .stRadio input ~ div {
        border-color: #e0e7ff !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1 {
        color: #1f2937;
        font-weight: 700;
    }
    
    h2 {
        color: #374151;
        font-weight: 600;
    }
    
    h3 {
        color: #4b5563;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Risk indicators */
    .risk-low {
        color: #10b981;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    .risk-medium {
        color: #f59e0b;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    .risk-high {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Sidebar Navigation with Icons
# =============================
now = datetime.now()
hour = now.hour

if hour < 12:
    greeting = "Good Morning"
elif hour < 18:
    greeting = "Good Afternoon"
else:
    greeting = "Good Evening"

time_str = now.strftime("%I:%M %p")   # e.g. 09:45 AM
date_str = now.strftime("%d %b %Y")   # e.g. 05 Feb 2026

st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 1.5rem 0;'>
        <h1 style='color: #fbbf24 !important; margin: 0; font-size: 2.5em;'>
            🏦
        </h1>
        <h2 style='color: #e0e7ff !important; margin: 0.5rem 0 0 0; font-size: 1.3em; font-weight: 600;'>
            {greeting}
        </h2>
        <div style='color: #e0e7ff !important; margin-top: 0.35rem; font-size: 0.95em; opacity: 0.9;'>
            {date_str} • {time_str}
        </div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='border: 1px solid #e0e7ff; opacity: 0.3; margin: 1rem 0;'>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Page",
    [
        "📊 Customer Overview",
        "🔍 Churn Patterns",
        "🎯 Predict & Explain",
        "⚙️ Advanced Settings"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='border: 1px solid #e0e7ff; opacity: 0.3; margin: 1rem 0;'>", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style='
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: #1e3a5f;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin-top: 1rem;
    '>
        <div style='font-size: 1.3em; margin-bottom: 0.5rem;'>💡</div>
        <div style='font-weight: 700; margin-bottom: 0.5rem; font-size: 1.1em;'>Quick Tip</div>
        <div style='font-size: 0.95em; line-height: 1.5;'>
            For Churn Patterns analysis depth, <Strong> deeper levels = more patterns + more time </Strong>. Choose Standard Patterns for quick and medium coverage analysis, use Deep analysis for comprehensive insights.
        </div>
    </div>
""", unsafe_allow_html=True)
# =============================
# Page 1 — Customer Overview (IMPROVED)
# =============================
if page == "📊 Customer Overview":
    st.title("📊 Customer Overview Dashboard")
    st.markdown("Get a comprehensive view of your customer portfolio and attrition metrics")
    
    st.markdown("---")
    
    # Key Metrics Row with improved cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    attrited = int(df["target"].sum())
    retained = total_customers - attrited
    attrition_rate = (attrited / total_customers * 100) if total_customers > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:0.9em;">👥 Total Customers</h3>
            <h2 style="margin:0.5em 0 0 0;">{total_customers:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-warning">
            <h3 style="margin:0; font-size:0.9em;">📉 Attrited</h3>
            <h2 style="margin:0.5em 0 0 0;">{attrited:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-success">
            <h3 style="margin:0; font-size:0.9em;">✅ Retained</h3>
            <h2 style="margin:0.5em 0 0 0;">{retained:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size:0.9em;">📊 Attrition Rate</h3>
            <h2 style="margin:0.5em 0 0 0;">{attrition_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Attrition Rate Distribution")
        
        # Create pie chart for attrition
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Retained', 'Attrited'],
            values=[retained, attrited],
            hole=0.4,
            marker_colors=['#10b981', '#ef4444'],
            textinfo='label+percent',
            textfont_size=14,
            pull=[0, 0.1]
        )])
        
        fig_pie.update_layout(
            showlegend=True,
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            annotations=[dict(text=f'{attrition_rate:.1f}%', x=0.5, y=0.5, font_size=24, showarrow=False)]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution")
        
        # Risk distribution
        df_scored["risk_band"] = df_scored["churn_prob"].apply(
            lambda p: "High Risk" if p >= 0.60 else ("Medium Risk" if p >= 0.30 else "Low Risk")
        )
        risk_counts = df_scored["risk_band"].value_counts()
        
        fig_risk = go.Figure(data=[go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            marker_color=['#10b981', '#f59e0b', '#ef4444'],
            text=risk_counts.values,
            textposition='auto',
        )])
        
        fig_risk.update_layout(
            xaxis_title="Risk Level",
            yaxis_title="Number of Customers",
            height=350,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("---")
    
    # High Risk Customers Watchlist
    st.subheader("⚠️ High Risk Customers - Priority Watchlist")
    st.caption("Customers with >60% churn probability requiring immediate attention")
    
    high_risk = df_scored[df_scored["churn_prob"] >= 0.60].copy()
    
    if high_risk.empty:
        st.success("✅ No high-risk customers detected!")
    else:
        high_risk = high_risk.sort_values("churn_prob", ascending=False)
        
        display_cols = ["CLIENTNUM", "churn_prob"]
        optional_cols = ["Card_Category", "Total_Trans_Ct", "Total_Trans_Amt", 
                        "Contacts_Count_12_mon", "Months_Inactive_12_mon"]
        for c in optional_cols:
            if c in high_risk.columns:
                display_cols.append(c)
        
        display_df = high_risk[display_cols].head(20).copy()
        display_df["churn_prob"] = (display_df["churn_prob"] * 100).round(1)
        display_df = display_df.rename(columns={"churn_prob": "Risk Score (%)"})
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk Score (%)": st.column_config.ProgressColumn(
                    "Risk Score (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            }
        )
        
        st.markdown(f"** Total High Risk Customers:** {len(high_risk)}")
    
    
    # =============================
    # Additional Insights (Selectable Feature)
    # =============================
    st.markdown("---")
    st.subheader("📌 Attrition Breakdown by Selected Feature")
    st.caption("Select a customer attribute to see its attrition rate distribution.")

    # banker-friendly selectable features (only show columns that exist)
    feature_candidates = [
        ("Card_Category", "💳 Card Category"),
        ("Gender", "👥 Gender"),
        ("Income_Category", "💰 Income Category"),
        ("Education_Level", "🎓 Education Level"),
        ("Marital_Status", "💍 Marital Status"),
        ("Customer_Age", "🎂 Age Group"),  # will bin if numeric
        ("Months_on_book", "🏦 Months with Bank"),  # will bin if numeric
    ]

    available = [(col, label) for col, label in feature_candidates if col in df.columns]

    if not available:
        st.info("No suitable feature columns available for breakdown charts.")
    else:
        # UI controls
        colA, colB = st.columns([2, 1])
        with colA:
            chosen_col = st.selectbox(
                "Select feature",
                options=[c for c, _ in available],
                format_func=lambda x: dict(available).get(x, x),
            )
        with colB:
            view_mode = st.selectbox("View", ["Attrition rate (%)", "Customer count"], index=0)

        plot_df = df.copy()

        # If numeric, bin it to make bar chart readable (banker-friendly)
        if chosen_col in ["Customer_Age", "Months_on_book"]:
            if chosen_col in plot_df.columns:
                try:
                    if chosen_col == "Customer_Age":
                        plot_df["__bin__"] = pd.cut(
                            plot_df[chosen_col],
                            bins=[0, 35, 45, 55, 100],
                            labels=["<=35", "36-45", "46-55", "56+"]
                        )
                    else:
                        # Months_on_book
                        plot_df["__bin__"] = pd.cut(
                            plot_df[chosen_col],
                            bins=[0, 24, 36, 48, 120],
                            labels=["<=24", "25-36", "37-48", "49+"]
                        )
                    group_col = "__bin__"
                    chart_title = f"{dict(available).get(chosen_col, chosen_col)} (Grouped)"
                except Exception:
                    group_col = chosen_col
                    chart_title = dict(available).get(chosen_col, chosen_col)
            else:
                group_col = chosen_col
                chart_title = dict(available).get(chosen_col, chosen_col)
        else:
            group_col = chosen_col
            chart_title = dict(available).get(chosen_col, chosen_col)

        # Compute breakdown
        g = plot_df.groupby(group_col)["target"].agg(["sum", "count"]).reset_index()
        g["rate"] = (g["sum"] / g["count"] * 100).round(1)

        # Sort categories (rate desc for rate view, count desc for count view)
        if view_mode == "Attrition rate (%)":
            g = g.sort_values("rate", ascending=False)
            y = g["rate"]
            y_title = "Attrition Rate (%)"
            text = g["rate"].apply(lambda v: f"{v:.1f}%")
        else:
            g = g.sort_values("count", ascending=False)
            y = g["count"]
            y_title = "Customers"
            text = g["count"].astype(int)

        fig_sel = go.Figure(data=[go.Bar(
            x=g[group_col].astype(str),
            y=y,
            text=text,
            textposition="auto",
        )])

        fig_sel.update_layout(
            title=f"{chart_title} — {view_mode}",
            xaxis_title=chart_title,
            yaxis_title=y_title,
            height=380,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig_sel, use_container_width=True)


# =============================
# Page 2 — ARM FP-GROWTH
# =============================
elif page == "🔍 Churn Patterns":

    st.title("🔍 Common Customer Behaviour Patterns")

    st.caption(
        "Identify common customer behaviour profiles and see how each profile relates to churn risk. "
        "Use this to prioritise engagement and understand typical customer segments."
    )

    st.markdown("---")

    if freq_itemsets.empty:
        st.error("Backend file missing. Run backend_generate_itemsets.py first.")
        st.stop()

    # selection
    level = st.select_slider(
        "Analysis depth",
        options=["Quick scan", "Standard patterns", "Detailed insights", "Deep analysis"],
        value="Standard patterns"
    )

    # parameters for each analysis depth level
    level_map = {
        "Quick scan": {"max_len": 2, "min_support": 0.05},
        "Standard patterns": {"max_len": 3, "min_support": 0.03},
        "Detailed insights": {"max_len": 4, "min_support": 0.015},
        "Deep analysis": {"max_len": 5, "min_support": 0.01},
    }

    params = level_map[level]

    # apply backend algo with filter
    view = freq_itemsets.copy()

    view = view[
        (view["item_count"] <= params["max_len"]) &
        (view["support"] >= params["min_support"])
    ]

    if view.empty:
        st.warning("No patterns found at this depth. Try deeper analysis.")
        st.stop()

    # default sorting = banker logic
    view = view.sort_values(
        ["churn_rate", "customers"],
        ascending=[False, False]
    )

    view = view.head(30)

    show = view.rename(columns={
        "items": "Customer Profile",
        "customers": "Customers", 
        "support": "Portfolio Share", #how many precentage is this combination happens among the customer
        "churn_rate": "Churn Risk" 
    })

    show["Portfolio Share"] = (show["Portfolio Share"] * 100).round(2)
    show["Churn Risk"] = (show["Churn Risk"] * 100).round(2)

    st.subheader("📊 Common customer profiles")

    st.dataframe(
        show[["Customer Profile", "Customers", "Portfolio Share", "Churn Risk"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Portfolio Share": st.column_config.ProgressColumn(
                "Portfolio Share (%)",
                min_value=0,
                max_value=100
            ),
            "Churn Risk": st.column_config.ProgressColumn(
                "Churn Risk (%)",
                min_value=0,
                max_value=100
            ),
        }
    )

    st.subheader("📈 Visual overview")

    fig = px.bar(
        show.sort_values("Churn Risk", ascending=True),
        x="Churn Risk",
        y="Customer Profile",
        orientation="h"
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("💼 How bankers can use this")

    st.write("• Each row represents a common customer behaviour profile.")
    st.write("• Profiles with higher churn risk should be prioritised for engagement.")
    st.write("• Stable profiles help define benchmark customer behaviour.")

# =============================
# Page 3 — Predict & Explain (OPTIMIZED)
# =============================
elif page == "🎯 Predict & Explain":
    st.title("🎯 Predict & Explain Customer Churn Risk")
    st.caption(f"Powered by **{best_model_key}** model with {metrics.loc[best_model_key, 'roc_auc']:.1%} accuracy")
    
    st.markdown("---")
    
    mode = st.radio(
        "Choose Analysis Mode",
        [" Risk Check (by Customer ID)", " Client Simulator"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if mode.startswith("🔍"):
        st.subheader("🔍 Quick Risk Check")
        st.markdown("Enter a customer ID to predict their churn risk")
        
        if "CLIENTNUM" not in df_scored.columns:
            st.error("❌ CLIENTNUM column not found in dataset.")
        else:
            col1, col2 = st.columns([2, 3])
            
            with col1:
                clientnum = st.text_input("Customer ID (CLIENTNUM)", value="", placeholder="Enter customer ID...")
            
            if clientnum.strip():
                try:
                    cid = int(clientnum.strip())
                    match = df_scored[df_scored["CLIENTNUM"] == cid]
                    
                    if match.empty:
                        st.warning("⚠️ No customer found with this ID.")
                    else:
                        row = match.iloc[0]
                        prob = float(row["churn_prob"])
                        band, label = prob_to_band(prob, risk_bands)
                        
                        # Display risk level with color coding
                        if band == "High":
                            risk_class = "risk-high"
                            icon = "🔴"
                        elif band == "Medium":
                            risk_class = "risk-medium"
                            icon = "🟡"
                        else:
                            risk_class = "risk-low"
                            icon = "🟢"
                        
                        st.markdown(f"### {icon} <span class='{risk_class}'>{band} Risk</span> — {label}", unsafe_allow_html=True)
                        
                        # Risk gauge
                        risk_bar(prob)
                        
                        st.markdown("---")
                        
                        # Two column layout for signals and actions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🚨 Risk Signals")
                            signals = highlight_signals(row, signal_profile)
                            if signals:
                                for s in signals:
                                    st.markdown(f"- {s}")
                            else:
                                st.info("✅ No strong warning signals detected")
                        
                        with col2:
                            st.markdown("### 💡 Recommended Actions")
                            if band == "High":
                                st.markdown("""
                                - 📞 **Immediate contact** recommended
                                - 🎁 Offer targeted retention incentive
                                - 🔍 Review recent service complaints
                                - 📊 Analyze activity decline patterns
                                """)
                            elif band == "Medium":
                                st.markdown("""
                                - 📧 Send value reminder communication
                                - 📈 Monitor activity over next 30 days
                                - 🎯 Consider cross-sell opportunities
                                - 💬 Gather feedback on satisfaction
                                """)
                            else:
                                st.markdown("""
                                - ✅ Maintain current relationship
                                - 🎯 Explore cross-sell based on needs
                                - 📊 Continue regular monitoring
                                - 🌟 Consider for loyalty programs
                                """)
                
                except ValueError:
                    st.error("❌ Customer ID must be numeric")
    
    else:
        st.subheader(" Client Simulator")
        st.markdown("Adjust customer characteristics to estimate churn likelihood for hypothetical scenarios")
        
        # Categorical options from dataset
        def cat_options(col):
            return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else []
        
        # Pre-compute all options (OPTIMIZATION - avoid recomputing in form)
        gender_opts = cat_options("Gender") or ["Unknown"]
        education_opts = cat_options("Education_Level") or ["Unknown"]
        marital_opts = cat_options("Marital_Status") or ["Unknown"]
        income_opts = cat_options("Income_Category") or ["Unknown"]
        card_opts = cat_options("Card_Category") or ["Unknown"]
        
        max_months = int(df["Months_on_book"].max()) if "Months_on_book" in df.columns else 60
        max_trans_ct = int(df["Total_Trans_Ct"].max()) if "Total_Trans_Ct" in df.columns else 200
        max_trans_amt = int(df["Total_Trans_Amt"].max()) if "Total_Trans_Amt" in df.columns else 20000
        max_contacts = int(df["Contacts_Count_12_mon"].max()) if "Contacts_Count_12_mon" in df.columns else 20
        
        # Create form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Demographics**")
            Gender = st.selectbox("Gender", gender_opts)
            Education_Level = st.selectbox("Education Level", education_opts)
            Marital_Status = st.selectbox("Marital Status", marital_opts)
        
        with col2:
            st.markdown("**💳 Financial Details**")
            Income_Category = st.selectbox("Income Category", income_opts)
            Card_Category = st.selectbox("Card Category", card_opts)
            Months_on_book = st.slider("Months with Bank", 0, max_months, 36)
        
        with col3:
            st.markdown("**📊 Activity Metrics**")
            Total_Trans_Ct = st.slider("Transaction Count", 0, max_trans_ct, 60)
            Total_Trans_Amt = st.slider("Transaction Amount ($)", 0, max_trans_amt, 4000)
            Contacts_Count_12_mon = st.slider("Service Contacts (12M)", 0, max_contacts, 2)
        
        if st.button("🎯 Calculate Risk", type="primary", use_container_width=True):
            # Build single-row dataframe
            row = {}
            for c, v in [
                ("Gender", Gender),
                ("Education_Level", Education_Level),
                ("Marital_Status", Marital_Status),
                ("Income_Category", Income_Category),
                ("Card_Category", Card_Category),
                ("Months_on_book", Months_on_book),
                ("Total_Trans_Ct", Total_Trans_Ct),
                ("Total_Trans_Amt", Total_Trans_Amt),
                ("Contacts_Count_12_mon", Contacts_Count_12_mon),
            ]:
                if c in df.columns:
                    row[c] = v
            
            X_raw = pd.DataFrame([row])
            
            # Predict (using cached preprocessing)
            scaler = best_pack["scaler"]
            model = best_pack["model"]
            X_m = preprocess_X_raw(X_raw, spec, scaler)
            prob = float(model.predict_proba(X_m)[:, 1][0])
            
            band, label = prob_to_band(prob, risk_bands)
            
            # Display results
            if band == "High":
                risk_class = "risk-high"
                icon = "🔴"
            elif band == "Medium":
                risk_class = "risk-medium"
                icon = "🟡"
            else:
                risk_class = "risk-low"
                icon = "🟢"
            
            st.markdown("---")
            st.markdown(f"### {icon} <span class='{risk_class}'>Predicted: {band} Risk</span> — {label}", unsafe_allow_html=True)
            
            risk_bar(prob)
            
            # Show signals
            temp = pd.Series({**row, "churn_prob": prob})
            signals = highlight_signals(temp, signal_profile)
            
            st.markdown("### 🚨 Detected Signals")
            if signals:
                for s in signals:
                    st.markdown(f"- {s}")
            else:
                st.info("✅ No strong warning signals detected from input parameters")

# =============================
# Page 4 — Advanced Settings
# =============================
else:
    st.title("⚙️ Advanced Settings & Technical Details")
    st.caption("Deep dive into model performance and technical configurations")
    
    st.markdown("---")
    
    # Model Performance Section
    st.subheader("📊 Model Performance Comparison")
    
    if metrics.empty:
        st.info("No metrics files found.")
    else:
        # Display metrics table
        metrics_display = metrics.copy()
        metrics_display = (metrics_display * 100).round(2)
        
        st.dataframe(
            metrics_display,
            use_container_width=True,
            column_config={
                "accuracy": st.column_config.ProgressColumn("Accuracy (%)", format="%.2f%%", min_value=0, max_value=100),
                "precision": st.column_config.ProgressColumn("Precision (%)", format="%.2f%%", min_value=0, max_value=100),
                "recall": st.column_config.ProgressColumn("Recall (%)", format="%.2f%%", min_value=0, max_value=100),
                "f1": st.column_config.ProgressColumn("F1 Score (%)", format="%.2f%%", min_value=0, max_value=100),
                "roc_auc": st.column_config.ProgressColumn("ROC AUC (%)", format="%.2f%%", min_value=0, max_value=100),
            }
        )
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("🔧 Model Selection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        chosen = st.selectbox(
            "Select Model",
            options=list(MODEL_FILES.keys()),
            index=list(MODEL_FILES.keys()).index(best_model_key) if best_model_key in MODEL_FILES else 0
        )
        
        pack = models_pack[chosen]
        st.success(f"✅ Currently using: **{chosen}**")
    
    # ROC Curves
    st.markdown("---")
    st.subheader("📈 ROC Curves Analysis")
    
    roc_path = os.path.join(ART_DIR, "roc_all_models.json")
    if os.path.exists(roc_path):
        with open(roc_path, "r", encoding="utf-8") as f:
            roc_data = json.load(f)
        
        pick = st.multiselect(
            "Select models to display",
            options=list(roc_data.keys()),
            default=[chosen] if chosen in roc_data else list(roc_data.keys())[:2]
        )
        
        if pick:
            fig = go.Figure()
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Random Classifier'
            ))
            
            # Add ROC curves
            colors = ['#667eea', '#f093fb', '#38ef7d', '#f5576c']
            for idx, name in enumerate(pick):
                if name not in roc_data:
                    continue
                fpr = roc_data[name]["fpr"]
                tpr = roc_data[name]["tpr"]
                auc = roc_data[name].get("auc", None)
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f"{name} (AUC={auc:.3f})" if auc is not None else name,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
            
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                title="ROC Curves - Model Comparison",
                height=500,
                hovermode='x unified',
                legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Dataset Preview
    st.subheader("📋 Dataset Preview")
    st.caption("First 100 rows for auditing and reporting purposes")
    
    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=400
    )
    
    # Download options
    st.markdown("---")
    st.subheader("💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_scored.to_csv(index=False)
        st.download_button(
            label="📥 Download Scored Dataset (CSV)",
            data=csv,
            file_name="customer_churn_scores.csv",
            mime="text/csv",
        )
    
    with col2:
        if not metrics.empty:
            metrics_csv = metrics.to_csv()
            st.download_button(
                label="📥 Download Model Metrics (CSV)",
                data=metrics_csv,
                file_name="model_performance_metrics.csv",
                mime="text/csv",
            )