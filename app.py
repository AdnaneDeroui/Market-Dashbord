import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import ta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Quant Market Terminal",
    layout="wide",
)

st.title("📊 Quant Market Terminal")

# ============================================================
# BLOOMBERG STYLE
# ============================================================

def set_terminal_style():

    plt.style.use("dark_background")

    rcParams.update({
        "figure.facecolor": "#0E1117",
        "axes.facecolor": "#0E1117",
        "axes.edgecolor": "#3A3F44",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "#2A2F36",
        "grid.alpha": 0.3,
        "font.size": 11
    })

set_terminal_style()

# ============================================================
# DATA ENGINE
# ============================================================

@st.cache_data(ttl=3600)
def load_data(ticker):

    df = yf.download(ticker, period="max", interval="1d", progress=False)

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.columns = [c.lower() for c in df.columns]

    df["avg"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    return df

# ============================================================
# TREND FEATURES
# ============================================================

def compute_trend_features(df):

    w = 30

    df["rsi"] = ta.momentum.RSIIndicator(df["avg"], window=w).rsi()

    df["stoch"] = ta.momentum.StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["avg"],
        window=w
    ).stoch()

    df["willr"] = ta.momentum.WilliamsRIndicator(
        df["high"],
        df["low"],
        df["avg"],
        lbp=w
    ).williams_r()

    return df

# ============================================================
# TREND REGIME MODEL
# ============================================================

def compute_trend_regime(df, n_clusters):

    df = compute_trend_features(df)

    df.dropna(inplace=True)

    features = ["rsi", "stoch", "willr"]

    scaler = StandardScaler()

    X = scaler.fit_transform(df[features])

    pca = KernelPCA(n_components=1)

    df["trend_pca"] = pca.fit_transform(X)

    model = BayesianGaussianMixture(n_components=n_clusters, random_state=0)

    df["regime"] = model.fit_predict(df[["trend_pca"]])

    return df

# ============================================================
# VOLATILITY REGIME MODEL
# ============================================================

def compute_volatility_features(df):

    for w in [5,10,20,50]:

        df[f"vol_{w}"] = df["close"].pct_change().rolling(w).std()

    return df


def compute_volatility_regime(df, n_clusters):

    df = compute_volatility_features(df)

    df.dropna(inplace=True)

    features = [c for c in df.columns if "vol_" in c]

    scaler = StandardScaler()

    X = scaler.fit_transform(df[features])

    pca = KernelPCA(n_components=1)

    df["vol_pca"] = pca.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=0)

    df["regime"] = model.fit_predict(df[["vol_pca"]])

    return df

# ============================================================
# MOMENTUM MODEL
# ============================================================

def compute_momentum(tickers):

    data = {}

    for t in tickers:

        df = load_data(t)

        if df is None:
            continue

        for w in [20,60,120,180,240]:

            df[f"perf_{w}"] = df["avg"].pct_change(w)

        df["score"] = df[[f"perf_{w}" for w in [20,60,120,180,240]]].mean(axis=1)

        data[t] = df[["score"]]

    df = pd.concat(data, axis=1)

    df.columns = df.columns.droplevel(1)

    df.dropna(inplace=True)

    return df

# ============================================================
# VISUALIZATION
# ============================================================

def filter_years(df, years):

    start_idx = len(df) - years*252

    if start_idx < 0:
        start_idx = 0

    return df.iloc[start_idx:]


def plot_regime_chart(df, regime_col, title):

    fig, ax = plt.subplots(figsize=(16,6), dpi=200)

    ax.plot(df.index, df["avg"], linewidth=1.5)

    clusters = df[regime_col].unique()

    cmap = plt.cm.get_cmap("tab10", len(clusters))

    for c in clusters:

        temp = df[df[regime_col] == c]

        ax.scatter(temp.index, temp["avg"], s=15, color=cmap(c), label=f"Regime {c}")

    ax.set_title(title)

    ax.set_yscale("log")

    ax.legend()

    ax.grid(True)

    return fig


def plot_momentum(df):

    fig, ax = plt.subplots(figsize=(18,8), dpi=200)

    colors = plt.cm.gist_ncar(np.linspace(0,1,len(df.columns)))

    for i,col in enumerate(df.columns):

        ax.plot(df.index, df[col], label=col, color=colors[i])

    ax.legend(ncol=3)

    ax.set_title("Cross Asset Momentum")

    ax.grid(True)

    return fig

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Market Controls")

analysis = st.sidebar.selectbox(

    "Module",

    [
        "Trend Regimes",
        "Volatility Regimes",
        "Momentum Assets",
        "Momentum Sectors",
        "Momentum International"
    ]
)

ticker = st.sidebar.text_input("Ticker", "SPY").upper()

clusters = st.sidebar.slider("Regimes",2,6,4)

years = st.sidebar.number_input(
    "Years of data",
    min_value=1,
    max_value=10,
    value=1
)

# ============================================================
# MOMENTUM GROUPS
# ============================================================

assets = ["SPY","QQQ","EFA","EEM","SLV","USO","VNQ","GLD"]

sectors = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLC","XLU"]

international = [
"EWH","EWZ","MCHI","EZA","EWY","EWU","EWM","EWS","EWT","EIRL","EWI","EWP",
"EWW","EIDO","EWA","EWQ","ECH","EWC","EWK","EWG","EWJ","EWD","EWO","EWL","EWN","INDA"
]

# ============================================================
# MAIN TERMINAL
# ============================================================

if analysis == "Trend Regimes":

    df = load_data(ticker)

    df = compute_trend_regime(df, clusters)

    df = filter_years(df, years)

    fig = plot_regime_chart(df,"regime",f"{ticker} Trend Regimes")

    st.pyplot(fig)


elif analysis == "Volatility Regimes":

    df = load_data(ticker)

    df = compute_volatility_regime(df, clusters)

    df = filter_years(df, years)

    fig = plot_regime_chart(df,"regime",f"{ticker} Volatility Regimes")

    st.pyplot(fig)


elif analysis == "Momentum Assets":

    df = compute_momentum(assets)

    df = filter_years(df, years)

    fig = plot_momentum(df)

    st.pyplot(fig)


elif analysis == "Momentum Sectors":

    df = compute_momentum(sectors)

    df = filter_years(df, years)

    fig = plot_momentum(df)

    st.pyplot(fig)


else:

    df = compute_momentum(international)

    df = filter_years(df, years)

    fig = plot_momentum(df)

    st.pyplot(fig)

st.markdown("---")

st.caption("Quant Research Terminal • Streamlit Prototype")
