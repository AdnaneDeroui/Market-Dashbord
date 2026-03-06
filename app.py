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
    page_title="Professional Market Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL STYLE (Bloomberg-like)
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
# FEATURE ENGINEERING
# ============================================================

def compute_trend_features(df):

    windows = [30]

    for w in windows:

        df[f"rsi_{w}"] = ta.momentum.RSIIndicator(df["avg"], window=w).rsi()

        df[f"stoch_{w}"] = ta.momentum.StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["avg"],
            window=w
        ).stoch()

        df[f"willr_{w}"] = ta.momentum.WilliamsRIndicator(
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

    features = [c for c in df.columns if "rsi" in c or "stoch" in c or "willr" in c]

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
# VISUALIZATION
# ============================================================

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


# ============================================================
# MOMENTUM MODEL
# ============================================================

def compute_momentum(tickers):

    data = {}

    for t in tickers:

        df = load_data(t)

        if df is None:
            continue

        for w in [20,60,120,200]:

            df[f"perf_{w}"] = df["avg"].pct_change(w)

        df["score"] = df[[f"perf_{w}" for w in [20,60,120,200]]].mean(axis=1)

        data[t] = df[["score"]]

    df = pd.concat(data, axis=1)

    df.columns = df.columns.droplevel(1)

    df.dropna(inplace=True)

    return df



def plot_momentum(df):

    fig, ax = plt.subplots(figsize=(18,8), dpi=200)

    colors = plt.cm.gist_ncar(np.linspace(0,1,len(df.columns)))

    for i,col in enumerate(df.columns):

        ax.plot(df.index, df[col], label=col, color=colors[i])

    ax.legend(ncol=2)

    ax.set_title("Cross Asset Momentum")

    ax.grid(True)

    return fig


# ============================================================
# UI LAYOUT (Professional Terminal)
# ============================================================

st.title("📊 Quant Market Terminal")

st.markdown("---")

# Sidebar Controls

st.sidebar.title("Market Controls")

analysis = st.sidebar.selectbox(
    "Module",
    [
        "Trend Regimes",
        "Volatility Regimes",
        "Momentum Dashboard"
    ]
)


ticker = st.sidebar.text_input("Ticker", "SPY").upper()

clusters = st.sidebar.slider("Regimes",2,6,4)


# ============================================================
# DASHBOARD HEADER METRICS
# ============================================================


df_price = load_data(ticker)

if df_price is not None:

    col1,col2,col3,col4 = st.columns(4)

    last = df_price["close"].iloc[-1]

    ret = df_price["close"].pct_change().iloc[-1]*100

    vol = df_price["close"].pct_change().rolling(20).std().iloc[-1]*np.sqrt(252)*100

    high = df_price["high"].rolling(252).max().iloc[-1]

    col1.metric("Price", f"{last:.2f}")

    col2.metric("Daily Return", f"{ret:.2f}%")

    col3.metric("20D Volatility", f"{vol:.2f}%")

    col4.metric("52W High", f"{high:.2f}")

st.markdown("---")


# ============================================================
# MAIN TERMINAL AREA
# ============================================================


if analysis == "Trend Regimes":

    df = compute_trend_regime(df_price.copy(), clusters)

    fig = plot_regime_chart(df, "regime", f"{ticker} Trend Regimes")

    st.pyplot(fig)


elif analysis == "Volatility Regimes":

    df = compute_volatility_regime(df_price.copy(), clusters)

    fig = plot_regime_chart(df, "regime", f"{ticker} Volatility Regimes")

    st.pyplot(fig)


else:

    tickers_default = [
        "SPY","QQQ","TLT","GLD","USO","VNQ",
        "XLF","XLE","XLK","XLI","XLP","XLY"
    ]

    tickers = st.sidebar.multiselect(
        "Assets",
        tickers_default,
        default=tickers_default
    )

    df = compute_momentum(tickers)

    fig = plot_momentum(df)

    st.pyplot(fig)


st.markdown("---")

st.caption("Quant Research Terminal • Streamlit Prototype")
