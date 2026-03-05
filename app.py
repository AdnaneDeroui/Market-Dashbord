import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import quantreo.features_engineering as fe
import ta
import warnings
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(page_title="Market Analysis Dashboard", layout="wide")
st.title("📈 Market Analysis Dashboard")

# Mise en cache des données pour éviter de télécharger à chaque interaction
@st.cache_data(ttl=3600)  # cache valide 1 heure
def load_data(ticker):
    df = yf.download(ticker, interval='1d', period='max', progress=False)
    if df.empty:
        return None
    # Nettoyage des colonnes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [col.lower() for col in df.columns]
    return df

# Fonctions d'analyse (adaptées pour retourner les figures)
def plot_trend_clusters(ticker, n_clusters=4, years=1):
    df = load_data(ticker)
    if df is None:
        st.error(f"Aucune donnée trouvée pour {ticker}")
        return None
    required = ['close', 'high', 'low', 'open', 'volume']
    if not all(col in df.columns for col in required):
        st.error(f"Colonnes manquantes pour {ticker}")
        return None

    # Calcul des indicateurs de tendance
    windows = [31]
    for w in windows:
        df[f'vol_rsi_{w}'] = ta.momentum.RSIIndicator(df['close'], window=w).rsi()
        df[f"vol_stoch_{w}"] = ta.momentum.StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"], window=w
        ).stoch_signal()
        df[f"vol_willr_{w}"] = ta.momentum.WilliamsRIndicator(
            df["high"], df["low"], df["close"], lbp=w
        ).williams_r()

    df.dropna(inplace=True)

    vol_features = [col for col in df.columns if col.startswith('vol_') and 'volatility' not in col]
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    scaler = StandardScaler()
    scaler.fit(train_df[vol_features])
    scaled_features = scaler.transform(df[vol_features])

    pca = KernelPCA(n_components=1)
    pca.fit(scaler.transform(train_df[vol_features]))
    df['volatility_pca'] = pca.transform(scaled_features)

    # Clustering avec BayesianGaussianMixture
    bgm = BayesianGaussianMixture(n_components=n_clusters, random_state=42)
    df['volatility_cluster'] = bgm.fit_predict(df[['volatility_pca']])

    # Filtre sur les dernières années
    start_idx = len(df) - (years * 252)
    plot_df = df.iloc[start_idx:].copy()

    # Création de la figure
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    cluster_colors = [cmap(i) for i in range(n_clusters)]

    ax.plot(plot_df.index, plot_df["close"], color="black", linewidth=1.4, alpha=0.6, label="Close Price")

    for cluster in range(n_clusters):
        cluster_data = plot_df[plot_df["volatility_cluster"] == cluster]
        ax.scatter(cluster_data.index, cluster_data["close"],
                   s=18, color=cluster_colors[cluster], alpha=0.85, label=f"Cluster {cluster}")

    ax.plot(plot_df.index, plot_df["close"].rolling(50).mean(),
            linestyle="--", linewidth=1.2, color="#E69F00", alpha=0.9, label="50D MA")
    ax.plot(plot_df.index, plot_df["close"].rolling(200).mean(),
            linestyle="--", linewidth=1.2, color="#0072B2", alpha=0.9, label="200D MA")

    ax.set_title(f"{ticker} – Trend Regimes (BayesianGaussianMixture)", fontsize=16, weight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Deuxième figure pour la composante PCA
    fig2, ax2 = plt.subplots(figsize=(16, 4), dpi=300)
    ax2.plot(plot_df.index, plot_df["volatility_pca"], color="#2C3E50", linewidth=1.4)
    ax2.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.6)
    ax2.set_title(f"{ticker} – Trend PCA Component", fontsize=14, weight="bold", pad=12)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Kernel PCA (1st Component)", fontsize=11)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig, fig2

def plot_volatility_clusters(ticker, n_clusters=4, years=1):
    df = load_data(ticker)
    if df is None:
        st.error(f"Aucune donnée trouvée pour {ticker}")
        return None
    required = ['close', 'high', 'low', 'open', 'volume']
    if not all(col in df.columns for col in required):
        st.error(f"Colonnes manquantes pour {ticker}")
        return None

    # Calcul des mesures de volatilité
    for w in [5, 10, 20, 50, 100]:
        df[f'vol_close_to_close_{w}'] = fe.volatility.close_to_close_volatility(df, window_size=w)
        df[f'vol_parkinson_{w}'] = fe.volatility.parkinson_volatility(df, high_col='high', low_col='low', window_size=w)
        df[f'vol_rogers_satchell_{w}'] = fe.volatility.rogers_satchell_volatility(
            df, high_col='high', low_col='low', open_col='open', close_col='close', window_size=w)
        df[f'vol_yang_zhang_{w}'] = fe.volatility.yang_zhang_volatility(
            df, high_col='high', low_col='low', open_col='open', close_col='close', window_size=w)

    df.dropna(inplace=True)

    vol_features = [col for col in df.columns if col.startswith('vol_') and 'volatility' not in col]
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    scaler = StandardScaler()
    scaler.fit(train_df[vol_features])
    scaled_features = scaler.transform(df[vol_features])

    pca = KernelPCA(n_components=1)
    pca.fit(scaler.transform(train_df[vol_features]))
    df['volatility_pca'] = pca.transform(scaled_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['volatility_cluster'] = kmeans.fit_predict(df[['volatility_pca']])

    start_idx = len(df) - (years * 252)
    plot_df = df.iloc[start_idx:].copy()

    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    cluster_colors = [cmap(i) for i in range(n_clusters)]

    ax.plot(plot_df.index, plot_df["close"], color="black", linewidth=1.4, alpha=0.6, label="Close Price")

    for cluster in range(n_clusters):
        cluster_data = plot_df[plot_df["volatility_cluster"] == cluster]
        ax.scatter(cluster_data.index, cluster_data["close"],
                   s=18, color=cluster_colors[cluster], alpha=0.85, label=f"Cluster {cluster}")

    ax.plot(plot_df.index, plot_df["close"].rolling(50).mean(),
            linestyle="--", linewidth=1.2, color="#E69F00", alpha=0.9, label="50D MA")
    ax.plot(plot_df.index, plot_df["close"].rolling(200).mean(),
            linestyle="--", linewidth=1.2, color="#0072B2", alpha=0.9, label="200D MA")

    ax.set_title(f"{ticker} – Volatility Regimes (K-Means)", fontsize=16, weight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(16, 4), dpi=300)
    ax2.plot(plot_df.index, plot_df["volatility_pca"], color="#2C3E50", linewidth=1.4)
    ax2.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.6)
    ax2.set_title(f"{ticker} – Volatility PCA Component", fontsize=14, weight="bold", pad=12)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Kernel PCA (1st Component)", fontsize=11)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    ax2.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig, fig2

def plot_momentum_scores(tickers, years=1):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty:
            st.warning(f"Pas de données pour {ticker}, ignoré.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        # Calcul des performances sur différentes périodes
        for t in [20, 60, 120, 180, 240]:
            df[f"Perf_{t}d"] = df["Close"].pct_change(t)
        df["Score"] = df[[f"Perf_{t}d" for t in [20,60,120,180,240]]].mean(axis=1)
        data[ticker] = df[["Score"]].dropna()

    if not data:
        st.error("Aucune donnée valide pour les tickers sélectionnés.")
        return None

    df_concat = pd.concat(data, axis=1)
    df_concat.columns = df_concat.columns.droplevel(1)
    df_concat.dropna(inplace=True)
    if len(df_concat) == 0:
        st.error("Pas assez de données pour calculer les scores.")
        return None

    start_idx = len(df_concat) - years * 252
    if start_idx < 0:
        start_idx = 0
    fig, ax = plt.subplots(figsize=(20, 10), dpi=600)
    df_concat.iloc[start_idx:].plot(ax=ax)
    ax.legend(df_concat.iloc[start_idx:].columns, loc='upper left')
    ax.set_title("Momentum Score for Different Asset Classes")
    ax.grid()
    plt.tight_layout()
    return fig

# Interface utilisateur
st.sidebar.header("Paramètres")
option = st.sidebar.selectbox(
    "Choisissez une analyse",
    ["Trend Clusters", "Volatility Clusters", "Momentum Scores"]
)

if option == "Trend Clusters":
    ticker = st.sidebar.text_input("Ticker", value="SPY").upper()
    n_clusters = st.sidebar.slider("Nombre de clusters", min_value=2, max_value=6, value=4)
    years = st.sidebar.number_input("Nombre d'années", min_value=1, max_value=10, value=1, step=1)
    if st.sidebar.button("Lancer l'analyse"):
        with st.spinner("Calcul en cours..."):
            result = plot_trend_clusters(ticker, n_clusters, years)
            if result:
                fig1, fig2 = result
                st.pyplot(fig1)
                st.pyplot(fig2)

elif option == "Volatility Clusters":
    ticker = st.sidebar.text_input("Ticker", value="SPY").upper()
    n_clusters = st.sidebar.slider("Nombre de clusters", min_value=2, max_value=6, value=4)
    years = st.sidebar.number_input("Nombre d'années", min_value=1, max_value=10, value=1, step=1)
    if st.sidebar.button("Lancer l'analyse"):
        with st.spinner("Calcul en cours..."):
            result = plot_volatility_clusters(ticker, n_clusters, years)
            if result:
                fig1, fig2 = result
                st.pyplot(fig1)
                st.pyplot(fig2)

else:  # Momentum Scores
    tickers_default = ["QQQ", "SPY", "EFA", "EEM", "TLT", "GLD", "USO", "SLV"]
    tickers = st.sidebar.multiselect("Sélectionnez les tickers", options=tickers_default, default=tickers_default)
    years = st.sidebar.number_input("Nombre d'années", min_value=1, max_value=10, value=1, step=1)
    if st.sidebar.button("Lancer l'analyse"):
        with st.spinner("Téléchargement et calcul..."):
            fig = plot_momentum_scores(tickers, years)
            if fig:
                st.pyplot(fig)