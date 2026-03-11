# app.py
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

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    st.warning("La bibliothèque 'fredapi' n'est pas installée. Le module macro-économique ne fonctionnera pas. Installez-la avec `pip install fredapi`.")
    FRED_AVAILABLE = False

# ============================================================
# Import des fonctions avancées de volatilité (quantreo)
# ============================================================
try:
    import quantreo.features_engineering as fe
    QUANTREO_AVAILABLE = True
except ImportError:
    st.error("La bibliothèque 'quantreo' n'est pas installée. Les calculs de volatilité avancés nécessitent ce package. Veuillez l'installer avec `pip install quantreo`.")
    QUANTREO_AVAILABLE = False
    # Définition de fonctions de secours simples pour éviter le crash
    class fe:
        class volatility:
            @staticmethod
            def close_to_close_volatility(df, window_size):
                return df['close'].pct_change().rolling(window_size).std()
            @staticmethod
            def parkinson_volatility(df, high_col, low_col, window_size):
                return (np.log(df[high_col]/df[low_col])**2).rolling(window_size).mean()**0.5
            @staticmethod
            def rogers_satchell_volatility(df, high_col, low_col, open_col, close_col, window_size):
                return ( (np.log(df[high_col]/df[close_col]) * np.log(df[high_col]/df[open_col]) +
                          np.log(df[low_col]/df[close_col]) * np.log(df[low_col]/df[open_col]) )
                       ).rolling(window_size).mean()**0.5
            @staticmethod
            def yang_zhang_volatility(df, high_col, low_col, open_col, close_col, window_size):
                # Version simplifiée, non exacte
                return df['close'].pct_change().rolling(window_size).std()

# ============================================================
# Import pytrends pour le module Google Trends
# ============================================================
try:
    from pytrends import dailydata
    from pytrends.request import TrendReq
    import datetime
    import time
    PYTENDS_AVAILABLE = True
except ImportError:
    st.warning("La bibliothèque 'pytrends' n'est pas installée. Le module Google Trends ne fonctionnera pas. Installez-la avec `pip install pytrends`.")
    PYTENDS_AVAILABLE = False

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Quant Market Terminal",
    layout="wide",
)

st.title("📊 Quant Market Terminal")

# ============================================================
# STYLE BLOOMBERG
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
# CHARGEMENT DES DONNÉES (avec cache)
# ============================================================
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, period="max", interval="1d", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [c.lower() for c in df.columns]
    # Prix moyen OHLC
    df["avg"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    return df

# ============================================================
# CHARGEMENT DES DONNÉES MACROS (avec cache)
# ============================================================
@st.cache_data(ttl=43200)
def load_macro_data(api_key):
    """
    Télécharge les indicateurs macro US depuis FRED.
    Nécessite une clé API.
    """
    if not FRED_AVAILABLE:
        return None

    try:
        fred = Fred(api_key=api_key)

        # Télécharger les séries brutes
        cpi = fred.get_series('CPIAUCSL')          # Inflation (CPI)
        fedfunds = fred.get_series('FEDFUNDS')     # Taux d'intérêt effectif
        unrate = fred.get_series('UNRATE')         # Taux de chômage
        gdp = fred.get_series('GDPC1')             # PIB réel (chaîné)

        # Créer un DataFrame avec toutes les séries
        df = pd.DataFrame({
            'CPI': cpi,
            'FedFunds': fedfunds,
            'Unemployment': unrate,
            'GDP': gdp
        })

        # Calculer les variations annuelles (en pourcentage)
        df['Inflation (CPI YoY)'] = df['CPI'].pct_change(12) * 100   # 12 mois → YoY
        df['Croissance PIB (YoY)'] = df['GDP'].pct_change(4) * 100    # 4 trimestres → YoY

        # Conserver uniquement les colonnes finales et renommer
        df_final = df[['Inflation (CPI YoY)', 'FedFunds', 'Unemployment', 'Croissance PIB (YoY)']].copy()
        df_final.columns = [
            'Inflation (CPI YoY)',
            "Taux d'Intérêt (Fed Funds)",
            'Taux de Chômage',
            'Croissance PIB (YoY)'
        ]

        return df_final.dropna()

    except Exception as e:
        st.error(f"Erreur lors du chargement des données macro: {e}")
        return None

# ============================================================
# CHARGEMENT DES DONNÉES GOOGLE TRENDS (avec cache)
# ============================================================
@st.cache_data(ttl=21600)  # 6 heures de cache
def load_trend_data(keyword, year_from, mon_from, year_to, mon_to):
    """
    Télécharge les données Google Trends pour un mot-clé sur une période donnée.
    """
    if not PYTENDS_AVAILABLE:
        return None
    try:
        # Petit délai pour éviter de surcharger l'API
        time.sleep(2)
        df = dailydata.get_daily_data(
            keyword,
            start_year=year_from,
            start_mon=mon_from,
            stop_year=year_to,
            stop_mon=mon_to
        )
        # La colonne porte le nom du mot-clé
        if keyword not in df.columns:
            st.error("La colonne des données n'a pas été trouvée.")
            return None
        return df
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données Google Trends : {e}")
        return None

# ============================================================
# INDICATEURS DE TENDANCE (fenêtre 31)
# ============================================================
def compute_trend_features(df):
    w = 31
    df['rsi'] = ta.momentum.RSIIndicator(df['avg'], window=w).rsi()
    df['stoch'] = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['avg'], window=w
    ).stoch_signal()
    df['willr'] = ta.momentum.WilliamsRIndicator(
        df['high'], df['low'], df['avg'], lbp=w
    ).williams_r()
    return df

# ============================================================
# MODÈLE DE RÉGIMES DE TENDANCE (Bayesian GMM + PCA)
# ============================================================
def compute_trend_regime(df, n_clusters):
    df = compute_trend_features(df)
    df.dropna(inplace=True)
    features = ['rsi', 'stoch', 'willr']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    pca = KernelPCA(n_components=1)
    df['trend_pca'] = pca.fit_transform(X)
    model = BayesianGaussianMixture(n_components=n_clusters, random_state=0)
    df['regime'] = model.fit_predict(df[['trend_pca']])
    return df

# ============================================================
# INDICATEURS DE VOLATILITÉ AVANCÉS (avec quantreo)
# ============================================================
def compute_volatility_features(df):
    if not QUANTREO_AVAILABLE:
        st.warning("quantreo non disponible, utilisation de calculs simplifiés.")
    for w in [5, 10, 20, 50, 100]:
        df[f'vol_close_to_close_{w}'] = fe.volatility.close_to_close_volatility(df, window_size=w)
        df[f'vol_parkinson_{w}'] = fe.volatility.parkinson_volatility(df, high_col='high', low_col='low', window_size=w)
        df[f'vol_rogers_satchell_{w}'] = fe.volatility.rogers_satchell_volatility(
            df, high_col='high', low_col='low', open_col='open', close_col='close', window_size=w)
        df[f'vol_yang_zhang_{w}'] = fe.volatility.yang_zhang_volatility(
            df, high_col='high', low_col='low', open_col='open', close_col='close', window_size=w)
    return df

# ============================================================
# MODÈLE DE RÉGIMES DE VOLATILITÉ (KMeans + PCA)
# ============================================================
def compute_volatility_regime(df, n_clusters):
    df = compute_volatility_features(df)
    df.dropna(inplace=True)
    features = [c for c in df.columns if c.startswith('vol_') and 'volatility' not in c]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    pca = KernelPCA(n_components=1)
    df['vol_pca'] = pca.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=0)
    df['regime'] = model.fit_predict(df[['vol_pca']])
    return df

# ============================================================
# MODÈLE DE MOMENTUM (score sur 20/60/120/180/240 jours)
# ============================================================
def compute_momentum(tickers):
    data = {}
    for t in tickers:
        df = load_data(t)
        if df is None:
            continue
        for w in [20, 60, 120, 180, 240]:
            df[f'perf_{w}'] = df['avg'].pct_change(w)
        df['score'] = df[[f'perf_{w}' for w in [20,60,120,180,240]]].mean(axis=1)
        data[t] = df[['score']]
    if not data:
        return None
    df = pd.concat(data, axis=1)
    df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

# ============================================================
# VISUALISATIONS
# ============================================================
def filter_years(df, years):
    start_idx = len(df) - years * 252
    if start_idx < 0:
        start_idx = 0
    return df.iloc[start_idx:]

def plot_regime_with_pca(df, regime_col, pca_col, title):
    """
    Crée deux figures :
    - Figure 1 : prix + clusters + moyennes mobiles
    - Figure 2 : composante PCA avec zones positives/négatives
    """
    # Figure 1 : prix et clusters
    fig1, ax1 = plt.subplots(figsize=(16, 6), dpi=200)
    ax1.plot(df.index, df['avg'], color='#F0F0F0', linewidth=1.5, alpha=0.9, label='Avg Price')
    clusters = df[regime_col].unique()
    cmap = plt.cm.get_cmap('tab10', len(clusters))
    for c in clusters:
        temp = df[df[regime_col] == c]
        ax1.scatter(temp.index, temp['avg'], s=15, color=cmap(c), label=f'Regime {c}')
    ax1.plot(df.index, df['avg'].rolling(50).mean(), '--', color='#E69F00', linewidth=1.2, label='50D MA')
    ax1.plot(df.index, df['avg'].rolling(200).mean(), '--', color='#56B4E9', linewidth=1.2, label='200D MA')
    ax1.set_title(title, fontsize=16, weight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    # Figure 2 : composante PCA
    fig2, ax2 = plt.subplots(figsize=(16, 3), dpi=200)
    ax2.fill_between(df.index, 0, df[pca_col], where=df[pca_col]>=0, color='#00FF00', alpha=0.3, label='Positive')
    ax2.fill_between(df.index, 0, df[pca_col], where=df[pca_col]<0, color='#FF0000', alpha=0.3, label='Negative')
    ax2.plot(df.index, df[pca_col], color='#FFFFFF', linewidth=1.2, label='PCA')
    ax2.axhline(0, linestyle='-', color='gray', alpha=0.6)
    ax2.set_title(f'{title} – Composante PCA', fontsize=14, weight='bold')
    ax2.set_ylabel('PCA value')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    return fig1, fig2

def plot_momentum_chart(df, title):
    fig, ax = plt.subplots(figsize=(18, 8), dpi=200)
    colors = plt.cm.gist_ncar(np.linspace(0, 1, len(df.columns)))
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], label=col, color=colors[i], linewidth=1.5)
    ax.legend(ncol=4, loc='upper left')
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_ylabel('Momentum Score')
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    return fig

def plot_macro_chart(df, title):
    """Affiche les indicateurs macroéconomiques."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=200)
    axes = axes.flatten()

    colors = ['#00FF00', '#FFA500', '#FF0000', '#56B4E9']

    for i, col in enumerate(df.columns):
        ax = axes[i]
        ax.plot(df.index, df[col], color=colors[i], linewidth=2, label=col)
        ax.fill_between(df.index, 0, df[col], alpha=0.2, color=colors[i])
        ax.set_title(col, fontsize=14, weight='bold')
        ax.set_ylabel('Valeur (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.suptitle(title, fontsize=18, weight='bold', y=1.02)
    plt.tight_layout()
    return fig

def plot_trend_chart(df, keyword, window, num):
    """
    Affiche la tendance Google Trends d'un mot-clé avec le style Bloomberg.
    """
    fig, ax = plt.subplots(figsize=(16, 6), dpi=200)

    # Série principale
    ax.plot(df.index, df[keyword], color="#F0F0F0", linewidth=1.5, alpha=0.9, label=keyword)

    # Calculs des bandes
    rolling_mean = df[keyword].rolling(window).mean()
    rolling_std = df[keyword].rolling(window).std()

    # Bande inférieure (verte)
    ax.plot(rolling_mean - num * rolling_std, color="#00FF00", linewidth=1.2, linestyle="--", label=f"Mean - {num}σ")
    # Bande supérieure (rouge)
    ax.plot(rolling_mean + num * rolling_std, color="#FF0000", linewidth=1.2, linestyle="--", label=f"Mean + {num}σ")
    # Remplissage entre les bandes
    ax.fill_between(df.index, rolling_mean - num * rolling_std, rolling_mean + num * rolling_std,
                    color="#2A2F36", alpha=0.3)

    # Titre et labels
    ax.set_title(f"Google Trends – {keyword.upper()}", fontsize=16, weight="bold")
    ax.set_ylabel("Intérêt de recherche")
    ax.legend(loc="upper left")

    # Grille
    ax.grid(True, alpha=0.3)

    # Formatage des dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    plt.tight_layout()
    return fig

# ============================================================
# LISTES DE TICKERS PAR CATÉGORIE
# ============================================================
# D'après la liste complète fournie
TICKERS_DEFAULT = [
    "QQQ", "SPY", "EFA", "EEM", "TLT", "GLD", "USO", "SLV", "VNQ",
    "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLC", "XLU",
    "XBI", "GDX", "GREK", "EWH", "EWZ", "MCHI", "EZA", "EWY", "EWU", "EWM", "EWS",
    "EWT", "EIRL", "EWI", "EWP", "EWW", "EIDO", "EWA", "EWQ", "ECH", "EWC", "EWK",
    "EWG", "EWJ", "EWD", "EWO", "EWL", "EWN", "VOO", "INDA"
]

# Groupes
ASSETS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "GLD", "USO", "SLV", "VNQ"]
SECTORS = ["XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLC", "XLU", "XBI", "GDX"]
INTERNATIONAL = [t for t in TICKERS_DEFAULT if t not in ASSETS + SECTORS and t not in ["VOO"]]
# On ajoute VOO à ASSETS pour être cohérent
ASSETS.append("VOO")

# ============================================================
# INTERFACE SIDEBAR
# ============================================================
st.sidebar.title("Market Controls")

# AJOUTER CETTE LIGNE pour récupérer la clé API FRED
FRED_API_KEY = "0b1be764964fa26152877296823e7f2d"
api_key = FRED_API_KEY

analysis = st.sidebar.selectbox(
    "Module",
    [
        "Trend Regimes",
        "Volatility Regimes",
        "Momentum Assets",
        "Momentum Sectors",
        "Momentum International",
        "US Macro Indicators",
        "Google Trends"                     # Nouveau module
    ]
)

ticker = st.sidebar.text_input("Ticker (pour régimes)", "SPY").upper()

clusters = st.sidebar.slider("Nombre de régimes", 2, 6, 4)

years = st.sidebar.number_input(
    "Années de données",
    min_value=1,
    max_value=10,
    value=1
)

# ============================================================
# CORPS PRINCIPAL
# ============================================================
if analysis == "Trend Regimes":
    df = load_data(ticker)
    if df is None:
        st.error(f"Impossible de charger {ticker}")
    else:
        df = compute_trend_regime(df, clusters)
        df = filter_years(df, years)
        fig1, fig2 = plot_regime_with_pca(df, 'regime', 'trend_pca', f"{ticker} – Régimes de Tendance")
        st.pyplot(fig1)
        st.pyplot(fig2)

elif analysis == "Volatility Regimes":
    df = load_data(ticker)
    if df is None:
        st.error(f"Impossible de charger {ticker}")
    else:
        df = compute_volatility_regime(df, clusters)
        df = filter_years(df, years)
        fig1, fig2 = plot_regime_with_pca(df, 'regime', 'vol_pca', f"{ticker} – Régimes de Volatilité")
        st.pyplot(fig1)
        st.pyplot(fig2)

elif analysis == "Momentum Assets":
    df = compute_momentum(ASSETS)
    if df is None:
        st.error("Aucune donnée disponible pour les actifs sélectionnés.")
    else:
        df = filter_years(df, years)
        fig = plot_momentum_chart(df, "Momentum Cross‑Asset (Actifs)")
        st.pyplot(fig)

elif analysis == "Momentum Sectors":
    df = compute_momentum(SECTORS)
    if df is None:
        st.error("Aucune donnée disponible pour les secteurs.")
    else:
        df = filter_years(df, years)
        fig = plot_momentum_chart(df, "Momentum Sectoriel (ETFs)")
        st.pyplot(fig)

elif analysis == "Momentum International":
    df = compute_momentum(INTERNATIONAL)
    if df is None:
        st.error("Aucune donnée disponible pour les ETFs internationaux.")
    else:
        df = filter_years(df, years)
        fig = plot_momentum_chart(df, "Momentum International")
        st.pyplot(fig)

elif analysis == "US Macro Indicators":
    if not api_key:
        st.warning("Veuillez entrer votre clé API FRED dans la barre latérale pour accéder aux données macro.")
    else:
        with st.spinner("Chargement des indicateurs macroéconomiques..."):
            df_macro = load_macro_data(api_key)
            if df_macro is not None:
                df_macro = filter_years(df_macro, years)
                fig = plot_macro_chart(df_macro, "Indicateurs Macroéconomiques US")
                st.pyplot(fig)
                st.subheader("Dernières valeurs")
                st.dataframe(df_macro.tail().round(2))
            else:
                st.error("Impossible de charger les données macro. Vérifiez votre clé API.")

# --- NOUVEAU : Google Trends ---
elif analysis == "Google Trends":
    if not PYTENDS_AVAILABLE:
        st.error("La bibliothèque 'pytrends' n'est pas installée. Veuillez l'installer avec `pip install pytrends`.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Paramètres Google Trends")
        keyword = st.sidebar.text_input("Mot-clé", "stock market crash").lower().replace(" ", "_")
        # Période (année/mois début et fin)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            year_from = st.number_input("Année début", min_value=2004, max_value=datetime.datetime.now().year, value=datetime.datetime.now().year-1)
            mon_from = st.number_input("Mois début", min_value=1, max_value=12, value=datetime.datetime.now().month)
        with col2:
            year_to = st.number_input("Année fin", min_value=2004, max_value=datetime.datetime.now().year, value=datetime.datetime.now().year)
            mon_to = st.number_input("Mois fin", min_value=1, max_value=12, value=datetime.datetime.now().month)

        window = st.sidebar.slider("Fenêtre moyenne mobile (jours)", 5, 200, 50)
        num_std = st.sidebar.number_input("Nombre d'écarts-types", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

        if st.sidebar.button("Lancer l'analyse"):
            with st.spinner("Téléchargement des données Google Trends..."):
                df_trend = load_trend_data(keyword, year_from, mon_from, year_to, mon_to)
                if df_trend is not None and not df_trend.empty:
                    # On peut éventuellement filtrer les années, mais les données sont déjà sur la période choisie
                    fig = plot_trend_chart(df_trend, keyword, window, num_std)
                    st.pyplot(fig)

                    # Afficher un aperçu des données
                    st.subheader("Aperçu des données")
                    st.dataframe(df_trend[[keyword, 'isPartial']].tail(10))
                else:
                    st.error("Impossible de charger les données. Vérifiez votre mot-clé ou la période.")

st.markdown("---")
st.caption("Quant Research Terminal • Streamlit Prototype")
