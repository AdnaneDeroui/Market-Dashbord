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
@st.cache_data(ttl=43200) # Cache de 12 heures car les données macro changent moins souvent
def load_macro_data(api_key):
    """
    Télécharge les indicateurs macro US depuis FRED.
    Nécessite une clé API.
    """
    if not FRED_AVAILABLE:
        return None

    try:
        fred = Fred(api_key=api_key)

        # Dictionnaire des séries à télécharger
        series_dict = {
            'Inflation (CPI YoY)': 'CPIAUCSL',
            'Taux d\'Intérêt (Fed Funds)': 'FEDFUNDS',
            'Taux de Chômage': 'UNRATE',
            'PIB Réel (GDP)': 'GDPC1'
        }

        data_frames = []
        for name, series_id in series_dict.items():
            # Télécharger les données
            series_data = fred.get_series(series_id)
            df_series = series_data.to_frame(name=name)
            data_frames.append(df_series)

        # Concaténer toutes les séries sur l'axe des colonnes
        from functools import reduce
        df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), data_frames)

        # Calculer les variations annuelles pour l'inflation et le PIB
        df['Inflation (CPI YoY)'] = df['Inflation (CPI YoY)'].pct_change(12) * 100  # Changement sur 12 mois pour le YoY
        df['Croissance PIB (YoY)'] = df['PIB Réel (GDP)'].pct_change(4) * 100 # Changement sur 4 trimestres

        # On supprime les colonnes originales si on le souhaite
        df = df.drop(columns=['Inflation (CPI YoY)_x', 'PIB Réel (GDP)']) # Attention aux noms après le merge
        df = df.rename(columns={'Inflation (CPI YoY)_y': 'Inflation (CPI YoY)'})
        
        # Nettoyer et garder seulement les colonnes qui nous intéressent
        df = df[['Inflation (CPI YoY)', 'Taux d\'Intérêt (Fed Funds)', 'Taux de Chômage', 'Croissance PIB (YoY)']]

        return df.dropna()

    except Exception as e:
        st.error(f"Erreur lors du chargement des données macro: {e}")
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
INTERNATIONAL = [t for t in TICKERS_DEFAULT if t not in ASSETS + SECTORS and t not in ["VOO"]]  # VOO est US, on le laisse dans INTERNATIONAL pour l'exemple
# On ajoute VOO à ASSETS pour être cohérent
ASSETS.append("VOO")

# ============================================================
# INTERFACE SIDEBAR
# ============================================================
st.sidebar.title("Market Controls")

analysis = st.sidebar.selectbox(
    "Module",
    [
        "Trend Regimes",
        "Volatility Regimes",
        "Momentum Assets",
        "Momentum Sectors",
        "Momentum International",
        "US Macro Indicators"  
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

else:  # Momentum International
    df = compute_momentum(INTERNATIONAL)
    if df is None:
        st.error("Aucune donnée disponible pour les ETFs internationaux.")
    else:
        df = filter_years(df, years)
        fig = plot_momentum_chart(df, "Momentum International")
        st.pyplot(fig)
        
# --- NOUVEAU : Macro Indicators ---
elif analysis == "US Macro Indicators":
    if not api_key:
        st.warning("Veuillez entrer votre clé API FRED dans la barre latérale pour accéder aux données macro.")
    else:
        with st.spinner("Chargement des indicateurs macroéconomiques..."):
            df_macro = load_macro_data(api_key)
            if df_macro is not None:
                df_macro = filter_years(df_macro, years)  # Réutilise votre fonction existante
                fig = plot_macro_chart(df_macro, "Indicateurs Macroéconomiques US")
                st.pyplot(fig)
                
                # Afficher un tableau des dernières valeurs
                st.subheader("Dernières valeurs")
                st.dataframe(df_macro.tail().round(2))
            else:
                st.error("Impossible de charger les données macro. Vérifiez votre clé API.")

st.markdown("---")
st.caption("Quant Research Terminal • Streamlit Prototype")
