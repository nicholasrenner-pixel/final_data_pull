# ============================================================
#  Stock Analysis Dashboard
#  Built with Streamlit + yfinance + pandas + plotly
#  Default stock: AAPL (Apple Inc.)
# ============================================================

# ---------- IMPORTS ----------
import streamlit as st          # The web app framework
import yfinance as yf           # Pulls real stock data from Yahoo Finance
import pandas as pd             # Data tables (like Excel in Python)
import plotly.graph_objects as go  # Interactive charts
from plotly.subplots import make_subplots  # Multiple charts stacked together
from datetime import datetime, timedelta   # For date math

# ============================================================
#  PAGE CONFIG  (must be the VERY FIRST streamlit command)
# ============================================================
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",          # Use full screen width
    initial_sidebar_state="expanded",
)

# ============================================================
#  CUSTOM CSS  — makes the app look clean & professional
# ============================================================
st.markdown("""
<style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ---- Global background ---- */
    .stApp {
        background: #0d1117;
        font-family: 'DM Sans', sans-serif;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #21262d;
    }

    /* ---- All text white ---- */
    html, body, [class*="css"] {
        color: #e6edf3;
    }

    /* ---- Metric cards ---- */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700;
        color: #e6edf3 !important;
        font-family: 'DM Mono', monospace;
    }

    /* ---- Signal badge helpers ---- */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.04em;
    }
    .badge-green  { background:#1a3a2a; color:#3fb950; border:1px solid #3fb950; }
    .badge-red    { background:#3a1a1a; color:#f85149; border:1px solid #f85149; }
    .badge-yellow { background:#3a301a; color:#d29922; border:1px solid #d29922; }
    .badge-blue   { background:#1a2a3a; color:#58a6ff; border:1px solid #58a6ff; }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8b949e;
        margin: 24px 0 12px 0;
        font-weight: 600;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Input boxes ---- */
    .stTextInput > div > div > input {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #e6edf3;
        font-family: 'DM Mono', monospace;
        font-size: 15px;
        text-transform: uppercase;
    }
    .stTextInput > div > div > input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 2px rgba(88,166,255,0.15);
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: #1f6feb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        width: 100%;
        padding: 10px 0;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #388bfd;
    }

    /* ---- Plotly chart backgrounds ---- */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }

    /* ---- Divider ---- */
    hr { border-color: #21262d; }

    /* ---- Info boxes ---- */
    .info-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  HELPER FUNCTIONS
#  (Each function does ONE job — easy to understand & reuse)
# ============================================================

@st.cache_data(ttl=300)   # Cache data for 5 min so we don't hammer the API
def get_stock_data(ticker: str) -> pd.DataFrame:
    """
    Downloads 6 months of daily closing prices from Yahoo Finance.

    Parameters:
        ticker: Stock symbol like 'AAPL' or 'TSLA'

    Returns:
        DataFrame with Date and Close columns, or empty DataFrame on error.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=182)   # ~6 months back

    try:
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # yfinance can return multi-level columns — flatten them
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if raw.empty:
            return pd.DataFrame()

        df = raw[["Close"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.dropna()
        return df

    except Exception:
        return pd.DataFrame()


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 20-day and 50-day Simple Moving Averages (SMA) to the DataFrame.

    A moving average smooths out day-to-day noise so you can see the trend.
    The 20-day looks at the last 20 closing prices; the 50-day looks at 50.
    """
    df = df.copy()
    df["MA20"] = df["Close"].rolling(window=20).mean()  # 20-day average
    df["MA50"] = df["Close"].rolling(window=50).mean()  # 50-day average
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates the 14-day Relative Strength Index (RSI).

    RSI measures momentum: how fast prices are moving up vs down.
    - RSI > 70  → stock may be OVERBOUGHT (consider selling)
    - RSI < 30  → stock may be OVERSOLD  (consider buying)
    - 30–70     → neutral zone
    """
    df = df.copy()

    # Step 1: Find how much price changed each day
    delta = df["Close"].diff()

    # Step 2: Separate gains (up days) and losses (down days)
    gain = delta.clip(lower=0)      # keep only positive moves
    loss = (-delta).clip(lower=0)   # keep only negative moves (flip sign)

    # Step 3: Smooth them with a rolling 14-day average
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Step 4: RSI formula
    rs  = avg_gain / avg_loss            # Relative Strength ratio
    rsi = 100 - (100 / (1 + rs))        # Scale to 0–100

    df["RSI"] = rsi
    return df


def get_trend_label(price: float, ma20: float, ma50: float) -> str:
    """
    Returns a trend label based on price vs moving averages.

    Strong Uptrend  : Price > MA20 > MA50  (everything stacked bullishly)
    Strong Downtrend: Price < MA20 < MA50  (everything stacked bearishly)
    Mixed Trend     : Anything else
    """
    if price > ma20 > ma50:
        return "strong_uptrend"
    elif price < ma20 < ma50:
        return "strong_downtrend"
    else:
        return "mixed"


def get_rsi_signal(rsi: float) -> str:
    """
    Interprets the RSI value into a trading signal.
    """
    if rsi > 70:
        return "overbought"
    elif rsi < 30:
        return "oversold"
    else:
        return "neutral"


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates daily percentage returns.

    Return = (today's price - yesterday's price) / yesterday's price × 100
    This tells us how much % the stock moved each day.
    """
    df = df.copy()
    df["Return"] = df["Close"].pct_change() * 100   # pct_change does the math
    return df


def calculate_volatility(returns: pd.Series) -> float:
    """
    Volatility = standard deviation of daily returns.

    High volatility → big swings, more risk.
    Low volatility  → stable price, less risk.
    """
    return float(returns.std())


# ============================================================
#  CHART BUILDERS  (each returns a Plotly figure)
# ============================================================

CHART_STYLE = dict(
    plot_bgcolor  = "#0d1117",
    paper_bgcolor = "#0d1117",
    font          = dict(family="DM Sans", color="#8b949e", size=12),
    margin        = dict(l=0, r=0, t=40, b=0),
    xaxis         = dict(gridcolor="#21262d", showgrid=True, zeroline=False),
    yaxis         = dict(gridcolor="#21262d", showgrid=True, zeroline=False),
)


def build_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Candlestick chart with MA20 and MA50 overlaid.
    Uses the full OHLC data for richer candlesticks when available.
    """
    fig = make_subplots(rows=1, cols=1)

    # Price line (we always have Close)
    fig.add_trace(go.Scatter(
        x    = df.index,
        y    = df["Close"],
        name = "Price",
        line = dict(color="#58a6ff", width=2),
        hovertemplate = "$%{y:.2f}<extra></extra>",
    ))

    # 20-day moving average
    fig.add_trace(go.Scatter(
        x         = df.index,
        y         = df["MA20"],
        name      = "MA 20",
        line      = dict(color="#f0a500", width=1.5, dash="dot"),
        hovertemplate = "MA20: $%{y:.2f}<extra></extra>",
    ))

    # 50-day moving average
    fig.add_trace(go.Scatter(
        x         = df.index,
        y         = df["MA50"],
        name      = "MA 50",
        line      = dict(color="#3fb950", width=1.5, dash="dash"),
        hovertemplate = "MA50: $%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_STYLE,
        title      = f"{ticker.upper()} — Price & Moving Averages",
        title_font = dict(size=15, color="#e6edf3"),
        legend     = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d",
                          borderwidth=1, font=dict(color="#8b949e")),
        hovermode  = "x unified",
        height     = 420,
    )
    return fig


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """
    RSI line chart with overbought (70) and oversold (30) reference bands.
    """
    fig = go.Figure()

    # Shaded overbought zone (70–100)
    fig.add_hrect(y0=70, y1=100, fillcolor="#3a1a1a",
                  opacity=0.4, line_width=0)

    # Shaded oversold zone (0–30)
    fig.add_hrect(y0=0, y1=30, fillcolor="#1a3a2a",
                  opacity=0.4, line_width=0)

    # RSI line
    fig.add_trace(go.Scatter(
        x    = df.index,
        y    = df["RSI"],
        name = "RSI (14)",
        line = dict(color="#a371f7", width=2),
        hovertemplate = "RSI: %{y:.1f}<extra></extra>",
    ))

    # Horizontal reference lines
    for level, color, label in [(70, "#f85149", "Overbought"),
                                  (30, "#3fb950", "Oversold"),
                                  (50, "#8b949e", "Midline")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color,
                      line_width=1, annotation_text=f" {label} ({level})",
                      annotation_font_color=color, annotation_font_size=11)

    fig.update_layout(
        **CHART_STYLE,
        title      = "RSI — Relative Strength Index (14-day)",
        title_font = dict(size=15, color="#e6edf3"),
        yaxis      = dict(range=[0, 100], gridcolor="#21262d"),
        height     = 280,
    )
    return fig


def build_returns_chart(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of daily returns, colored green (up) or red (down).
    """
    colors = ["#3fb950" if r >= 0 else "#f85149"
              for r in df["Return"].fillna(0)]

    fig = go.Figure(go.Bar(
        x     = df.index,
        y     = df["Return"],
        marker_color = colors,
        name  = "Daily Return %",
        hovertemplate = "%{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_color="#30363d", line_width=1)

    fig.update_layout(
        **CHART_STYLE,
        title      = "Daily Returns (%)",
        title_font = dict(size=15, color="#e6edf3"),
        height     = 260,
    )
    return fig


# ============================================================
#  SIDEBAR — User inputs live here
# ============================================================

with st.sidebar:
    st.markdown("### 📈 Stock Analyzer")
    st.markdown('<div class="section-header">Configuration</div>',
                unsafe_allow_html=True)

    # Stock ticker input — default is AAPL
    ticker_input = st.text_input(
        "Ticker Symbol",
        value    = "AAPL",
        max_chars = 10,
        help     = "Enter any valid stock ticker, e.g. TSLA, MSFT, NVDA",
    ).strip().upper()

    analyze_btn = st.button("Analyze Stock")

    st.markdown("---")
    st.markdown('<div class="section-header">How to Read This</div>',
                unsafe_allow_html=True)

    st.markdown("""
    **Trend (Moving Averages)**
    - 🟢 Strong Uptrend → Price > MA20 > MA50
    - 🔴 Strong Downtrend → Price < MA20 < MA50
    - 🟡 Mixed → Everything else

    **RSI (Momentum)**
    - RSI > 70 → Overbought ⚠️ Possible Sell
    - RSI < 30 → Oversold 💡 Possible Buy
    - 30–70 → Neutral zone

    **Volatility**
    - Standard deviation of daily returns
    - Higher = bigger price swings = more risk
    """)

    st.markdown("---")
    st.caption("Data: Yahoo Finance · 6-month window")
    st.caption("Not financial advice. Educational use only.")


# ============================================================
#  MAIN CONTENT — Everything the user sees on the right
# ============================================================

# Run analysis when button clicked OR on first load (default AAPL)
# We use session_state so the data persists between interactions
if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

if analyze_btn and ticker_input:
    st.session_state.ticker = ticker_input

active_ticker = st.session_state.ticker

# ---- Page title ----
st.markdown(f"## {active_ticker} &nbsp; Stock Analysis",
            unsafe_allow_html=True)
st.markdown('<div class="section-header">6-Month Daily Data · Part 1: Individual Stock</div>',
            unsafe_allow_html=True)

# ---- Load data ----
with st.spinner(f"Fetching data for {active_ticker}…"):
    df_raw = get_stock_data(active_ticker)

# Handle bad ticker
if df_raw.empty:
    st.error(f"⚠️ Could not find data for **{active_ticker}**. "
             "Check the ticker symbol and try again.")
    st.stop()   # Stop the script — nothing else to show

# ---- Run calculations ----
df = add_moving_averages(df_raw)
df = calculate_rsi(df)
df = calculate_daily_returns(df)

# ---- Pull out the latest values (last row) ----
latest      = df.dropna(subset=["MA20", "MA50", "RSI"]).iloc[-1]
price       = float(latest["Close"])
ma20        = float(latest["MA20"])
ma50        = float(latest["MA50"])
rsi_value   = float(latest["RSI"])
trend       = get_trend_label(price, ma20, ma50)
rsi_signal  = get_rsi_signal(rsi_value)
volatility  = calculate_volatility(df["Return"].dropna())

# ---- Total return over the period ----
first_price = float(df["Close"].dropna().iloc[0])
total_return_pct = ((price - first_price) / first_price) * 100

# ============================================================
#  ROW 1: Key Metric Cards
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Current Price",  f"${price:,.2f}")
c2.metric("20-Day MA",      f"${ma20:,.2f}")
c3.metric("50-Day MA",      f"${ma50:,.2f}")
c4.metric("RSI (14)",       f"{rsi_value:.1f}")
c5.metric("6-Mo Return",    f"{total_return_pct:+.1f}%")

st.markdown("")   # spacer

# ============================================================
#  ROW 2: Signal Badges — Trend + RSI + Volatility
# ============================================================

# Map internal labels → display text & CSS class
TREND_MAP = {
    "strong_uptrend":   ("🟢 Strong Uptrend",   "badge-green"),
    "strong_downtrend": ("🔴 Strong Downtrend", "badge-red"),
    "mixed":            ("🟡 Mixed Trend",       "badge-yellow"),
}
RSI_MAP = {
    "overbought": ("⚠️ Overbought — Possible Sell", "badge-red"),
    "oversold":   ("💡 Oversold — Possible Buy",    "badge-green"),
    "neutral":    ("⚪ RSI Neutral",                 "badge-blue"),
}

trend_text, trend_cls = TREND_MAP[trend]
rsi_text,   rsi_cls   = RSI_MAP[rsi_signal]

# Volatility label (arbitrary thresholds for context)
if volatility < 1.0:
    vol_text, vol_cls = f"🔵 Low Volatility ({volatility:.2f}%)", "badge-blue"
elif volatility < 2.5:
    vol_text, vol_cls = f"🟡 Moderate Volatility ({volatility:.2f}%)", "badge-yellow"
else:
    vol_text, vol_cls = f"🔴 High Volatility ({volatility:.2f}%)", "badge-red"

b1, b2, b3 = st.columns(3)
with b1:
    st.markdown(f"""
    <div class="info-box">
        <div class="section-header" style="margin-top:0">Trend Signal</div>
        <span class="badge {trend_cls}">{trend_text}</span>
        <p style="margin:12px 0 0 0; font-size:13px; color:#8b949e;">
            Price vs MA20 vs MA50
        </p>
    </div>""", unsafe_allow_html=True)

with b2:
    st.markdown(f"""
    <div class="info-box">
        <div class="section-header" style="margin-top:0">RSI Signal</div>
        <span class="badge {rsi_cls}">{rsi_text}</span>
        <p style="margin:12px 0 0 0; font-size:13px; color:#8b949e;">
            14-day Relative Strength Index
        </p>
    </div>""", unsafe_allow_html=True)

with b3:
    st.markdown(f"""
    <div class="info-box">
        <div class="section-header" style="margin-top:0">Volatility</div>
        <span class="badge {vol_cls}">{vol_text}</span>
        <p style="margin:12px 0 0 0; font-size:13px; color:#8b949e;">
            Std dev of daily returns
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
#  ROW 3: Price Chart
# ============================================================
st.plotly_chart(build_price_chart(df, active_ticker),
                use_container_width=True)

# ============================================================
#  ROW 4: RSI + Returns side-by-side
# ============================================================
chart_left, chart_right = st.columns(2)
with chart_left:
    st.plotly_chart(build_rsi_chart(df), use_container_width=True)
with chart_right:
    st.plotly_chart(build_returns_chart(df), use_container_width=True)

st.markdown("---")

# ============================================================
#  ROW 5: Raw Data Table (collapsed by default)
# ============================================================
with st.expander("📋 View Raw Data Table"):
    display_df = df[["Close", "MA20", "MA50", "RSI", "Return"]].copy()
    display_df.columns = ["Close ($)", "MA 20", "MA 50", "RSI", "Return (%)"]
    display_df = display_df.round(2).sort_index(ascending=False)
    st.dataframe(display_df, use_container_width=True, height=300)

# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; color:#30363d; font-size:11px; margin-top:40px;">
    Educational project · Data from Yahoo Finance · Not financial advice
</div>
""", unsafe_allow_html=True)
