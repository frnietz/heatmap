import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import date, timedelta, datetime
from typing import List, Dict

st.set_page_config(page_title="Stock Heatmap Pro", layout="wide")
st.title("📊 Stock Heatmap — Pro")
st.caption("Hierarchy: Sector → Industry → Ticker • Size: Market Cap • Color: Return")

# -----------------------
# Session init & helpers
# -----------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = set()
if "alerts" not in st.session_state:
    # alerts[ticker] = {"threshold": 5.0, "direction": "up"/"down"}
    st.session_state.alerts = {}
if "last_alerts" not in st.session_state:
    st.session_state.last_alerts = []  # store recent messages

def save_store():
    store = {
        "watchlist": list(st.session_state.watchlist),
        "alerts": st.session_state.alerts
    }
    with open("store.json", "w") as f:
        json.dump(store, f)

def load_store():
    try:
        with open("store.json", "r") as f:
            obj = json.load(f)
        st.session_state.watchlist = set(obj.get("watchlist", []))
        st.session_state.alerts = obj.get("alerts", {})
    except Exception:
        pass

load_store()

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Universe & Window")
    basket = st.selectbox("Universe", ["S&P 100 (built-in)", "Custom list"], index=0)
    if basket == "Custom list":
        tickers_text = st.text_area("Tickers", "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, UNH, XOM")
        tickers = sorted(list({t.strip().upper() for t in tickers_text.replace("\n"," ").replace(";",",").replace(" ", ",").split(",") if t.strip()}))
    else:
        df_sp100 = pd.read_csv("sp100_sample.csv")
        tickers = df_sp100["Symbol"].tolist()

    st.markdown("---")
    period_choice = st.radio("Return Window", ["1D","5D","1M","3M","6M","YTD","1Y","Custom"], index=2)
    if period_choice == "Custom":
        c1, c2 = st.columns(2)
        with c1: start_date = st.date_input("Start", date.today() - timedelta(days=90))
        with c2: end_date = st.date_input("End", date.today())
        if start_date >= end_date:
            st.error("Start date must be before end date."); st.stop()
    else:
        end_date = date.today()
        start_date = {
            "1D": end_date - timedelta(days=3),
            "5D": end_date - timedelta(days=10),
            "1M": end_date - timedelta(days=35),
            "3M": end_date - timedelta(days=110),
            "6M": end_date - timedelta(days=210),
            "YTD": date(end_date.year,1,1),
            "1Y": end_date - timedelta(days=370)
        }[period_choice]

    st.markdown("---")
    st.subheader("Filters")
    min_mcap = st.number_input("Min Market Cap (USD, billions)", min_value=0.0, value=0.0, step=1.0)
    st.caption("Tip: Use fewer tickers for faster loads. yfinance provides sector/industry/market cap fields.")

# -----------------------
# Data helpers
# -----------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _info_for_ticker(t):
    tk = yf.Ticker(t)

    # Name
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    name = info.get("shortName") or info.get("longName") or t

    # Sector / Industry
    sector = info.get("sector") or "Unknown"
    industry = info.get("industry") or "Unknown"

    # Market cap — try fast_info, then info, then fallback: sharesOutstanding * previousClose
    mcap = None
    try:
        fi = getattr(tk, "fast_info", {}) or {}
        mcap = fi.get("market_cap", None)
        if mcap is None:
            mcap = info.get("marketCap", None)
        if mcap is None:
            shares = info.get("sharesOutstanding")
            prev = fi.get("previous_close") or info.get("previousClose")
            if shares and prev:
                mcap = float(shares) * float(prev)
    except Exception:
        pass

    return dict(ticker=t, name=name, sector=sector, industry=industry, market_cap=mcap)


@st.cache_data(show_spinner=False, ttl=900)
def _price_return_for_tickers(tickers, start_date, end_date):
    # add a small buffer around the window to catch missing sessions
    pad_start = start_date - timedelta(days=3)
    pad_end = end_date + timedelta(days=1)

    data = yf.download(
        tickers=tickers, start=pad_start, end=pad_end,
        auto_adjust=True, progress=False, group_by="ticker"
    )

    rets = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                s = data[(t, "Adj Close")].dropna()
            else:
                s = data["Adj Close"].dropna()

            # Trim back to the chosen window only, after the buffer
            s = s[(s.index.date >= start_date) & (s.index.date <= end_date)]
            # Fail-safe: if still <2 points, keep last 2 available from buffer
            if len(s) < 2:
                s = s.tail(2)

            if len(s) >= 2:
                r = (s.iloc[-1] / s.iloc[0]) - 1.0
                rets[t] = float(r)
        except Exception:
            continue
    return rets


@st.cache_data(show_spinner=False, ttl=900)
def _latest_price_change(tickers: List[str]) -> Dict[str, float]:
    # Fetch last 2 trading days to approximate daily % change
    end = date.today()
    start = end - timedelta(days=7)
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    chg = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                s = data[(t, "Adj Close")].dropna().tail(2)
            else:
                s = data["Adj Close"].dropna().tail(2)
            if len(s) == 2:
                chg[t] = float((s.iloc[-1] / s.iloc[0]) - 1.0) * 100.0  # percent
        except Exception:
            pass
    return chg

# -----------------------
# Load metadata and prices
# -----------------------
with st.spinner("Fetching company info..."):
    meta = pd.DataFrame([_info_for_ticker(t) for t in tickers])

# Convert market cap; keep rows even if NaN
meta["market_cap"] = pd.to_numeric(meta["market_cap"], errors="coerce")

# Apply min market cap only where mcap is known
if min_mcap > 0:
    known = meta["market_cap"].notna()
    meta = pd.concat([
        meta[known & (meta["market_cap"] >= min_mcap * 1e9)],
        meta[~known]  # keep unknowns to avoid empty df
    ])

# Assign a tiny placeholder size for unknown market caps so they still render
meta["market_cap_filled"] = meta["market_cap"].fillna(1e6)  # $1M placeholder

with st.spinner("Fetching prices & computing returns..."):
    ret_map = _price_return_for_tickers(meta["ticker"].tolist(), start_date, end_date)
    meta["return"] = meta["ticker"].map(ret_map)
    meta = meta.dropna(subset=["return"])

if meta.empty:
    st.warning("No data to display. Try different dates, a smaller universe, or lower min market cap.")
    st.stop()

# -----------------------
# Tabs
# -----------------------
tab_heatmap, tab_sectors, tab_watch, tab_alerts = st.tabs(["🌳 Heatmap", "📚 Sectors", "⭐ Watchlist", "⏰ Alerts"])

with tab_heatmap:
    st.subheader("Treemap Heatmap")
    q = np.nanmax(np.abs(meta["return"].quantile([0.1, 0.9]).values))
    if not np.isfinite(q) or q == 0: q = float(np.nanmax(np.abs(meta["return"].values)) or 0.05)
    cmin, cmax = -q, q

    fig = px.treemap(
        meta,
        path=["sector", "industry", "ticker"],
        values="market_cap_filled",
        color="return",
        color_continuous_scale="RdYlGn",
        range_color=(cmin, cmax),
        hover_data={"name": True, "market_cap": ":,.0f", "return": ":.2%"}
    )
    fig.update_layout(margin=dict(l=4, r=4, t=30, b=4), coloraxis_colorbar=dict(title="Return"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Add to Watchlist")
    sel = st.multiselect("Choose tickers to add", meta["ticker"].tolist(), placeholder="Type ticker(s)")
    if st.button("➕ Add to Watchlist", use_container_width=False) and sel:
        st.session_state.watchlist.update(sel)
        save_store()
        st.success(f"Added {len(sel)} tickers to watchlist.")

    st.markdown("### Data")
    cols = ["ticker","name","sector","industry","market_cap","return"]
    st.dataframe(meta[cols].sort_values("market_cap_filled", ascending=False), use_container_width=True)

with tab_sectors:
    st.subheader("Sector & Industry Breakdown")
    sectors = sorted(meta["sector"].dropna().unique().tolist())
    sector_tabs = st.tabs(sectors if sectors else ["All"])
    for i, sec in enumerate(sectors):
        with sector_tabs[i]:
            df = meta[meta["sector"] == sec]
            industries = df.groupby("industry").agg(
                mcap=("market_cap","sum"),
                avg_ret=("return","mean"),
                count=("ticker","nunique")
            ).reset_index().sort_values("mcap", ascending=False)

            c1, c2 = st.columns([2,3])
            with c1:
                st.metric("Tickers", int(df["ticker"].nunique()))
                st.metric("Total Mkt Cap", f"${df['market_cap'].sum():,.0f}")
                st.metric("Avg Return", f"{df['return'].mean()*100:,.2f}%")
            with c2:
                bar = px.bar(industries, x="industry", y="mcap", hover_data={"avg_ret":":.2%", "count":True}, title=f"{sec} — Market Cap by Industry")
                bar.update_layout(xaxis_title="", yaxis_title="Market Cap (USD)")
                st.plotly_chart(bar, use_container_width=True)

            st.dataframe(industries, use_container_width=True)

with tab_watch:
    st.subheader("Your Watchlist")
    wl = sorted(list(st.session_state.watchlist))
    if not wl:
        st.info("Watchlist is empty. Add tickers from the Heatmap tab.")
    else:
        st.write(f"{len(wl)} tickers in watchlist.")
        wl_df = meta[meta["ticker"].isin(wl)].copy()
        missing = [t for t in wl if t not in wl_df["ticker"].tolist()]
        if missing:
            # Pull minimal info for missing tickers (e.g., filtered out by mcap)
            extra = pd.DataFrame([_info_for_ticker(t) for t in missing])
            wl_df = pd.concat([wl_df, extra], ignore_index=True)
        st.dataframe(wl_df[["ticker","name","sector","industry","market_cap","return"]], use_container_width=True)

        rem = st.multiselect("Remove tickers", wl, placeholder="Select to remove")
        if st.button("🗑️ Remove from Watchlist") and rem:
            st.session_state.watchlist.difference_update(rem)
            save_store()
            st.warning(f"Removed: {', '.join(rem)}")

with tab_alerts:
    st.subheader("Price Change Alerts (Daily %)")
    st.caption("Define a daily % change threshold. When exceeded, alerts appear below.")

    wl = sorted(list(st.session_state.watchlist)) or meta["ticker"].head(10).tolist()
    ticker_sel = st.selectbox("Ticker", wl, index=0)
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.number_input("Threshold (%)", min_value=0.1, max_value=50.0, value=3.0, step=0.1)
    with col2:
        direction = st.selectbox("Direction", ["up","down","both"], index=2)

    if st.button("✅ Save Alert Rule"):
        st.session_state.alerts[ticker_sel] = {"threshold": float(threshold), "direction": direction}
        save_store()
        st.success(f"Rule saved for {ticker_sel}: {direction} {threshold:.1f}%")

    if st.session_state.alerts:
        st.write("### Active Rules")
        rules_df = pd.DataFrame([{"ticker": t, **cfg} for t,cfg in st.session_state.alerts.items()])
        st.dataframe(rules_df, use_container_width=True)
        if st.button("❌ Clear All Rules"):
            st.session_state.alerts = {}
            save_store()

        st.markdown("---")
        st.write("### Check Alerts Now")
        check = st.button("🔁 Refresh & Evaluate")
        if check:
            tickers_to_check = list(st.session_state.alerts.keys())
            changes = _latest_price_change(tickers_to_check)  # % change today vs last close
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            hits = []
            for t, cfg in st.session_state.alerts.items():
                change = changes.get(t)
                if change is None:
                    continue
                cond = (
                    (cfg["direction"] == "up" and change >= cfg["threshold"]) or
                    (cfg["direction"] == "down" and change <= -cfg["threshold"]) or
                    (cfg["direction"] == "both" and (change >= cfg["threshold"] or change <= -cfg["threshold"]))
                )
                if cond:
                    hits.append(f"[{now}] {t} moved {change:.2f}% (rule: {cfg['direction']} {cfg['threshold']:.1f}%)")
            if hits:
                st.session_state.last_alerts = hits + st.session_state.last_alerts
                st.success(f"{len(hits)} alert(s) triggered.")
            else:
                st.info("No alerts triggered.")

        if st.session_state.last_alerts:
            st.markdown("### Recent Alerts")
            for msg in st.session_state.last_alerts[:20]:
                st.write("• " + msg)
    else:
        st.info("No alert rules yet. Add one above.")

