import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="GEX Butterfly Scanner", layout="wide")

# --- Fallback Black-Scholes Gamma (if Cboe gamma field is 0/missing) ---
def calculate_bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

@st.cache_data(ttl=300)
def fetch_cboe_chain(ticker_symbol):
    """Fetches full option chain + spot price from Cboe's free delayed JSON endpoint."""
    clean_ticker = ticker_symbol.upper().strip()
    # Cboe prefixes index tickers like SPX, NDX, RUT with an underscore
    index_tickers = {"SPX", "NDX", "RUT", "VIX", "XSP"}
    cboe_symbol = f"_{clean_ticker}" if clean_ticker in index_tickers else clean_ticker

    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{cboe_symbol}.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        return None, pd.DataFrame(), f"HTTP {resp.status_code}"

    payload = resp.json()
    data = payload.get("data", {})
    spot = data.get("current_price") or data.get("close") or data.get("last_trade_price")
    options = data.get("options", [])

    if not spot or not options:
        return None, pd.DataFrame(), "Empty option chain returned by Cboe"

    spot = float(spot)
    now = datetime.now()
    parsed = []

    for opt in options:
        # Cboe option symbol format: e.g., SPY261218C00600000
        sym = opt.get("option", "")
        if len(sym) < 15:
            continue

        # Extract YYMMDD, C/P, and Strike from right side of OCC symbol
        strike_raw = sym[-8:]
        opt_type = sym[-9]
        exp_raw = sym[-15:-9]

        try:
            strike = int(strike_raw) / 1000.0
            exp_dt = datetime.strptime(exp_raw, "%y%m%d")
            exp_str = exp_dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

        oi = float(opt.get("open_interest") or 0)
        iv = float(opt.get("iv") or 0)
        gamma = float(opt.get("gamma") or 0)

        if oi <= 0:
            continue

        if gamma <= 0 and iv > 0:
            days_to_exp = max((exp_dt - now).days + 0.5, 0.5)
            gamma = calculate_bs_gamma(spot, strike, days_to_exp / 365.0, 0.045, iv)

        if gamma <= 0:
            continue

        gex_val = gamma * oi * 100 * (spot ** 2) * 0.01
        if opt_type == "C":
            parsed.append({"expiration": exp_str, "strike": strike, "call_gex": gex_val, "put_gex": 0.0})
        elif opt_type == "P":
            parsed.append({"expiration": exp_str, "strike": strike, "call_gex": 0.0, "put_gex": -gex_val})

    if not parsed:
        return spot, pd.DataFrame(), "No active open interest contracts found"

    df = pd.DataFrame(parsed)
    return spot, df, None

def find_zero_gamma(df, spot):
    """Finds the Zero Gamma flip strike closest to current spot price."""
    if df.empty or len(df) < 2:
        return 0.0
    near = df[(df["strike"] >= spot * 0.85) & (df["strike"] <= spot * 1.15)].copy()
    if len(near) < 2:
        near = df.copy()

    near["cum_gex"] = near["net_gex"].cumsum()
    flips = []
    for i in range(1, len(near)):
        prev_val = near.iloc[i - 1]["cum_gex"]
        curr_val = near.iloc[i]["cum_gex"]
        if (prev_val < 0 and curr_val >= 0) or (prev_val > 0 and curr_val <= 0):
            s1 = near.iloc[i - 1]["strike"]
            s2 = near.iloc[i]["strike"]
            weight = abs(prev_val) / (abs(prev_val) + abs(curr_val))
            flips.append(s1 + weight * (s2 - s1))

    if flips:
        return round(min(flips, key=lambda x: abs(x - spot)), 2)

    idx_min = near["net_gex"].abs().idxmin()
    return round(float(near.loc[idx_min, "strike"]), 2)

# --- Streamlit UI ---
st.title("GEX Butterfly & ThinkScript Level Scanner")

col_in1, col_in2, col_in3 = st.columns([1, 1.5, 1])
with col_in1:
    ticker = st.text_input("Ticker Symbol (e.g. SPY, QQQ, SPX, AAPL)", value="SPY").upper().strip()
with col_in3:
    range_pct = st.slider("Strike Range (+/- % from Spot)", min_value=3, max_value=20, value=8)

with st.spinner(f"Fetching Cboe option chain for {ticker}..."):
    spot, raw_df, err_msg = fetch_cboe_chain(ticker)

if err_msg or raw_df.empty:
    st.error(f"Could not retrieve option chain data for '{ticker}' ({err_msg}). Check the ticker symbol and try again.")
    st.stop()

expirations = sorted(raw_df["expiration"].unique().tolist())

with col_in2:
    mode = st.radio(
        "Expiration Scope",
        ["Single Expiration (Butterfly DTE)", "Next 4 Expirations (Near-Term)", "All Available Expirations"],
        horizontal=True
    )

if mode == "Single Expiration (Butterfly DTE)":
    chosen_exp = st.selectbox("Select Target Butterfly Expiration", expirations, index=0)
    target_exps = [chosen_exp]
    exp_label = chosen_exp
elif mode == "Next 4 Expirations (Near-Term)":
    target_exps = expirations[:4]
    exp_label = f"NEXT 4 ({expirations[0]} to {expirations[min(3, len(expirations)-1)]})"
else:
    target_exps = expirations
    exp_label = "ALL EXPIRATIONS"

filtered_df = raw_df[raw_df["expiration"].isin(target_exps)]
gex_df = filtered_df.groupby("strike", as_index=False)[["call_gex", "put_gex"]].sum()
gex_df["net_gex"] = gex_df["call_gex"] + gex_df["put_gex"]
gex_df["call_gex_m"] = gex_df["call_gex"] / 1e6
gex_df["put_gex_m"] = gex_df["put_gex"] / 1e6
gex_df = gex_df.sort_values("strike").reset_index(drop=True)

# Filter display & wall detection around spot
lower_bound = spot * (1 - range_pct / 100.0)
upper_bound = spot * (1 + range_pct / 100.0)
window_df = gex_df[(gex_df["strike"] >= lower_bound) & (gex_df["strike"] <= upper_bound)].copy()

if window_df.empty:
    window_df = gex_df.copy()

# Key Level Extraction
zero_gamma = find_zero_gamma(gex_df, spot)
net_gex_billions = round(gex_df["net_gex"].sum() / 1e9, 2)

top_calls = window_df.sort_values("call_gex", ascending=False)
major_call_wall = float(top_calls.iloc[0]["strike"]) if len(top_calls) > 0 else 0.0
sec_call_wall = float(top_calls.iloc[1]["strike"]) if len(top_calls) > 1 else 0.0

top_puts = window_df.sort_values("put_gex", ascending=True)
major_put_wall = float(top_puts.iloc[0]["strike"]) if len(top_puts) > 0 else 0.0
sec_put_wall = float(top_puts.iloc[1]["strike"]) if len(top_puts) > 1 else 0.0

# --- Top Metrics Banner ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Spot Price", f"${spot:.2f}")
m2.metric("Zero Gamma Flip", f"${zero_gamma:.2f}", f"{spot - zero_gamma:+.2f} vs Spot")
m3.metric("Major Call Wall", f"${major_call_wall:.2f}", f"{major_call_wall - spot:+.2f}")
m4.metric("Major Put Wall", f"${major_put_wall:.2f}", f"{major_put_wall - spot:+.2f}")
m5.metric("Total Net GEX", f"${net_gex_billions:+.2f}B", "Positive (Pin)" if spot >= zero_gamma else "Negative (Trend)")

# --- Paste-Ready ThinkScript Inputs Table ---
st.subheader("Paste Into ThinkScript (`GEX_Butterfly_Levels`)")
tos_inputs = pd.DataFrame([
    {"ThinkScript Input": "expirationLabel", "Value to Enter": exp_label},
    {"ThinkScript Input": "zeroGammaLevel", "Value to Enter": f"{zero_gamma:.2f}"},
    {"ThinkScript Input": "majorCallWall", "Value to Enter": f"{major_call_wall:.2f}"},
    {"ThinkScript Input": "majorPutWall", "Value to Enter": f"{major_put_wall:.2f}"},
    {"ThinkScript Input": "secondaryCallTarget", "Value to Enter": f"{sec_call_wall:.2f}"},
    {"ThinkScript Input": "secondaryPutTarget", "Value to Enter": f"{sec_put_wall:.2f}"},
    {"ThinkScript Input": "netGexBillions", "Value to Enter": f"{net_gex_billions:.2f}"},
])
st.dataframe(tos_inputs, use_container_width=True, hide_index=True)

# --- Interactive Strike-by-Strike GEX Chart ---
st.subheader(f"{ticker} Gamma Exposure by Strike ({exp_label})")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=window_df["strike"],
    y=window_df["call_gex_m"],
    name="Call GEX ($M)",
    marker_color="#00c853"
))
fig.add_trace(go.Bar(
    x=window_df["strike"],
    y=window_df["put_gex_m"],
    name="Put GEX ($M)",
    marker_color="#ff1744"
))

fig.add_vline(x=spot, line_width=2, line_dash="solid", line_color="white", annotation_text=f"Spot {spot:.2f}")
if zero_gamma > 0:
    fig.add_vline(x=zero_gamma, line_width=2, line_dash="dash", line_color="yellow", annotation_text=f"Zero Gamma {zero_gamma:.2f}")

fig.update_layout(
    barmode="relative",
    template="plotly_dark",
    xaxis_title="Strike Price",
    yaxis_title="Gamma Exposure ($ Millions per 1% Move)",
    height=480,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)
