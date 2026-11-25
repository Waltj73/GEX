import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="GEX Butterfly Scanner", layout="wide")

# --- Black-Scholes Gamma Calculation ---
def calculate_bs_gamma(S, K, T, r, sigma):
    """Computes Black-Scholes Gamma for a given strike."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=300)
def fetch_ticker_data(ticker_symbol):
    """Fetches spot price and available option expiration dates."""
    tk = yf.Ticker(ticker_symbol)
    hist = tk.history(period="5d")
    if hist.empty:
        return None, []
    spot = float(hist["Close"].iloc[-1])
    expirations = list(tk.options)
    return spot, expirations

@st.cache_data(ttl=300)
def compute_gex_profile(ticker_symbol, spot, selected_expirations, risk_free_rate=0.045):
    """Calculates strike-by-strike Call GEX, Put GEX, and Net GEX ($ Millions)."""
    tk = yf.Ticker(ticker_symbol)
    now = datetime.now()
    records = []

    for exp_str in selected_expirations:
        try:
            chain = tk.option_chain(exp_str)
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            # Calculate time to expiration in years (minimum 0.5 trading days to avoid div-by-zero on 0DTE)
            days_to_exp = max((exp_dt - now).days + 0.5, 0.5)
            T = days_to_exp / 365.0

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            # Compute Gamma for Calls
            for _, row in calls.iterrows():
                oi = row.get("openInterest", 0)
                iv = row.get("impliedVolatility", 0)
                strike = row.get("strike", 0)
                if pd.notna(oi) and oi > 0 and pd.notna(iv) and iv > 0:
                    gamma = calculate_bs_gamma(spot, strike, T, risk_free_rate, iv)
                    # Standard GEX: Calls are positive dealer gamma (+100 shares * Spot^2 * 0.01)
                    call_gex = gamma * oi * 100 * (spot ** 2) * 0.01
                    records.append({"strike": strike, "call_gex": call_gex, "put_gex": 0.0})

            # Compute Gamma for Puts
            for _, row in puts.iterrows():
                oi = row.get("openInterest", 0)
                iv = row.get("impliedVolatility", 0)
                strike = row.get("strike", 0)
                if pd.notna(oi) and oi > 0 and pd.notna(iv) and iv > 0:
                    gamma = calculate_bs_gamma(spot, strike, T, risk_free_rate, iv)
                    # Standard GEX: Puts are negative dealer gamma (-100 shares * Spot^2 * 0.01)
                    put_gex = -gamma * oi * 100 * (spot ** 2) * 0.01
                    records.append({"strike": strike, "call_gex": 0.0, "put_gex": put_gex})

        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    grouped = df.groupby("strike", as_index=False).sum()
    grouped["net_gex"] = grouped["call_gex"] + grouped["put_gex"]
    # Convert to Millions for clean readability
    grouped["call_gex_m"] = grouped["call_gex"] / 1e6
    grouped["put_gex_m"] = grouped["put_gex"] / 1e6
    grouped["net_gex_m"] = grouped["net_gex"] / 1e6
    return grouped.sort_values("strike").reset_index(drop=True)

def find_zero_gamma(df, spot):
    """Finds the Zero Gamma flip strike closest to current spot price."""
    if df.empty or len(df) < 2:
        return 0.0
    # Filter to strikes within +/- 15% of spot to ignore deep OTM noise
    near = df[(df["strike"] >= spot * 0.85) & (df["strike"] <= spot * 1.15)].copy()
    if len(near) < 2:
        near = df.copy()

    # Compute cumulative GEX curve from lowest strike up to find the sign flip
    near["cum_gex"] = near["net_gex"].cumsum()
    flips = []
    for i in range(1, len(near)):
        prev_val = near.iloc[i - 1]["cum_gex"]
        curr_val = near.iloc[i]["cum_gex"]
        if (prev_val < 0 and curr_val >= 0) or (prev_val > 0 and curr_val <= 0):
            s1 = near.iloc[i - 1]["strike"]
            s2 = near.iloc[i]["strike"]
            # Linear interpolation between the two bracketing strikes
            weight = abs(prev_val) / (abs(prev_val) + abs(curr_val))
            flip_strike = s1 + weight * (s2 - s1)
            flips.append(flip_strike)

    if flips:
        # Return flip closest to spot
        return round(min(flips, key=lambda x: abs(x - spot)), 2)

    # Fallback: strike with minimum absolute net GEX near spot
    idx_min = near["net_gex"].abs().idxmin()
    return round(float(near.loc[idx_min, "strike"]), 2)

# --- Streamlit UI ---
st.title("GEX Butterfly & ThinkScript Level Scanner")

col_in1, col_in2, col_in3 = st.columns([1, 1.5, 1])
with col_in1:
    ticker = st.text_input("Ticker Symbol", value="SPY").upper().strip()
with col_in3:
    range_pct = st.slider("Strike Range (+/- % from Spot)", min_value=3, max_value=20, value=8)

spot, expirations = fetch_ticker_data(ticker)

if not spot or not expirations:
    st.error(f"Could not retrieve option chain data for '{ticker}'. Check the ticker symbol and try again.")
    st.stop()

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

with st.spinner(f"Calculating GEX profile for {ticker}..."):
    gex_df = compute_gex_profile(ticker, spot, target_exps)

if gex_df.empty:
    st.warning("No valid open interest / implied volatility data returned for the selected expiration(s).")
    st.stop()

# Filter display & wall detection around spot
lower_bound = spot * (1 - range_pct / 100.0)
upper_bound = spot * (1 + range_pct / 100.0)
window_df = gex_df[(gex_df["strike"] >= lower_bound) & (gex_df["strike"] <= upper_bound)].copy()

if window_df.empty:
    window_df = gex_df.copy()

# Key Level Extraction
zero_gamma = find_zero_gamma(gex_df, spot)
net_gex_billions = round(gex_df["net_gex"].sum() / 1e9, 2)

# Top 2 Call Walls (highest positive Call GEX)
top_calls = window_df.sort_values("call_gex", ascending=False)
major_call_wall = float(top_calls.iloc[0]["strike"]) if len(top_calls) > 0 else 0.0
sec_call_wall = float(top_calls.iloc[1]["strike"]) if len(top_calls) > 1 else 0.0

# Top 2 Put Walls (most negative Put GEX)
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
