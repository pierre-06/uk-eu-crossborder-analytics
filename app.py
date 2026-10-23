import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="UK-EU Equity Risk & VaR Engine", layout="wide")

st.title("Cross-Border Equity Risk Analytics Engine")
st.markdown("Automated risk pipeline comparing **FTSE 100 (UK)** vs **CAC 40 (France)** normalised in GBP.")

# --- 1 & 2: NEW SIDEBAR INPUTS (Monetary & Asset Allocation) ---
st.sidebar.header("Client Portfolio Parameters")
portfolio_value = st.sidebar.number_input("Initial Portfolio Value (£)", min_value=1000, value=100000, step=1000)

ftse_weight = st.sidebar.slider("FTSE 100 Allocation (%)", 0, 100, 60) / 100.0
cac_weight = 1.0 - ftse_weight
st.sidebar.markdown(f"*CAC 40 Allocation: {cac_weight*100:.0f}%*")

st.sidebar.header("Risk Engine Parameters")
time_horizon = st.sidebar.selectbox("Lookback Period", ["1Y", "3Y", "5Y", "10Y"], index=2)
confidence_level = st.sidebar.slider("VaR Confidence Level (%)", 90, 99, 95) / 100.0
period_map = {"1Y": "1y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}

@st.cache_data
def load_and_process_data(period):
    tickers = ['^FTSE', '^FCHI', 'EURGBP=X']
    raw_data = yf.download(tickers, period=period)['Close']
    
    raw_data.index = pd.to_datetime(raw_data.index, utc=True).tz_localize(None)
    data = raw_data.ffill().dropna().copy()
    data['CAC40_GBP'] = data['^FCHI'] * data['EURGBP=X']
    
    returns = pd.DataFrame()
    returns['FTSE100'] = data['^FTSE'].pct_change()
    returns['CAC40_EUR'] = data['^FCHI'].pct_change()
    returns['CAC40_GBP'] = data['CAC40_GBP'].pct_change()
    
    return data.dropna(), returns.dropna()

data, returns = load_and_process_data(period_map[time_horizon])

# --- NEW: Calculate Blended Portfolio Returns ---
returns['Blended_Portfolio'] = (returns['FTSE100'] * ftse_weight) + (returns['CAC40_GBP'] * cac_weight)

def calculate_var_cvar(returns_series, confidence):
    var = np.percentile(returns_series, (1 - confidence) * 100)
    cvar = returns_series[returns_series <= var].mean()
    return var, cvar

ftse_var, ftse_cvar = calculate_var_cvar(returns['FTSE100'], confidence_level)
cac_var, cac_cvar = calculate_var_cvar(returns['CAC40_GBP'], confidence_level)
blend_var, blend_cvar = calculate_var_cvar(returns['Blended_Portfolio'], confidence_level)

# --- NEW: Display Client-Facing Monetary Risk Metrics ---
st.markdown("---")
st.subheader(f"Portfolio Risk Exposure ({confidence_level*100:.0f}% Confidence)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Blended Portfolio Daily VaR", f"£{(blend_var * portfolio_value * -1):,.2f}", f"{blend_var*100:.2f}%", delta_color="inverse")
with col2:
    st.metric("FTSE 100 Standalone VaR", f"£{(ftse_var * portfolio_value * -1):,.2f}", f"{ftse_var*100:.2f}%", delta_color="inverse")
with col3:
    st.metric("CAC 40 (GBP) Standalone VaR", f"£{(cac_var * portfolio_value * -1):,.2f}", f"{cac_var*100:.2f}%", delta_color="inverse")

st.markdown(f"> **Advisory Insight:** Under normal market conditions, there is a {(1-confidence_level)*100:.0f}% probability that the blended client portfolio will lose more than **£{(blend_var * portfolio_value * -1):,.2f}** in a single trading day.")

# --- UPDATE: Add Blended Data to Charts ---
st.markdown("---")
st.subheader("Normalised Return Performance (Base = 100)")
norm_data = (data[['^FTSE', 'CAC40_GBP']] / data[['^FTSE', 'CAC40_GBP']].iloc[0]) * 100
norm_data['Blended_Portfolio'] = (norm_data['^FTSE'] * ftse_weight) + (norm_data['CAC40_GBP'] * cac_weight)

fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(x=norm_data.index, y=norm_data['^FTSE'], name="FTSE 100", line=dict(dash='dot')))
fig_perf.add_trace(go.Scatter(x=norm_data.index, y=norm_data['CAC40_GBP'], name="CAC 40 (GBP)", line=dict(dash='dot')))
fig_perf.add_trace(go.Scatter(x=norm_data.index, y=norm_data['Blended_Portfolio'], name="Blended Portfolio", line=dict(width=3)))
fig_perf.update_layout(xaxis_title="Date", yaxis_title="Indexed Value", hovermode="x unified")
st.plotly_chart(fig_perf, use_container_width=True)

st.subheader("30-Day Rolling Volatility Comparison")
rolling_vol = returns[['FTSE100', 'CAC40_GBP', 'Blended_Portfolio']].rolling(window=30).std() * np.sqrt(252) * 100
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol['FTSE100'], name="FTSE Volatility"))
fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol['CAC40_GBP'], name="CAC Volatility"))
fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol['Blended_Portfolio'], name="Blended Volatility", line=dict(width=3)))
fig_vol.update_layout(xaxis_title="Date", yaxis_title="Annualised Volatility (%)", hovermode="x unified")
st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("**Engineered by Elohim Pierre** | BSc Accounting & Finance, Aston University")
