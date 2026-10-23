import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="UK-EU Equity Risk & VaR Engine", layout="wide")

st.title("Cross-Border Equity Risk Analytics Engine")
st.markdown("Automated risk pipeline comparing **FTSE 100 (UK)** vs **CAC 40 (France)** normalised in GBP.")

st.sidebar.header("Model Parameters")
time_horizon = st.sidebar.selectbox("Lookback Period", ["1Y", "3Y", "5Y", "10Y"], index=2)
confidence_level = st.sidebar.slider("VaR Confidence Level (%)", 90, 99, 95) / 100.0
period_map = {"1Y": "1y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}

@st.cache_data
def load_and_process_data(period):
    tickers = ['^FTSE', '^FCHI', 'EURGBP=X']
    raw_data = yf.download(tickers, period=period)['Close']
    
    # PATCH: Convert to UTC and strip time zones so London and Paris align perfectly
    raw_data.index = pd.to_datetime(raw_data.index, utc=True).tz_localize(None)
    
    # Forward-fill to cover overlapping bank holidays, then drop remaining empty rows
    data = raw_data.ffill().dropna().copy()
    
    data['CAC40_GBP'] = data['^FCHI'] * data['EURGBP=X']
    
    returns = pd.DataFrame()
    returns['FTSE100'] = data['^FTSE'].pct_change()
    returns['CAC40_EUR'] = data['^FCHI'].pct_change()
    returns['CAC40_GBP'] = data['CAC40_GBP'].pct_change()
    
    return data.dropna(), returns.dropna()
data, returns = load_and_process_data(period_map[time_horizon])

def calculate_var_cvar(returns_series, confidence):
    var = np.percentile(returns_series, (1 - confidence) * 100)
    cvar = returns_series[returns_series <= var].mean()
    return var, cvar

ftse_var, ftse_cvar = calculate_var_cvar(returns['FTSE100'], confidence_level)
cac_var, cac_cvar = calculate_var_cvar(returns['CAC40_GBP'], confidence_level)

col1, col2, col3, col4 = st.columns(4)
col1.metric("FTSE 100 Daily VaR", f"{ftse_var*100:.2f}%")
col2.metric("FTSE 100 Daily CVaR", f"{ftse_cvar*100:.2f}%")
col3.metric("CAC 40 (GBP) Daily VaR", f"{cac_var*100:.2f}%")
col4.metric("CAC 40 (GBP) Daily CVaR", f"{cac_cvar*100:.2f}%")

st.markdown("---")
st.subheader("Normalised Return Performance (Base = 100)")
norm_data = (data[['^FTSE', 'CAC40_GBP']] / data[['^FTSE', 'CAC40_GBP']].iloc[0]) * 100
fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(x=norm_data.index, y=norm_data['^FTSE'], name="FTSE 100 (GBP)"))
fig_perf.add_trace(go.Scatter(x=norm_data.index, y=norm_data['CAC40_GBP'], name="CAC 40 (GBP Adjusted)"))
fig_perf.update_layout(xaxis_title="Date", yaxis_title="Indexed Value", hovermode="x unified")
st.plotly_chart(fig_perf, use_container_width=True)

st.subheader("30-Day Rolling Volatility Comparison")
rolling_vol = returns[['FTSE100', 'CAC40_GBP']].rolling(window=30).std() * np.sqrt(252) * 100
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol['FTSE100'], name="FTSE 100 Volatility (%)"))
fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol['CAC40_GBP'], name="CAC 40 Volatility (%)"))
fig_vol.update_layout(xaxis_title="Date", yaxis_title="Annualised Volatility (%)", hovermode="x unified")
st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("**Engineered by Elohim Pierre** | BSc Accounting & Finance, Aston University")

