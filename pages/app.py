import os
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

st.set_page_config(page_title="Advanced Real Estate Intelligence", layout="wide", page_icon="🏢")

@st.cache_data
def load_historical_data():
    db_path = "/app/src/database/real_estate.db"
    if not os.path.exists(db_path):
        return pd.DataFrame()
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql("SELECT month, sector, amount_new_house_transactions FROM new_house_transactions", engine)
    df['date'] = pd.to_datetime(df['month'], format='%Y-%b', errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df['month_num'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

@st.cache_data
def load_predictions():
    pred_path = "/app/data/forecast_68_79.csv"
    if not os.path.exists(pred_path):
        return pd.DataFrame()
    preds = pd.read_csv(pred_path)
    preds['date'] = pd.to_datetime(preds['date'])
    preds['month_num'] = preds['date'].dt.month
    preds['year'] = preds['date'].dt.year
    return preds

df_hist = load_historical_data()
df_pred = load_predictions()
sectors = sorted(df_hist['sector'].unique(), key=lambda x: int(x.split()[-1]) if str(x).split()[-1].isdigit() else 0) if not df_hist.empty else []

st.sidebar.title("🏢 Real Estate Hub")
page = st.sidebar.radio("Navigation Menu", [
    "1. 🌍 Executive Dashboard", 
    "2. 📈 Sector Deep Dive", 
    "3. 🔀 Sector Comparison",
    "4. 🎛️ Scenario Simulator",
    "5. ☀️ Seasonality Analysis",
    "6. 📉 Volatility & Risk",
    "7. 🏆 Growth Leaderboard",
    "8. 📊 Distribution Analysis",
    "9. 🗃️ Data Explorer",
    "10. ⚙️ Model Control Center"
])

st.sidebar.markdown("---")
st.sidebar.caption("Data synced with SQLite Database and Ridge-Logistic Forecast Model.")

if df_hist.empty:
    st.error("Database not found or empty. Please check your SQLite setup.")
    st.stop()

if page == "1. 🌍 Executive Dashboard":
    st.title("🌍 Executive Macro Dashboard")
    st.markdown("City-wide aggregated trends and high-level performance metrics.")
    
    city_hist = df_hist.groupby('date')['amount_new_house_transactions'].sum().reset_index()
    
    st.subheader("Aggregate Market Trend")
    lookback = st.slider("Historical Lookback (Months)", min_value=12, max_value=len(city_hist), value=len(city_hist), step=6)
    
    fig_macro = go.Figure()
    fig_macro.add_trace(go.Scatter(x=city_hist['date'].tail(lookback), y=city_hist['amount_new_house_transactions'].tail(lookback), 
                                   mode='lines', name='City Historical', fill='tozeroy', line=dict(color='teal')))
    
    if not df_pred.empty:
        city_pred = df_pred.groupby('date')['prediction'].sum().reset_index()
        fig_macro.add_trace(go.Scatter(x=city_pred['date'], y=city_pred['prediction'], 
                                       mode='lines', name='City Forecast', line=dict(color='orange', dash='dash')))
    
    fig_macro.update_layout(xaxis_title="Date", yaxis_title="Total Transaction Amount", hovermode="x unified")
    st.plotly_chart(fig_macro, use_container_width=True)

elif page == "2. 📈 Sector Deep Dive":
    st.title("📈 Sector Deep Dive")
    
    selected_sector = st.selectbox("Select Sector", sectors)
    hist_filtered = df_hist[df_hist['sector'] == selected_sector]
    pred_filtered = df_pred[df_pred['sector'] == selected_sector] if not df_pred.empty else pd.DataFrame()
    
    smooth_window = st.slider("Apply Rolling Average Smoothing (Months)", 1, 12, 1)
    
    hist_filtered['smoothed'] = hist_filtered['amount_new_house_transactions'].rolling(window=smooth_window, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_filtered['date'], y=hist_filtered['smoothed'], mode='lines+markers', name='Historical', line=dict(color='royalblue')))
    
    if not pred_filtered.empty:
        fig.add_trace(go.Scatter(x=pred_filtered['date'], y=pred_filtered['prediction'], mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
        
    fig.update_layout(title=f"Transaction Volume: {selected_sector.title()}", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

elif page == "3. 🔀 Sector Comparison":
    st.title("🔀 Sector Comparison")
    st.markdown("Overlay multiple sectors to compare trends.")
    
    selected_sectors = st.multiselect("Select up to 5 Sectors", sectors, default=sectors[:2] if len(sectors)>1 else sectors)
    
    if selected_sectors:
        fig_comp = go.Figure()
        for sec in selected_sectors[:5]:
            sec_data = df_hist[df_hist['sector'] == sec]
            fig_comp.add_trace(go.Scatter(x=sec_data['date'], y=sec_data['amount_new_house_transactions'], mode='lines', name=sec))
            
        fig_comp.update_layout(title="Historical Comparison", hovermode="x unified")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Select at least one sector.")

elif page == "4. 🎛️ Scenario Simulator":
    st.title("🎛️ Scenario Simulator")
    st.markdown("Apply macro-economic shocks to the baseline forecast to simulate best/worst case scenarios.")
    
    if df_pred.empty:
        st.warning("Generate forecasts first in the Control Center.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            shock_pct = st.slider("Global Demand Shock (%)", min_value=-50, max_value=100, value=0, step=5)
        with col2:
            decay_rate = st.slider("Month-over-Month Decay/Growth (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.5)
            
        sim_pred = df_pred.copy()
        sim_pred['month_index'] = sim_pred.groupby('sector').cumcount()
        sim_pred['simulated_prediction'] = sim_pred['prediction'] * (1 + shock_pct/100) * ((1 + decay_rate/100) ** sim_pred['month_index'])
        
        sim_city = sim_pred.groupby('date')[['prediction', 'simulated_prediction']].sum().reset_index()
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=sim_city['date'], y=sim_city['prediction'], name='Baseline Forecast', line=dict(color='gray', dash='dash')))
        fig_sim.add_trace(go.Scatter(x=sim_city['date'], y=sim_city['simulated_prediction'], name='Simulated Scenario', line=dict(color='purple')))
        fig_sim.update_layout(title=f"City-Wide Forecast: {shock_pct}% Shock & {decay_rate}% MoM Growth", hovermode="x unified")
        st.plotly_chart(fig_sim, use_container_width=True)

elif page == "5. ☀️ Seasonality Analysis":
    st.title("☀️ Seasonality Analysis")
    st.markdown("Discover which months drive the most volume.")
    
    selected_sector = st.selectbox("Select Sector for Seasonality", ["All City"] + sectors)
    
    if selected_sector == "All City":
        sea_data = df_hist.groupby('month_num')['amount_new_house_transactions'].mean().reset_index()
    else:
        sea_data = df_hist[df_hist['sector'] == selected_sector].groupby('month_num')['amount_new_house_transactions'].mean().reset_index()
        
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sea_data['Month Name'] = [month_names[i-1] for i in sea_data['month_num']]
    
    fig_sea = px.bar(sea_data, x='Month Name', y='amount_new_house_transactions', 
                     title=f"Average Transactions by Month ({selected_sector})",
                     color='amount_new_house_transactions', color_continuous_scale='Blues')
    st.plotly_chart(fig_sea, use_container_width=True)

elif page == "6. 📉 Volatility & Risk":
    st.title("📉 Volatility & Risk Analysis")
    
    roll_window = st.slider("Rolling Volatility Window (Months)", 3, 24, 6)
    selected_sector = st.selectbox("Select Sector to Analyze", sectors)
    
    vol_data = df_hist[df_hist['sector'] == selected_sector].copy()
    vol_data['rolling_std'] = vol_data['amount_new_house_transactions'].rolling(roll_window).std()
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=vol_data['date'], y=vol_data['rolling_std'], fill='tozeroy', line=dict(color='crimson')))
    fig_vol.update_layout(title=f"{roll_window}-Month Rolling Standard Deviation (Risk)", yaxis_title="Volatility (Std Dev)")
    st.plotly_chart(fig_vol, use_container_width=True)

elif page == "7. 🏆 Growth Leaderboard":
    st.title("🏆 Projected Growth Leaderboard")
    
    if df_pred.empty:
        st.warning("Generate forecasts to view the leaderboard.")
    else:
        last_12 = df_hist[df_hist['date'] >= df_hist['date'].max() - pd.DateOffset(months=12)]
        hist_sum = last_12.groupby('sector')['amount_new_house_transactions'].sum().rename("Last_12M_Actuals")
        pred_sum = df_pred.groupby('sector')['prediction'].sum().rename("Next_12M_Forecast")
        
        board = pd.concat([hist_sum, pred_sum], axis=1).fillna(0)
        board['Growth_Pct'] = ((board['Next_12M_Forecast'] - board['Last_12M_Actuals']) / (board['Last_12M_Actuals'] + 1e-5)) * 100
        
        st.subheader("Top 10 Sectors by Projected Growth (%)")
        top_10 = board.nlargest(10, 'Growth_Pct').reset_index()
        fig_board = px.bar(top_10, x='sector', y='Growth_Pct', text='Growth_Pct', color='Growth_Pct', color_continuous_scale='Greens')
        fig_board.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_board, use_container_width=True)

elif page == "8. 📊 Distribution Analysis":
    st.title("📊 Transaction Distribution")
    st.markdown("Analyze the spread and outliers of transaction amounts.")
    
    selected_sector = st.selectbox("Select Sector", ["All City"] + sectors, key="dist_sec")
    bins = st.slider("Number of Histogram Bins", 10, 100, 30)
    
    plot_data = df_hist if selected_sector == "All City" else df_hist[df_hist['sector'] == selected_sector]
    
    fig_hist = px.histogram(plot_data, x="amount_new_house_transactions", nbins=bins, 
                            title=f"Distribution of Transaction Amounts ({selected_sector})",
                            marginal="box", color_discrete_sequence=['indigo'])
    st.plotly_chart(fig_hist, use_container_width=True)

elif page == "9. 🗃️ Data Explorer":
    st.title("🗃️ Raw Data Explorer")
    st.markdown("Filter and export the raw database.")
    
    min_val, max_val = float(df_hist['amount_new_house_transactions'].min()), float(df_hist['amount_new_house_transactions'].max())
    range_filter = st.slider("Filter by Transaction Amount", min_value=min_val, max_value=max_val, value=(min_val, max_val))
    
    filtered_df = df_hist[(df_hist['amount_new_house_transactions'] >= range_filter[0]) & 
                          (df_hist['amount_new_house_transactions'] <= range_filter[1])]
    
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button("Download Filtered Data (CSV)", filtered_df.to_csv(index=False).encode('utf-8'), "filtered_data.csv", "text/csv")

elif page == "10. ⚙️ Model Control Center":
    st.title("⚙️ Model Control Center")
    st.markdown("Trigger training and inference pipelines directly from the UI.")
    
    st.warning("Running this will execute `/app/models/train_model.py`, which retrains the Multi-Horizon Ridge-Logistic models and overwrites the forecast CSV.")
    
    if st.button("🚀 Run Training & Forecasting Pipeline", type="primary"):
        with st.spinner("Executing pipeline... This will take a moment."):
            try:
                result = subprocess.run(["python", "/app/models/train_model.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Pipeline executed successfully!")
                    st.cache_data.clear()
                    st.code(result.stdout, language='text')
                else:
                    st.error("Pipeline failed.")
                    st.code(result.stderr, language='text')
            except Exception as e:
                st.error(f"Error executing script: {e}")