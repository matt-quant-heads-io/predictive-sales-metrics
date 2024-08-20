import streamlit as st
import sys

import pandas as pd
import altair as alt
from urllib.error import URLError
import plotly.express as px
import plotly.graph_objects as go
import streamviz

import numpy as np


from helper import get_eod_api_data_by_ticker, process_second_measure_input_data, process_similarweb_data, compute_rquared_and_correlations, get_start_date_from_sm_file, generate_predictors_tab, generate_tracking_page

METRICS = ["Observed Sales","Observed Transactions","Observed Customers","Sales per Customer","Average Transaction Value", "Transactions per Customer"] #["Observed Sales","Observed Transactions","Observed Customers","Sales per Customer","Average Transaction Value","Transactions per Customer","Share of Sales","Share of Transactions"]
SIMILAR_WEB_API_KEY = "7d015188e27e4ee1867f64a18213cbf5"

st.set_page_config(page_title="Backtester", page_icon="📊", layout="wide")
# st.markdown("# Backtest Results")
st.sidebar.markdown("### Backtest Parameters")

df = None
ticker = None
revenue_dict = None
normalized_eod_df = None
final_df = None
company_website = None
metric_to_df_map = {}
final_similar_web_df = None
uploaded_file = None
sw_uploaded_file = None
eod_data_by_ticker = None
coeffs_df = None
similar_web_data_df = None
raw_dfs_map = {}
mse = None
r2 = None
current_preds_df = None
df_ma = None
tracking_df_current_year = None
tracking_df_prior_year = None

try:
    with st.sidebar.form("my_form"):
        uploaded_file = st.file_uploader("Choose a Second Measure file", type=['xlsx', 'csv'])
        sw_uploaded_file = st.file_uploader("Choose a SimilarWeb file", type=['xlsx'])
        submitted = st.form_submit_button("Submit")
            

        if submitted:
            if uploaded_file is not None:
                uploaded_file.seek(0)
                if ticker is None:
                    ticker = uploaded_file.name.split("_")[0]
                start_date = get_start_date_from_sm_file(uploaded_file)
                eod_data_by_ticker, company_website = get_eod_api_data_by_ticker(ticker, start_date)
                for metric in METRICS:
                    uploaded_file.seek(0)
                    df, ticker = process_second_measure_input_data(uploaded_file,eod_data_by_ticker, metrics=[metric], ticker=ticker)
                    raw_dfs_map[metric] = df
                    final_df = df.merge(eod_data_by_ticker, how='inner',  left_on=df.index, right_on=eod_data_by_ticker.index)
                    final_df.rename({"key_0": "Date"}, axis=1, inplace=True)
                    final_df.set_index("Date", inplace=True)
                    metric_to_df_map[metric] = final_df

            
            if sw_uploaded_file is not None:
                sw_uploaded_file.seek(0)
                similar_web_data_df = process_similarweb_data(sw_uploaded_file, eod_data_by_ticker)
                raw_dfs_map["Similar Web Data"] = similar_web_data_df
                final_similar_web_df = similar_web_data_df.merge(eod_data_by_ticker, how='inner',  left_on=similar_web_data_df.index, right_on=eod_data_by_ticker.index)
                
                final_similar_web_df.rename({"key_0": "Date"}, axis=1,inplace=True)
                final_similar_web_df.set_index("Date", inplace=True)
                metric_to_df_map["Similar Web Data"] = final_similar_web_df

            # NOTE: Create Predictors Page
            # coeffs_df, mse, r2, test_pts, train_pts, current_preds_df = generate_predictors_tab(metric_to_df_map, raw_dfs_map)
            sw_uploaded_file.seek(0)
            tracking_df_prior_year, tracking_df_current_year = generate_tracking_page(ticker, sw_uploaded_file, raw_dfs_map["Similar Web Data"][["Visits",  "Unique Visitors",  "Total Page Views"]])


            # NOTE: Today vs. t - 1 yr

                    
    if ticker:
        # TODO: use this to fill out streamlit "Metrics" that contain r2 and correlation
        backtest_df = compute_rquared_and_correlations(metric_to_df_map, eod_data_by_ticker)
        st.markdown(f"# Backtest Results {ticker}")       
        
        parent_tabs = st.tabs(["Correlations", "Predictors", "Current Tracking"])
        
        # Correlations tab
        with parent_tabs[0]:
            sw_and_sm_tabs = st.tabs(["Similar Web", "Second Measure"])

            with sw_and_sm_tabs[0]:
                st.markdown(f"### SW Stats")
                row1 = st.columns(3)
                row2 = st.columns(3)

                idx = 0
                for col in row1 + row2:
                    tile = col.container(height=120)
                    if idx == 0:
                        tile.metric("Visits % Change ${r^2}$", round(backtest_df[(backtest_df["x"] == "Visits % Change")]["R_squared"],4))
                    elif idx == 1:
                        tile.metric("Unique Visitors % Change ${r^2}$", round(backtest_df[(backtest_df["x"] == "Unique Visitors % Change")]["R_squared"],4))
                    elif idx == 2:
                        tile.metric("Total Page Views % Change ${r^2}$", round(backtest_df[(backtest_df["x"] == "Total Page Views % Change")]["R_squared"],4))
                    elif idx == 3:
                        tile.metric("Visits % Change Corr.", round(backtest_df[(backtest_df["x"] == "Visits % Change")]["Correlation"],4))
                    elif idx == 4:
                        tile.metric("Unique Visitors % Change Corr.", round(backtest_df[(backtest_df["x"] == "Unique Visitors % Change")]["Correlation"],4))
                    elif idx == 5:
                        tile.metric("Total Page Views % Change Corr.", round(backtest_df[(backtest_df["x"] == "Total Page Views % Change")]["Correlation"],4))
                    
                    idx += 1          

                df_sw = metric_to_df_map["Similar Web Data"]
                df_sw["Quarters"] = df_sw.index.values 
                trace1 = go.Scatter(x=df_sw['Quarters'], y=df_sw['Visits % Change'], mode='lines+markers', name='Visits % Change', line=dict(color='blue'))
                trace2 = go.Scatter(x=df_sw['Quarters'], y=df_sw['Unique Visitors % Change'], mode='lines+markers', name='Unique Visitors % Change', line=dict(color='yellow'))
                trace3 = go.Scatter(x=df_sw['Quarters'], y=df_sw['Total Page Views % Change'], mode='lines+markers', name='Total Page Views % Change', line=dict(color='orange'))
                trace4 = go.Scatter(x=df_sw['Quarters'], y=df_sw['Rev % Change'], mode='lines+markers+text', name='Rev % Change', line=dict(color='purple'))
                data = [trace1, trace2, trace3, trace4]
                layout = go.Layout(title='SW Predictors vs. Revenue (Q) % Change', xaxis=dict(title='Quarterly Dates'), yaxis=dict(title='Y'), hovermode='closest')
                fig = go.Figure(data=data, layout=layout)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with st.expander("Similar Web Table"):
                    st.dataframe(df_sw)

            with sw_and_sm_tabs[1]:         
                st.markdown(f"### SM Stats")
                row3 = st.columns(6)
                row4 = st.columns(6)

                idx = 0
                for col in row3 + row4:
                    tile2 = col.container(height=120)
                    if idx == 0:
                        tile2.metric("Obs. Sales % Chg ${r^2}$", round(backtest_df[(backtest_df["x"] == "Observed Sales (All % Change)")]["R_squared"],4))
                    elif idx == 1:
                        tile2.metric("Obs. Txns % Chg ${r^2}$", round(backtest_df[(backtest_df["x"] == "Observed Transactions (All % Change)")]["R_squared"],4))
                    elif idx == 2:
                        tile2.metric("Obs. Custs % Chg ${r^2}$", round(backtest_df[(backtest_df["x"] == "Observed Customers (All % Change)")]["R_squared"],4))
                    elif idx == 3:
                        tile2.metric("Sales per Cust. % Chg ${r^2}$", round(backtest_df[(backtest_df["x"] == "Sales per Customer (All % Change)")]["R_squared"],4))
                    elif idx == 4:
                        tile2.metric("Avg Txn Value % Chg ${r^2}$", round(backtest_df[(backtest_df["x"] == "Average Transaction Value (All % Change)")]["R_squared"],4))
                    elif idx == 5:
                        tile2.metric("Txns per Cust. % Chg ${r^2}$", round(backtest_df[(backtest_df["x"] == "Transactions per Customer (All % Change)")]["R_squared"],4))            
                    elif idx == 6:
                        tile2.metric("Obs. Sales % Chg Corr.", round(backtest_df[(backtest_df["x"] == "Observed Sales (All % Change)")]["Correlation"],4))
                    elif idx == 7:
                        tile2.metric("Obs. Txns % Chg Corr.", round(backtest_df[(backtest_df["x"] == "Observed Transactions (All % Change)")]["Correlation"],4))
                    elif idx == 8:
                        tile2.metric("Obs. Custs % Chg Corr.", round(backtest_df[(backtest_df["x"] == "Observed Customers (All % Change)")]["Correlation"],4))
                    elif idx == 9:
                        tile2.metric("Sales per Cust. % Chg Corr.", round(backtest_df[(backtest_df["x"] == "Sales per Customer (All % Change)")]["Correlation"],4))
                    elif idx == 10:
                        tile2.metric("Avg Txn Value % Chg Corr.", round(backtest_df[(backtest_df["x"] == "Average Transaction Value (All % Change)")]["Correlation"],4))
                    elif idx == 11:
                        tile2.metric("Txns per Customer % Chg ${r^2}$", round(backtest_df[(backtest_df["x"] == "Transactions per Customer (All % Change)")]["Correlation"],4))
                    
                    idx += 1

                df_sm = metric_to_df_map["Observed Sales"]
                df_sm["Quarters"] = df_sm.index.values
                trace_a = go.Scatter(x=df_sm['Quarters'], y=metric_to_df_map["Observed Sales"]["All % Change"].values, mode='lines+markers', name='Observed Sales % Change', line=dict(color='blue'))
                trace_b = go.Scatter(x=df_sm['Quarters'], y=metric_to_df_map["Observed Transactions"]["All % Change"], mode='lines+markers', name='Observed Transactions % Change', line=dict(color='teal'))
                trace_c = go.Scatter(x=df_sm['Quarters'], y=metric_to_df_map["Sales per Customer"]["All % Change"], mode='lines+markers', name='Sales per Customer % Change', line=dict(color='pink'))
                trace_d = go.Scatter(x=df_sm['Quarters'], y=metric_to_df_map["Average Transaction Value"]["All % Change"], mode='lines+markers', name='Average Transaction Value % Change', line=dict(color='green'))
                trace_e = go.Scatter(x=df_sm['Quarters'], y=metric_to_df_map["Transactions per Customer"]["All % Change"], mode='lines+markers', name='Transactions per Customer % Change', line=dict(color='cyan'))
                trace_f = go.Scatter(x=df_sm['Quarters'], y=metric_to_df_map["Observed Sales"]["Rev % Change"], mode='lines+markers+text', name='Rev % Change % Change', line=dict(color='purple'))
                data2 = [trace_a, trace_b, trace_c, trace_d, trace_e, trace_f]
                layout2 = go.Layout(title='Second Measure vs. Revenue (Q) % Change', xaxis=dict(title='Quarterly Dates'), yaxis=dict(title='Y'), hovermode='closest')
                fig2 = go.Figure(data=data2, layout=layout2)
                st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

                with st.expander("Second Measure Table"):
                    sm_metrics = st.tabs(METRICS)
                    for i, metric_i in enumerate(METRICS):
                        with sm_metrics[i]:
                            st.header(metric_i)
                            st.dataframe(metric_to_df_map[metric_i], height=800, width=1600)

                # print(f"backtest_df\n{backtest_df}")
        # with parent_tabs[1]:
        #     st.markdown(f"### What is the linear combination of features (from SimilarWeb and Second Measure) that minimizes the MSE of Rev % change predictions over the last {len(metric_to_df_map['Observed Sales'].values)} quarters?")
        #     st.markdown(f"Attribution coefficients")
        #     st.dataframe(coeffs_df)
        #     error_cols = st.columns(2)
        #     with error_cols[0]:
        #         tile3 = st.container(height=120)
        #         tile3.metric("MSE", round(mse,4))
        #     with error_cols[1]:
        #         tile4 = st.container(height=120)
        #         tile4.metric("${r^2}$", round(r2,4))

            
        #     trace_z = go.Scatter(x=current_preds_df.index, y=current_preds_df["Rev % Pred"], mode='lines+markers', name='Rev % Chg Preds', line=dict(color='blue'))
        #     reversed_preds_df = current_preds_df.sort_index(ascending=True)
        #     trace_ma = go.Scatter(x=reversed_preds_df.index, y=reversed_preds_df["Rev % Pred"].rolling(window=5, min_periods=5).mean(), mode='lines+markers', name='SMA', line=dict(color='green'))
        #     data3 = [trace_z, trace_ma]
        #     layout3 = go.Layout(title='Attribution Cooeficients Applied to Daily inputs to Back out % Rev Chg Preds for Next Quarter', xaxis=dict(title='Dates'), yaxis=dict(title='Y'), hovermode='closest')
        #     fig3 = go.Figure(data=data3, layout=layout3)
            
            
        #     st.plotly_chart(fig3, theme="streamlit", use_container_width=True)

        #     with st.expander("Rev % Chg Predictions"):
        #         st.dataframe(current_preds_df, height=800, width=1600)

        # with parent_tabs[2]:
        #     st.markdown(f"### Where are predictors tracking versus today 1 year ago?")
        #     trace_visits = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year["Visits_CumSum"], mode='lines+markers', name=f'Visits {tracking_df_current_year.index.year[0]}', line=dict(color='blue'))
        #     trace_unique = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year["Unique_CumSum"], mode='lines+markers', name=f'Unique {tracking_df_current_year.index.year[0]}', line=dict(color='yellow'))
        #     trace_views = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year["Views_CumSum"], mode='lines+markers', name=f'Views {tracking_df_current_year.index.year[0]}', line=dict(color='red'))

        #     prior_trace_visits = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_prior_year["Visits_CumSum"], mode='lines+markers', name=f'Visits {tracking_df_current_year.index.year[0]-1}', line=dict(color='purple'))
        #     prior_trace_unique = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_prior_year["Unique_CumSum"], mode='lines+markers', name=f'Unique {tracking_df_current_year.index.year[0]-1}', line=dict(color='green'))
        #     prior_trace_views = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_prior_year["Views_CumSum"], mode='lines+markers', name=f'Views {tracking_df_current_year.index.year[0]-1}', line=dict(color='teal'))


        #     data4 = [trace_visits, trace_unique, trace_views, prior_trace_visits, prior_trace_unique, prior_trace_views]
        #     layout4 = go.Layout(title='Similar Predictors Tracking vs. Last Year', xaxis=dict(title='Dates'), yaxis=dict(title='Y'), hovermode='closest')
        #     fig4 = go.Figure(data=data4, layout=layout4)
            
            
        #     st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

        #     with st.expander(f"Similar Web {tracking_df_current_year.index.year[0]}"):
        #         st.dataframe(tracking_df_current_year, height=800, width=1600)

        #     with st.expander(f"Similar Web {tracking_df_current_year.index.year[0]-1}"):
        #         st.dataframe(tracking_df_prior_year, height=800, width=1600)

        with parent_tabs[2]:
            st.markdown(f"### Where are predictors tracking versus today 1 year ago?")
            # trace_visits = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year.visits, mode='lines+markers', name=f'Visits {tracking_df_current_year.index.year[0]}', line=dict(color='blue'))
            # trace_unique = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year.uniques, mode='lines+markers', name=f'Unique {tracking_df_current_year.index.year[0]}', line=dict(color='yellow'))
            # trace_views = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year.views, mode='lines+markers', name=f'Views {tracking_df_current_year.index.year[0]}', line=dict(color='red'))
            trace_visits = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year.visits, mode='lines+markers', name=f'Visits', line=dict(color='blue'))
            trace_unique = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year.uniques, mode='lines+markers', name=f'Unique', line=dict(color='yellow'))
            trace_views = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_current_year.views, mode='lines+markers', name=f'Views', line=dict(color='red'))

            # prior_trace_visits = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_prior_year.visits, mode='lines+markers', name=f'Visits {tracking_df_current_year.index.year[0]-1}', line=dict(color='purple'))
            # prior_trace_unique = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_prior_year.uniques, mode='lines+markers', name=f'Unique {tracking_df_current_year.index.year[0]-1}', line=dict(color='green'))
            # prior_trace_views = go.Scatter(x=tracking_df_current_year.index, y=tracking_df_prior_year.views, mode='lines+markers', name=f'Views {tracking_df_current_year.index.year[0]-1}', line=dict(color='teal'))


            # data4 = [trace_visits, trace_unique, trace_views, prior_trace_visits, prior_trace_unique, prior_trace_views]
            data4 = [trace_visits, trace_unique, trace_views]
            layout4 = go.Layout(title='Similar Predictors Tracking vs. Last Year', xaxis=dict(title='Dates'), yaxis=dict(title='Y'), hovermode='closest')
            fig4 = go.Figure(data=data4, layout=layout4)
            
            
            st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

            with st.expander(f"Similar Web {tracking_df_current_year.index.year[0]}"):
                st.dataframe(tracking_df_current_year, height=800, width=1600)

            with st.expander(f"Similar Web {tracking_df_current_year.index.year[0]-1}"):
                st.dataframe(tracking_df_prior_year, height=800, width=1600)
            
            

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )