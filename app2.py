import streamlit as st
import sys

import pandas as pd
import altair as alt
from urllib.error import URLError
import plotly.express as px

import numpy as np

from helper import get_eod_api_data_by_ticker, process_second_measure_input_data, process_similarweb_data, compute_rquared_and_correlations, get_start_date_from_sm_file

METRICS = ["Observed Sales","Observed Transactions","Observed Customers","Sales per Customer","Average Transaction Value","Transactions per Customer"] #["Observed Sales","Observed Transactions","Observed Customers","Sales per Customer","Average Transaction Value","Transactions per Customer","Share of Sales","Share of Transactions"]
SIMILAR_WEB_API_KEY = "7d015188e27e4ee1867f64a18213cbf5"

st.set_page_config(page_title="Backtester", page_icon="📊", layout="wide")
st.markdown("# Backtest Results")
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

try:
    with st.sidebar.form("my_form"):
        uploaded_file = st.file_uploader("Choose a Second Measure file", type=['xlsx', 'csv'])
        sw_uploaded_file = st.file_uploader("Choose a SimilarWeb file", type=['xlsx'])
        submitted = st.form_submit_button("Submit")
            

        if submitted:
            # uploaded_file.seek(0)
            # with open(f"/home/ubuntu/predictive-sales-metrics/input_data/{uploaded_file.name}", "wb") as f:
            #     f.write(uploaded_file.read())

            # sw_uploaded_file.seek(0)
            # with open(f"/home/ubuntu/predictive-sales-metrics/input_data/{sw_uploaded_file.name}", "wb") as f:
            #     f.write(sw_uploaded_file.read())

            # sys.exit(0)
            
            if uploaded_file is not None:
                uploaded_file.seek(0)
                if ticker is None:
                    ticker = uploaded_file.name.split("_")[0]
                start_date = get_start_date_from_sm_file(uploaded_file)
                eod_data_by_ticker, company_website = get_eod_api_data_by_ticker(ticker, start_date)
                for metric in METRICS:
                    uploaded_file.seek(0)
                    df, ticker = process_second_measure_input_data(uploaded_file,eod_data_by_ticker, metrics=[metric], ticker=ticker)
                    final_df = df.merge(eod_data_by_ticker, how='inner',  left_on=df.index, right_on=eod_data_by_ticker.index)
                    final_df.rename({"key_0": "Date"}, axis=1, inplace=True)
                    final_df.set_index("Date", inplace=True)
                    metric_to_df_map[metric] = final_df

            
            if sw_uploaded_file is not None:
                sw_uploaded_file.seek(0)
                similar_web_data_df = process_similarweb_data(sw_uploaded_file, eod_data_by_ticker)
                similar_web_data_df.set_index("Quarterly Date", inplace=True)
                final_similar_web_df = similar_web_data_df.merge(eod_data_by_ticker, how='inner',  left_on=similar_web_data_df.index, right_on=eod_data_by_ticker.index)
                print(f"Hi End1: {eod_data_by_ticker}")
                final_similar_web_df.rename({"key_0": "Date"}, axis=1,inplace=True)
                final_similar_web_df.set_index("Date", inplace=True)
                metric_to_df_map["Similar Web Data"] = final_similar_web_df
                    
    if ticker:
        backtest_df = compute_rquared_and_correlations(metric_to_df_map, eod_data_by_ticker)
        st.write(f"Ticker: {ticker}")
        st.write(f"company website: {company_website}")
        sw_and_bt_tabs = st.tabs(["Backtest", "Similar Web"])
        
        with sw_and_bt_tabs[0]:
             st.header("Backtest")
             st.dataframe(backtest_df, height=800, width=1600)

        with sw_and_bt_tabs[1]:
             st.header("Similar Web")
             st.dataframe(metric_to_df_map["Similar Web Data"], height=800, width=1600)
             fig = px.scatter(
                metric_to_df_map["Similar Web Data"],
                x=metric_to_df_map["Similar Web Data"].Date,
                y=metric_to_df_map["Similar Web Data"]["Visits % Change"],
                size="pop",
                color="continent",
                hover_name="country",
                log_x=True,
                size_max=60,
            )
     
        sm_metrics = st.tabs(METRICS)
        for i, metric_i in enumerate(METRICS):
            with sm_metrics[i]:
                st.header(metric_i)
                st.altair_chart(metric_to_df_map[metric_i], *, use_container_width=False, theme="streamlit", key=None, on_select="ignore", selection_mode=None)
                # st.dataframe(metric_to_df_map[metric_i], height=800, width=1600)
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )