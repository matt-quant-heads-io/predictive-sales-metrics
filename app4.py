import streamlit as st
import sys

import pandas as pd
import altair as alt
from urllib.error import URLError

import numpy as np

from helper import test_compute_rquared_and_correlations, get_eod_api_data_by_ticker, process_second_measure_input_data, process_similarweb_data, compute_rquared_and_correlations, get_start_date_from_sm_file, test_process_similarweb_data, test_process_second_measure_input_data, test_get_start_date_from_sm_file

METRICS = ["Observed Sales","Observed Transactions","Observed Customers","Sales per Customer","Average Transaction Value","Transactions per Customer"] #["Observed Sales","Observed Transactions","Observed Customers","Sales per Customer","Average Transaction Value","Transactions per Customer","Share of Sales","Share of Transactions"]
SIMILAR_WEB_API_KEY = "7d015188e27e4ee1867f64a18213cbf5"
OUTPUT_DIR = "/home/ubuntu/predictive-sales-metrics/final_outputs"

st.set_page_config(page_title="Sales Analysis", page_icon="📊", layout="wide")
st.markdown("# Sales Analysis")
st.sidebar.header("Sales Analysis")

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
FILE_PREFIX = "DOCU_"
SM_PATH = f"/home/ubuntu/predictive-sales-metrics/input_data/{FILE_PREFIX}SM.csv"
SW_PATH = f"/home/ubuntu/predictive-sales-metrics/input_data/{FILE_PREFIX}SW.xlsx"
FILE_NAME = f"{FILE_PREFIX}SM.csv"

ticker = FILE_PREFIX.split("_")[0]
start_date = test_get_start_date_from_sm_file(SM_PATH, ticker=ticker)
eod_data_by_ticker, company_website = get_eod_api_data_by_ticker(ticker, start_date)
for metric in METRICS:
    df, ticker = test_process_second_measure_input_data(SM_PATH, eod_data_by_ticker, metrics=[metric], ticker=ticker)
    final_df = df.merge(eod_data_by_ticker, how='inner',  left_on=df.index, right_on=eod_data_by_ticker.index)
    final_df.rename({"key_0": "Date"}, axis=1, inplace=True)
    final_df.set_index("Date", inplace=True)
    metric_to_df_map[metric] = final_df

            

similar_web_data_df = test_process_similarweb_data(SW_PATH, eod_data_by_ticker)
similar_web_data_df.set_index("Quarterly Date", inplace=True)
print(f"==========================================================")
print(f"similar web index:\n{similar_web_data_df.index.values}\n\n")
print(f"eod index:\n{eod_data_by_ticker.index.values}\n\n")
print(f"==========================================================")
print(f"eod_data_by_ticker: {eod_data_by_ticker}")
final_similar_web_df = similar_web_data_df.merge(eod_data_by_ticker, how='inner',  left_on=similar_web_data_df.index, right_on=eod_data_by_ticker.index)
print(f"Hi End1: {eod_data_by_ticker}")
final_similar_web_df.rename({"key_0": "Date"}, axis=1,inplace=True)
final_similar_web_df.set_index("Date", inplace=True)
metric_to_df_map["Similar Web Data"] = final_similar_web_df
                    

backtest_df = test_compute_rquared_and_correlations(metric_to_df_map, eod_data_by_ticker)

# st.header("Backtest")
backtest_df.to_csv("Backtest.csv", index=True)
print(f"Hi End2: {eod_data_by_ticker}")
metric_to_df_map["Similar Web Data"].to_csv(f"{OUTPUT_DIR}/sw.csv", index=True)
print(f"Hi End3: {eod_data_by_ticker}")


for metric_i in METRICS:
    metric_to_df_map[metric_i].to_csv(f"{OUTPUT_DIR}/{metric_i}.csv", index=True)




    