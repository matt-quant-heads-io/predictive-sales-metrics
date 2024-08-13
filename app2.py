import streamlit as st

import pandas as pd
import altair as alt
from urllib.error import URLError

import numpy as np

from helper import get_eod_api_data_by_ticker, process_second_measure_input_data

METRICS = ["Observed Sales","Observed Transactions","Observed Customers","Sales per Customer","Average Transaction Value","Transactions per Customer","Share of Sales","Share of Transactions"]


num_quarters_to_get_financials_for = 5

st.set_page_config(page_title="Sales Analysis", page_icon="📊", layout="wide")



st.markdown("# Sales Analysis")
st.sidebar.header("Sales Analysis")

@st.cache_data
def get_ticker_data():
    df = pd.read_csv("/Users/matt/predictive-sales-metrics/companies.csv")
    return df.set_index("Ticker")

df = None
ticker = None
revenue_dict = None
normalized_eod_df = None
final_df = None
company_website = None
metric_to_df_map = {}

try:
    df = get_ticker_data()
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Choose a file")
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
            

        if submitted:
            if uploaded_file is not None:
                # TODO: make this return a list of tickers once we add functionality for uploading zip files (list of ecxel files)
                
                for metric in METRICS:
                    df, ticker = process_second_measure_input_data(uploaded_file, metrics=[metric])
                    eod_data_by_ticker, company_website = get_eod_api_data_by_ticker(ticker, len(df.index)//3)
                                
                    revenue_dict = {"Month": list(df.index.values), "Total Revenue By Quarter": []}
                    # eod_data_by_ticker.set_index("Month", inplace=True)


                    for date_val in df.index.values:
                        # print(f"date_val: {date_val}")
                        if date_val in eod_data_by_ticker.index.values:
                            print(f"Found matching value!")
                            revenue_dict["Total Revenue By Quarter"].append(eod_data_by_ticker["Revenue"][date_val])
                        else:
                            revenue_dict["Total Revenue By Quarter"].append(np.NaN)
                            # revenue_dict = np.NaN
                    
                    normalized_eod_df = pd.DataFrame(revenue_dict)
                    normalized_eod_df.set_index("Month", inplace=True)
                    final_df = df.merge(normalized_eod_df, how='inner',  left_on=df.index, right_on=normalized_eod_df.index)
                    # final_df["Date"] = final_df["key_0"]
                    final_df.rename({"key_0": "Date"}, axis=1, inplace=True)
                    print(f"final_df: {final_df.head()}")
                    final_df.set_index("Date", inplace=True)
                    metric_to_df_map[metric] = final_df
                    # st.write(f"Ticker: {ticker}")
                    # st.write(f"company website: {company_website}")
                    # st.write(f"Months back: {len(df.index)}")
                    
                    # st.dataframe(final_df, height=800, width=1200) 
    
    if ticker:
        st.write(f"Ticker: {ticker}")
        st.write(f"company website: {company_website}")
        # st.dataframe(final_df, height=800, width=1200)

        # Add tabs here:
        obs_sales, obs_txns, obs_cust, sales_per_cust, avg_txn_val, txns_per_customer, share_of_sales, share_of_txns = st.tabs(METRICS)
        # [,,,,,]
        print(f"metric_to_df_map:\n{metric_to_df_map}")
        with obs_sales:
            st.header("Observed Sales")
            # print(f"obs_sales: {obs_sales}, metric_to_df_map.keys(): {metric_to_df_map.keys()}")
            # print(f"obs_sales: {obs_sales}")
            # st.dataframe(metric_to_df_map[obs_sales], height=800, width=1200)
            st.dataframe(metric_to_df_map["Observed Sales"], height=800, width=1200)
        with obs_txns:
            st.header("Observed Transactions")
            st.dataframe(metric_to_df_map["Observed Transactions"], height=800, width=1200)
        with obs_cust:
            st.header("Observed Customers")
            st.dataframe(metric_to_df_map["Observed Customers"], height=800, width=1200)
        with sales_per_cust:
            st.header("Sales per Customer")
            st.dataframe(metric_to_df_map["Sales per Customer"], height=800, width=1200)
        with avg_txn_val:
            st.header("Average Transaction Value")
            st.dataframe(metric_to_df_map["Average Transaction Value"], height=800, width=1200)
        with txns_per_customer:
            st.header("Transactions per Customer")
            st.dataframe(metric_to_df_map["Transactions per Customer"], height=800, width=1200)
        with share_of_sales:
            st.header("Share of Sales")
            st.dataframe(metric_to_df_map["Share of Sales"], height=800, width=1200)
        with share_of_txns:
            st.header("Share of Transactions")
            st.dataframe(metric_to_df_map["Share of Transactions"], height=800, width=1200)
        


        
            
   
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )