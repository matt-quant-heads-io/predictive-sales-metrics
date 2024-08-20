import os
import json

import requests
import pandas as pd 
import scipy
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# from sklearn.metrics import r2_score
import math
from sec_edgar_api import EdgarClient
from sec_cik_mapper import StockMapper

from sklearn.preprocessing import StandardScaler


# NOTE: uncomment this for the first time you run the app
# import pip
# pip.main(["install", "openpyxl"])

EODHD_API_TOKEN = "667849cef19004.34002384"
EODHD_URL_FOR_FINANCIALS = 'https://eodhd.com/api/fundamentals/{ticker}.US?api_token={api_token}&fmt=json' 
SIMILAR_WEB_API_KEY = ""
#TODO: change this to get the path automatically
PATH_TO_INPUT_DATA = "/home/ubuntu/predictive-sales-metrics" #"/home/ubuntu/predictive-sales-metrics" #"/Users/matt/predictive-sales-metrics/input_data"


def get_cik_from_ticker(ticker):
    mapper = StockMapper()
    cik = mapper.ticker_to_cik[ticker.upper()]
    return cik

# def get_earnings_dates_from_edgar_api(ticker):
#     cik = get_cik_from_ticker(ticker)
#     edgar = EdgarClient(user_agent="<msf122133@gmail.com")
#     res = edgar.get_submissions(cik=cik)

#     recent_filings = res["filings"]["recent"]["primaryDocDescription"] #["primaryDocument"]
#     recent_report_dates = res["filings"]["recent"]["reportDate"]
#     relevant_indexes_for_report_dates = []
#     for i, recent_filing in enumerate(recent_filings):
#         recent_filing_lower_case = recent_filing.lower()
#         if recent_filing_lower_case == "10-q" or recent_filing_lower_case == "10-k":
#             relevant_indexes_for_report_dates.append(i)

#     relevant_dates = []
#     for j, report_date in enumerate(recent_report_dates):
#         if j in relevant_indexes_for_report_dates:
#             relevant_dates.append(report_date)

#     report_dates = pd.DataFrame({})
#     report_dates["report_dates"] = pd.to_datetime(relevant_dates)
#     report_dates["dummy_column"] = [i for i in range(len(relevant_dates))] 
#     report_dates.set_index("report_dates", inplace=True)
#     report_dates.sort_index(ascending=True)

#     return report_dates



def get_earnings_dates_from_edgar_api(ticker):
    headers = {'User-Agent': "email@address.com"}
    cik = get_cik_from_ticker(ticker)

    # get company specific filing metadata
    filingMetadata = requests.get(
        f'https://data.sec.gov/submissions/CIK{cik}.json',
        headers=headers
    )

    filingMetadata.json()['filings']
    filingMetadata.json()['filings'].keys()
    filingMetadata.json()['filings']['recent']
    filingMetadata.json()['filings']['recent'].keys()

    # dictionary to dataframe
    all_forms = pd.DataFrame.from_dict(
        filingMetadata.json()['filings']['recent']
    )
    
    cols_of_interest = [c for c in all_forms['primaryDocDescription'].unique() if '10-q' in c.lower() or '10-k' in c.lower()]
    all_forms = all_forms[all_forms['primaryDocDescription'].isin(cols_of_interest)]

    # import pdb 
    # pdb.set_trace()
    
    dummy_dict = {}
    dummy_dict["report_dates"] = pd.to_datetime(all_forms.reportDate.values)
    dummy_dict["dummy_data"] = [i for i in range(len(all_forms.reportDate.values))]
    report_dates_df = pd.DataFrame(dummy_dict)
    report_dates_df.set_index("report_dates", inplace=True)
    report_dates_df.sort_index(ascending=True)
    # Exxon, Chevron, Lilly

    # import pdb 
    # pdb.set_trace()
    
    return report_dates_df
    


def get_eod_api_data_by_ticker(ticker, oldest_dt_from_sm):
    try:
        req_url = EODHD_URL_FOR_FINANCIALS.format(ticker=ticker, api_token=EODHD_API_TOKEN)
        json_data = requests.get(req_url).json()
        
        web_url = json_data["General"]["WebURL"]
        data = json_data["Financials"]
        filing_dates = set()
        rev_data =[]
        all_date_rev_dicts = []
        for idx, (date, data_dict) in enumerate(data["Income_Statement"]["quarterly"].items()):
            all_date_rev_dicts.append({
                "Date": date,
                "rev": float(data_dict["totalRevenue"])
            })

        revelant_date_rev_dicts = {"Date": [], "Total Revenue By Quarter": []}
        for date_rev_dict in all_date_rev_dicts:
            date = date_rev_dict["Date"]
            rev = date_rev_dict["rev"]
            eod_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            if eod_date < oldest_dt_from_sm:
                continue

            revelant_date_rev_dicts["Date"].append(eod_date)
            revelant_date_rev_dicts["Total Revenue By Quarter"].append(rev)

        df = pd.DataFrame(revelant_date_rev_dicts)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True, ascending=False)
        df["Rev % Change"] = ((df["Total Revenue By Quarter"] - df["Total Revenue By Quarter"].shift(-4))/df["Total Revenue By Quarter"].shift(-4))*100

        actual_dates_df = get_earnings_dates_from_edgar_api(ticker)
        for i, idx_val in enumerate(df.index.values):
            trimmed_dates_df = actual_dates_df[actual_dates_df.index >= idx_val]
            if len(trimmed_dates_df)==0:
                continue
                
            actual_qtr_date = trimmed_dates_df.index.values[-1]
            df.rename(index={idx_val:actual_qtr_date},inplace=True)

        return df, web_url

             
    except Exception as e:
        print(f"Error: {e}")
        return {}

def get_start_date_from_sm_file(uploaded_file_obj, ticker=None):
    if ".csv" in uploaded_file_obj.name:
        df = pd.read_csv(uploaded_file_obj)
    else:
        raise ValueError(f"File must contain .csv extension!")

    df.rename({df.columns[0]: "Date"}, axis=1, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df.index[0]


def process_second_measure_input_data(uploaded_file_obj, rev_df, metrics=["Observed Sales"], ticker=None):
    if ".csv" in uploaded_file_obj.name:
        df = pd.read_csv(uploaded_file_obj)
    else:
        raise ValueError(f"File must contain .csv extension!")

    # print(f"Hi1")
    df.rename({df.columns[0]: "Date"}, axis=1, inplace=True)
    df = df[["Date", "Name"] + metrics]

    # Take unique list of brand ID
    brand_ids = list(df["Name"].unique())
    # dates = list(df["Date"].unique())
    # brand_ids = list(df["Name"].values)
    dates = list(df["Date"].values)
    

    merged_data_dict = {"Date": dates}
    for brand_id in brand_ids:
        df_by_brand = df[df["Name"] == brand_id]
        if brand_id not in merged_data_dict:
            merged_data_dict[brand_id] = []
        for metric_val in df_by_brand[metrics[0]].values:
            merged_data_dict[brand_id].append(metric_val)

    # for k,v in merged_data_dict.items():
    #     print(f"{k}={len(v)}")
    df_metrics_by_brand = pd.DataFrame(merged_data_dict)
    df_metrics_by_brand["Date"] = pd.to_datetime(df_metrics_by_brand["Date"])

    
    # df_metrics_by_brand.set_index("Month", inplace=True)
    df_metrics_by_brand["Quarterly Bucket"] = df_metrics_by_brand["Date"].apply(lambda val: determine_quarter(val, rev_df))
    # print(df_metrics_by_brand["Quarterly Bucket"])

    results_dict = {k:[] for k in df_metrics_by_brand.columns if k not in ["Quarterly Bucket"]}
    # results_dict["Month"] = list(df_metrics_by_brand["Month"].values)
    for bucket in df_metrics_by_brand["Quarterly Bucket"].unique():
        df_bucket = df_metrics_by_brand[df_metrics_by_brand["Quarterly Bucket"] == bucket]
        for col in df_metrics_by_brand.columns:
            if col == "Date":
                results_dict[col].append(bucket)
            elif col == "Quarterly Bucket":
                continue
            else:
                if metrics[0] in ["Observed Sales","Observed Transactions","Observed Customers"]:
                    col_sum = df_bucket[col].sum()
                    results_dict[col].append(col_sum)
                else:
                    col_mean = df_bucket[col].mean()
                    results_dict[col].append(col_mean)
        
    new_sm_df = pd.DataFrame(results_dict)
    new_sm_df.set_index("Date", inplace=True)
    new_sm_df.sort_index(inplace=True, ascending=False)
    # print(f"new_sm_df:\n{new_sm_df}")

    if metrics[0] in ["Observed Sales","Observed Transactions","Observed Customers"]:
        new_sm_df["All"] = new_sm_df.apply(lambda row: row[brand_ids].sum(), axis=1)
    else:
        new_sm_df["All"] = new_sm_df.apply(lambda row: row[brand_ids].mean(), axis=1)

    new_sm_df["All % Change"] = (new_sm_df["All"] - new_sm_df["All"].shift(-4))/new_sm_df["All"].shift(-4)
    new_sm_df["All % Change"] *= 100.0

    return new_sm_df, ticker


def determine_quarter(given_date, rev_df):
    # print(f"rev_df: {rev_df}")
    given_year, given_month, given_day = str(given_date).split("-")
    given_day = given_day.split(" ")[0]
    # print(f"PRIOR rev_df for date {given_date}:\n{rev_df}")
    new_rev_df = rev_df[rev_df.index >= given_date]
    # print(f"AFTER rev_df for date {given_date}:\n{new_rev_df}")
    return new_rev_df.index.values[-1] if len(new_rev_df.index.values) > 0 else given_date
    

def process_similarweb_data(uploaded_file_obj, rev_df):
    if ".xlsx" in uploaded_file_obj.name:
        similar_web_df = pd.read_excel(uploaded_file_obj, sheet_name="Total")
    else:
        raise ValueError(f"File must contain .xlsx extension!")

    similar_web_df.rename({similar_web_df.columns[0]: "Date"}, axis=1, inplace=True)
    similar_web_df["Quarterly Bucket"] = similar_web_df["Date"].apply(lambda val: determine_quarter(val, rev_df))
    unique_quarterly_values = list(similar_web_df["Quarterly Bucket"].unique())
    similar_web_df.set_index("Quarterly Bucket", inplace=True)
    similar_web_df.sort_index(inplace=True, ascending=False)
    
    # print(f"similar_web_df: {similar_web_df}")
    
    similar_web_df["Visits % Change"] = (similar_web_df["Visits"] - similar_web_df["Visits"].shift(-4))/similar_web_df["Visits"].shift(-4)
    similar_web_df["Visits % Change"] *= 100.0
    similar_web_df["Unique Visitors % Change"] = (similar_web_df["Unique Visitors"] - similar_web_df["Unique Visitors"].shift(-4))/similar_web_df["Unique Visitors"].shift(-4)
    similar_web_df["Unique Visitors % Change"] *= 100.0
    similar_web_df["Total Page Views % Change"] = (similar_web_df["Total Page Views"] - similar_web_df["Total Page Views"].shift(-4))/similar_web_df["Total Page Views"].shift(-4)
    similar_web_df["Total Page Views % Change"] *= 100.0
    

    
    # results_dict = {"Quarterly Date": [], "Visits": [], "Unique Visitors": [], "Total Page Views": [], "Visits % Change": [], "Unique Visitors % Change": [], "Total Page Views % Change": []}
    results_dict = {"Quarterly Date": [], "Visits": [], "Unique Visitors": [], "Total Page Views": []}
    for bucket in unique_quarterly_values:
        df_bucket = similar_web_df[similar_web_df.index == bucket]
        
        visits_sum = df_bucket["Visits"].sum()
        results_dict["Visits"].append(visits_sum)
        # visits_pct_chg_avg = df_bucket["Visits % Change"].mean()
        # results_dict["Visits % Change"].append(visits_pct_chg_avg)

        unique_visits_sum = df_bucket["Unique Visitors"].sum()
        results_dict["Unique Visitors"].append(unique_visits_sum)
        # unique_visits_cfg_avg = df_bucket["Unique Visitors % Change"].mean()
        # results_dict["Unique Visitors % Change"].append(unique_visits_cfg_avg)
        
        
        total_page_views_sum = df_bucket["Total Page Views"].sum()
        results_dict["Total Page Views"].append(total_page_views_sum) 
        # total_page_views_pct_chg_avg = df_bucket["Total Page Views % Change"].mean()
        # results_dict["Total Page Views % Change"].append(total_page_views_pct_chg_avg)

        start_date_bound = str(df_bucket.index.values[0]).split("T")[0]
        end_date_bound = str(df_bucket.index.values[-1]).split("T")[0]
        results_dict["Quarterly Date"].append(bucket)

    new_sw_df = pd.DataFrame(results_dict)
    new_sw_df.set_index("Quarterly Date", inplace=True)
    new_sw_df.sort_index(inplace=True, ascending=False)

    new_sw_df["Visits % Change"] = ((new_sw_df["Visits"] - new_sw_df["Visits"].shift(-4))/new_sw_df["Visits"].shift(-4))*100
    new_sw_df["Unique Visitors % Change"] = ((new_sw_df["Unique Visitors"] - new_sw_df["Unique Visitors"].shift(-4))/new_sw_df["Unique Visitors"].shift(-4))*100
    new_sw_df["Total Page Views % Change"] = ((new_sw_df["Total Page Views"] - new_sw_df["Total Page Views"].shift(-4))/new_sw_df["Total Page Views"].shift(-4))*100
    
    return new_sw_df


def r_and_rsquared(df, x, y):
    corr = df[x].corr(df[y], method="pearson")
    return corr, corr**2


def compute_rsquared_and_correlations_for_similarweb_data(k, df, rev_df, results_dict):
    df['Rev % Change'] = rev_df['Rev % Change']
    corr, rsquared = r_and_rsquared(df, "Visits", "Total Revenue By Quarter")
    results_dict["x"].append("Visits")
    results_dict["y"].append("Revenue By Quarter")
    results_dict["Correlation"].append(corr)
    results_dict["R_squared"].append(rsquared)
    
    corr, rsquared = r_and_rsquared(df, "Unique Visitors", "Total Revenue By Quarter")
    results_dict["x"].append("Unique Visitors")
    results_dict["y"].append("Revenue By Quarter")
    results_dict["Correlation"].append(corr)
    results_dict["R_squared"].append(rsquared)
    
    corr, rsquared = r_and_rsquared(df, "Total Page Views", "Total Revenue By Quarter")
    results_dict["x"].append("Total Page Views")
    results_dict["y"].append("Revenue By Quarter")
    results_dict["Correlation"].append(corr)
    results_dict["R_squared"].append(rsquared)
    
    corr, rsquared = r_and_rsquared(df, "Visits % Change", "Rev % Change")
    results_dict["x"].append("Visits % Change")
    results_dict["y"].append("Rev % Change")
    results_dict["Correlation"].append(corr)
    results_dict["R_squared"].append(rsquared)
    
    corr, rsquared = r_and_rsquared(df, "Unique Visitors % Change", "Rev % Change")
    results_dict["x"].append("Unique Visitors % Change")
    results_dict["y"].append("Rev % Change")
    results_dict["Correlation"].append(corr)
    results_dict["R_squared"].append(rsquared)
    
    corr, rsquared = r_and_rsquared(df, "Total Page Views % Change", "Rev % Change")
    results_dict["x"].append("Total Page Views % Change")
    results_dict["y"].append("Rev % Change")
    results_dict["Correlation"].append(corr)
    results_dict["R_squared"].append(rsquared)
    return results_dict


def compute_rsquared_and_correlations_for_second_measure_data(k, df, rev_df, results_dict):
    df['Rev % Change'] = rev_df['Rev % Change']
    corr, rsquared = r_and_rsquared(df, "All", "Total Revenue By Quarter")
    
    results_dict["x"].append(f"{k} (All)")
    results_dict["y"].append("Revenue By Quarter")
    results_dict["Correlation"].append(corr)
    results_dict["R_squared"].append(rsquared)
    
    corr, rsquared = r_and_rsquared(df, "All % Change", "Rev % Change")
    results_dict["x"].append(f"{k} (All % Change)")
    results_dict["y"].append("Rev % Change")
    results_dict["Correlation"].append(corr)
    results_dict["R_squared"].append(rsquared)
    return results_dict


def compute_rquared_and_correlations(metric_to_dataframe, rev_df):
    results_dict = {"x": [], "y":[], "R_squared": [], "Correlation": []}
    
    for k, df in metric_to_dataframe.items():
        if k == "Similar Web Data":
            compute_rsquared_and_correlations_for_similarweb_data(k, df, rev_df, results_dict)
        else:
            # print(f"about to compute stats for df[{k}]:\n{df}")
            compute_rsquared_and_correlations_for_second_measure_data(k, df, rev_df,results_dict)

    results_df = pd.DataFrame(results_dict)
    return results_df


def generate_predictors_tab(metric_to_df_map, raw_dfs_map):
    # SM
    obs_sales_pct_chg = metric_to_df_map["Observed Sales"]["All % Change"]
    obs_txns_pct_chg = metric_to_df_map["Observed Transactions"]["All % Change"]
    obs_cust_pct_chg = metric_to_df_map["Observed Customers"]["All % Change"]
    sals_per_cust_pct_chg = metric_to_df_map["Sales per Customer"]["All % Change"]
    avg_txns_pct_chg = metric_to_df_map["Average Transaction Value"]["All % Change"]
    txn_per_cust_pct_chg = metric_to_df_map["Transactions per Customer"]["All % Change"]

    # SW
    visits_pct_chg = metric_to_df_map["Similar Web Data"]["Visits % Change"]
    unique_visits_pct_chg = metric_to_df_map["Similar Web Data"]["Unique Visitors % Change"]
    views_pct_chg = metric_to_df_map["Similar Web Data"]["Total Page Views % Change"]
    rev_pct_chg = metric_to_df_map["Similar Web Data"]["Rev % Change"]

    names_to_vals_tuples = [
        ("obs_sales_pct_chg", obs_sales_pct_chg),
        ("obs_txns_pct_chg", obs_txns_pct_chg),
        ("obs_cust_pct_chg", obs_cust_pct_chg),
        ("sals_per_cust_pct_chg", sals_per_cust_pct_chg),
        ("avg_txns_pct_chg", avg_txns_pct_chg),
        ("txn_per_cust_pct_chg", txn_per_cust_pct_chg),
        ("visits_pct_chg", visits_pct_chg),
        ("unique_visits_pct_chg", unique_visits_pct_chg),
        ("views_pct_chg", views_pct_chg),
        ("rev_pct_chg", rev_pct_chg),
    ]

    data_dict = {n[0]:[] for n in names_to_vals_tuples}
    for (n, v) in names_to_vals_tuples:
        for v_i in v.values:
            data_dict[n].append(v_i)

    new_df = pd.DataFrame(data_dict)
    new_df["Date"] = metric_to_df_map["Observed Sales"].index.values
    new_df.set_index("Date", inplace=True)
    
    ss = StandardScaler()
    new_df_scaled = pd.DataFrame(ss.fit_transform(new_df),columns = new_df.columns)
    new_df_scaled = new_df_scaled[:len(new_df_scaled)-1]

    data = new_df_scaled[["obs_sales_pct_chg","obs_txns_pct_chg","obs_cust_pct_chg","sals_per_cust_pct_chg","avg_txns_pct_chg","txn_per_cust_pct_chg","visits_pct_chg","unique_visits_pct_chg","views_pct_chg"]]
    target = new_df_scaled[["rev_pct_chg"]]
    
    train_split = len(data) -  max(2, int(len(data)*0.15))
    test_pts = len(data) - train_split
    train_pts = len(data) - test_pts
    
    X_train = data[:] #data[:train_split]
    y_train = target[:]
    ss_target = StandardScaler().fit(y_train)


    X_test = data[train_split:]
    y_test = target[train_split:]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    rev_pct_chg_pred = regr.predict(X_test)

    mse = mean_squared_error(y_test, rev_pct_chg_pred)
    r2 = r2_score(y_test, rev_pct_chg_pred)
    coeffs_df = pd.DataFrame(regr.coef_, columns=["obs_sales_pct_chg","obs_txns_pct_chg","obs_cust_pct_chg","sals_per_cust_pct_chg","avg_txns_pct_chg","txn_per_cust_pct_chg","visits_pct_chg","unique_visits_pct_chg","views_pct_chg"])

    
    # CURRENT DATA
    # SM
    intersect_raw_dfs_map = raw_dfs_map.copy() #{}
    intersect_dict = {}
    for k, v in intersect_raw_dfs_map.items():
        intersect_raw_dfs_map[k].sort_index(ascending=False, inplace=True)

    for k, v in intersect_raw_dfs_map.items():
        if k == "Similar Web Data":
            continue
        else:
            intersect_raw_dfs_map[k].rename({"All % Change": k}, axis=1, inplace=True)

    sw_index = intersect_raw_dfs_map["Similar Web Data"].index.values
    sm_index = intersect_raw_dfs_map["Observed Sales"].index.values

    index_values = set()
    all_index_vals = []
    for z in raw_dfs_map["Observed Sales"].index.values:
        all_index_vals.append(z)

    for z in raw_dfs_map["Similar Web Data"].index.values:
        all_index_vals.append(z)



    for i in all_index_vals:
        if i in sw_index and i in sw_index:
            index_values.add(i)

    
    ref_index = list(index_values) 
    
    intersect_dict2 = {}
    common_index = set()
    for k, _df in intersect_raw_dfs_map.items():
        if k == "Similar Web Data":
            intersect_dict2["Visits % Change"] = []
            intersect_dict2["Unique Visitors % Change"] = []
            intersect_dict2["Total Page Views % Change"] = []
            for index in ref_index:
                if index not in intersect_raw_dfs_map[k].index.values:
                    continue
                common_index.add(index)
                vals_df = intersect_raw_dfs_map["Similar Web Data"][intersect_raw_dfs_map[k].index == index]
                intersect_dict2["Visits % Change"].extend(vals_df["Visits % Change"])
                intersect_dict2["Unique Visitors % Change"].extend(vals_df["Unique Visitors % Change"])
                intersect_dict2["Total Page Views % Change"].extend(vals_df["Total Page Views % Change"])

        else:
            intersect_dict2[k] = []
            for index in ref_index:
                if index not in intersect_raw_dfs_map[k].index.values:
                    continue
                common_index.add(index)
                vals_df = intersect_raw_dfs_map[k][intersect_raw_dfs_map[k].index == index] 
                intersect_dict2[k].extend(vals_df[k])

    length = min([len(v) for k,v in intersect_dict2.items()] + [len(list(common_index))])

    # length = min_len if len(intersect_dict2) > len(list(common_index)) else len(list(common_index))
    for k, v in intersect_dict2.items():
        intersect_dict2[k] = v[:length]
    
    current_df = pd.DataFrame(intersect_dict2)
    current_df["Date"] = list(common_index)[:length]
    current_df.set_index("Date", inplace=True)
    current_df.sort_index(ascending=False, inplace=True)
    current_df.dropna(inplace=True)
    current_df = current_df[current_df.index > new_df.index.values[0]]
    
    current_scaled = pd.DataFrame(ss.fit_transform(current_df),columns = data.columns)
    current_scaled.dropna(inplace=True)
    current_index_values = current_df.index.values
    rev_pct_chg_pred_scaled = regr.predict(current_scaled)

    rev_pct_chg_pred_descaled = ss_target.inverse_transform(rev_pct_chg_pred_scaled)

    current_preds_df = pd.DataFrame({"Rev % Pred": rev_pct_chg_pred_descaled.reshape(-1)})
    current_preds_df["Dates"] = current_index_values
    current_preds_df.set_index("Dates",inplace=True)

    return coeffs_df, mse, r2, test_pts, train_pts, current_preds_df



# def generate_tracking_page(df_sw):
#     # print(f"df_sw: {df_sw}")
#     # import pdb
#     # pdb.set_trace()
#     current_dt = df_sw.index.values[0]
#     current_year = df_sw.index.year[0]
#     # import pdb 
#     # pdb.set_trace()

    

#     start_dt_for_current_year = np.datetime64(f'{current_year}-01-01')
#     month_prefix = '' if df_sw.index.month[0] >= 10 else '0'
#     day_prefix = '' if df_sw.index.day[0] >= 10 else '0'
#     cutoff_dt_for_prior_year = np.datetime64(f'{current_year-1}-{month_prefix}{df_sw.index.month[0]}-{day_prefix}{df_sw.index.day[0]}')

#     df_prior_year = df_sw[(df_sw.index >= np.datetime64(f'{current_year-1}-01-01')) & ((df_sw.index <= cutoff_dt_for_prior_year))]
#     df_prior_year["Visits_CumSum"] = df_prior_year[::-1]["Visits"].cumsum()
#     df_prior_year["Unique_CumSum"] = df_prior_year[::-1]["Unique Visitors"].cumsum()
#     df_prior_year["Views_CumSum"] = df_prior_year[::-1]["Total Page Views"].cumsum()
#     # df_prior_year["Current_Year_Index"] = 
#     index_alignments_for_prior_years = []
#     for i in df_prior_year.index:
#         i_month_prefix = '' if i.month >= 10 else '0'
#         i_day_prefix = '' if i.day >= 10 else '0'
#         index_alignments_for_prior_years.append(np.datetime64(f'{current_year}-{i_month_prefix}{i.month}-{i_day_prefix}{i.day}'))

#     df_prior_year["index_alignments_for_prior_years"] = index_alignments_for_prior_years
#     df_prior_year["Quarterly"] = df_prior_year.index.values
#     df_prior_year.set_index("index_alignments_for_prior_years", inplace=True)

#     df_current_year = df_sw[(df_sw.index >= start_dt_for_current_year) & (df_sw.index <= current_dt)]
#     df_current_year["Visits_CumSum"] = df_current_year[::-1]["Visits"].cumsum()
#     df_current_year["Unique_CumSum"] = df_current_year[::-1]["Unique Visitors"].cumsum()
#     df_current_year["Views_CumSum"] = df_current_year[::-1]["Total Page Views"].cumsum()

#     return df_current_year, df_prior_year


def generate_tracking_page(ticker, sw_uploaded_file, df_sw):
    if ".xlsx" in sw_uploaded_file.name:
        df_sw = pd.read_excel(sw_uploaded_file, sheet_name='Total')
    else:
        raise ValueError(f"File must contain .csv extension!")

    # print(f"Hi1")
    df_sw.rename({df_sw.columns[0]: "Date"}, axis=1, inplace=True)
    df_sw["Date"] = pd.to_datetime(df_sw["Date"])
    df_sw.set_index("Date", inplace=True)
    # df_sw = df_sw[["Date", "Name"] + metrics]
    
    df_sw.sort_index(ascending=False, inplace=True)

    current_dt = df_sw.index.values[0]
    current_year = df_sw.index.year[0]

    this_year_start_date = df_sw.index.date[0] + datetime.timedelta(days=-365)
    this_year_end_date = df_sw.index.date[0]
    
    last_year_end_date = this_year_start_date + datetime.timedelta(days=-1)
    last_year_start_date = this_year_start_date + datetime.timedelta(days=-365)
    actual_dates_df = get_earnings_dates_from_edgar_api(ticker)
    actual_dates_df
    df_calcs = pd.DataFrame({})
    df_calcs["end_dates"] = actual_dates_df.index.values
    df_calcs["start_dates"] = df_calcs["end_dates"].shift(-4)
    df_calcs["start_dates"] = df_calcs["start_dates"] + np.timedelta64(1,'D')
    visits_cumsums = []
    unique_cumsums = []
    views_cumsums = []

    # import pdb 
    # pdb.set_trace()

    df_calcs = df_calcs[df_calcs["start_dates"] >= df_sw.index.values[-1]]
    # df_calcs["Visits_CumSum"] = [0]*len()

    # import pdb 
    # pdb.set_trace()

    
    visits = []
    uniques = []
    views = []
    dates_starts = []
    dates_ends = []
    raw_dates = []


    for i in range(len(df_calcs)):
        if i > 3:
            break
        end_dates, start_dates = df_calcs.iloc[i, :2]
        raw_index_vals = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)].index.values
        raw_dates.extend(raw_index_vals)
        visits_i = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Visits"].values
        visits.extend(visits_i)
        dates_starts.extend([dates_starts]*len(visits_i))
        dates_ends.extend([dates_ends]*len(visits_i))
        
        uniques_i = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Unique Visitors"].values
        uniques.extend(uniques_i)


        views_i = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Total Page Views"].values
        views.extend(views_i)

    print(len(visits), len(uniques), len(views), len(dates_starts), len(dates_ends))
    df_totals = pd.DataFrame({"Dates":raw_dates,"dates_starts": dates_starts, "dates_ends": dates_ends, "visits": visits, "uniques": uniques, "views": views})
    df_totals.set_index("Dates", inplace=True)
    df_totals.sort_index(ascending=False, inplace=True)


    prior_visits = []
    prior_uniques = []
    prior_views = []
    prior_dates_starts = []
    prior_dates_ends = []
    prior_raw_dates = []


    for i in range(4, len(df_calcs)):
        if i > 7:
            break
        end_dates, start_dates = df_calcs.iloc[i, :2]
        raw_index_vals = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)].index.values
        prior_raw_dates.extend(raw_index_vals)
        visits_i = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Visits"].values
        prior_visits.extend(visits_i)
        prior_dates_starts.extend([prior_dates_starts]*len(visits_i))
        prior_dates_ends.extend([prior_dates_ends]*len(visits_i))
        
        uniques_i = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Unique Visitors"].values
        prior_uniques.extend(uniques_i)


        views_i = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Total Page Views"].values
        prior_views.extend(views_i)

    print(len(prior_raw_dates), len(prior_visits), len(prior_uniques), len(prior_views), len(prior_dates_starts), len(prior_dates_ends))
    df_totals_last_year = pd.DataFrame({"Dates":prior_raw_dates,"dates_starts": prior_dates_starts, "dates_ends": prior_dates_ends, "visits": prior_visits, "uniques": prior_uniques, "views": prior_views})
    df_totals_last_year.set_index("Dates", inplace=True)
    df_totals_last_year.sort_index(ascending=False, inplace=True)
    # import pdb 
    # pdb.set_trace()
    return df_totals_last_year, df_totals
        

    


# for i in range(len(df_calcs)):
#     end_dates, start_dates = df_calcs.iloc[i, :2]
#     visits_cumsum = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Visits"][::-1]
#     visits_cumsums.append(visits_cumsum)
    
#     unique_cumsum = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Unique Visitors"][::-1]
#     unique_cumsums.append(unique_cumsum)


#     views_cumsum = df_sw[(df_sw.index >= start_dates) & (df_sw.index <= end_dates)]["Total Page Views"][::-1]
#     views_cumsums.append(views_cumsum)    
#     # import pdb 
#     # pdb.set_trace()

# df_calcs["Visits_CumSum"] = visits_cumsum
# df_calcs["Uniques_CumSum"] = unique_cumsum
# df_calcs["Views_CumSum"] = views_cumsum

# # import pdb 
# # pdb.set_trace()



# start_dt_for_current_year = np.datetime64(f'{current_year}-01-01')
# month_prefix = '' if df_sw.index.month[0] >= 10 else '0'
# day_prefix = '' if df_sw.index.day[0] >= 10 else '0'
# cutoff_dt_for_prior_year = np.datetime64(f'{current_year-1}-{month_prefix}{df_sw.index.month[0]}-{day_prefix}{df_sw.index.day[0]}')

# df_prior_year = df_sw[(df_sw.index >= np.datetime64(str(last_year_start_date))) & (df_sw.index <= np.datetime64(str(last_year_end_date)))]
# df_this_year = df_sw[(df_sw.index >= np.datetime64(str(this_year_start_date))) & (df_sw.index <= np.datetime64(str(this_year_end_date)))]
# df_prior_year["Visits_CumSum"] = df_prior_year[::-1]["Visits"].cumsum()
# df_prior_year["Unique_CumSum"] = df_prior_year[::-1]["Unique Visitors"].cumsum()
# df_prior_year["Views_CumSum"] = df_prior_year[::-1]["Total Page Views"].cumsum()
# # df_prior_year["Current_Year_Index"] = 
# index_alignments_for_prior_years = []
# for i in df_prior_year.index:
#     i_month_prefix = '' if i.month >= 10 else '0'
#     i_day_prefix = '' if i.day >= 10 else '0'
#     index_alignments_for_prior_years.append(np.datetime64(f'{current_year}-{i_month_prefix}{i.month}-{i_day_prefix}{i.day}'))

# df_prior_year["index_alignments_for_prior_years"] = index_alignments_for_prior_years
# df_prior_year["Quarterly"] = df_prior_year.index.values
# df_prior_year.set_index("index_alignments_for_prior_years", inplace=True)

# df_current_year = df_sw[(df_sw.index >= start_dt_for_current_year) & (df_sw.index <= current_dt)]
# df_current_year["Visits_CumSum"] = df_current_year[::-1]["Visits"].cumsum()
# df_current_year["Unique_CumSum"] = df_current_year[::-1]["Unique Visitors"].cumsum()
# df_current_year["Views_CumSum"] = df_current_year[::-1]["Total Page Views"].cumsum()

# return df_current_year, df_prior_year

    