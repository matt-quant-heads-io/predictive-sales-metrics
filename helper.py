import os
import json

import requests
import pandas as pd
import numpy as np
from sec_cik_mapper import StockMapper
from sklearn.preprocessing import StandardScaler


# NOTE: uncomment this for the first time you run the app
# import pip
# pip.main(["install", "openpyxl"])


EODHD_API_TOKEN = "667849cef19004.34002384"
EODHD_URL_FOR_FINANCIALS = (
    "https://eodhd.com/api/fundamentals/{ticker}.US?api_token={api_token}&fmt=json"
)
PATH_TO_INPUT_DATA = "/home/ubuntu/predictive-sales-metrics"


def get_cik_from_ticker(ticker):
    mapper = StockMapper()
    cik = mapper.ticker_to_cik[ticker.upper()]
    return cik


def get_earnings_dates_from_edgar_api(ticker):
    headers = {"User-Agent": "email@address.com"}
    cik = get_cik_from_ticker(ticker)

    filingMetadata = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers
    )

    filingMetadata.json()["filings"]
    filingMetadata.json()["filings"].keys()
    filingMetadata.json()["filings"]["recent"]
    filingMetadata.json()["filings"]["recent"].keys()

    all_forms = pd.DataFrame.from_dict(filingMetadata.json()["filings"]["recent"])
    cols_of_interest = [
        c
        for c in all_forms["primaryDocDescription"].unique()
        if "10-q" in c.lower() or "10-k" in c.lower()
    ]
    all_forms = all_forms[all_forms["primaryDocDescription"].isin(cols_of_interest)]

    dummy_dict = {}
    dummy_dict["report_dates"] = pd.to_datetime(all_forms.reportDate.values)
    dummy_dict["dummy_data"] = [i for i in range(len(all_forms.reportDate.values))]
    report_dates_df = pd.DataFrame(dummy_dict)
    report_dates_df.set_index("report_dates", inplace=True)

    return report_dates_df


def get_start_date_from_sm_file(uploaded_file_obj, ticker=None):
    df = pd.read_csv(uploaded_file_obj)
    df.rename({df.columns[0]: "Date"}, axis=1, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df.index[0]


def determine_quarter(given_date, rev_df):
    _, _, given_day = str(given_date).split("-")
    given_day = given_day.split(" ")[0]
    new_rev_df = rev_df[rev_df.index >= given_date]
    return (
        new_rev_df.index.values[-1] if len(new_rev_df.index.values) > 0 else given_date
    )


def get_eod_api_data_by_ticker(ticker, df_sm):
    try:
        oldest_dt_from_sm = np.datetime64(str(df_sm.index.date[-1]))
        req_url = EODHD_URL_FOR_FINANCIALS.format(
            ticker=ticker.upper(), api_token=EODHD_API_TOKEN
        )
        json_data = requests.get(req_url).json()

        data = json_data["Financials"]
        all_date_rev_dicts = []
        for date, data_dict in data["Income_Statement"]["quarterly"].items():
            all_date_rev_dicts.append(
                {"Date": np.datetime64(date), "rev": float(data_dict["totalRevenue"])}
            )

        revelant_date_rev_dicts = {"Date": [], "Total Revenue By Quarter": []}
        for date_rev_dict in all_date_rev_dicts:
            date = date_rev_dict["Date"]
            rev = date_rev_dict["rev"]

            revelant_date_rev_dicts["Date"].append(date)
            revelant_date_rev_dicts["Total Revenue By Quarter"].append(rev)

        df_rev_all_columns = pd.DataFrame(revelant_date_rev_dicts)
        df_rev_all_columns["Date"] = pd.to_datetime(df_rev_all_columns["Date"])
        df_rev_all_columns.set_index("Date", inplace=True)
        df_rev_all_columns.sort_index(inplace=True, ascending=False)
        df_rev_all_columns["Rev % Change"] = (
            (
                df_rev_all_columns["Total Revenue By Quarter"]
                - df_rev_all_columns["Total Revenue By Quarter"].shift(-4)
            )
            / df_rev_all_columns["Total Revenue By Quarter"].shift(-4)
        ) * 100
        df_rev_all_columns = df_rev_all_columns[
            df_rev_all_columns.index >= oldest_dt_from_sm
        ]

        actual_dates_df = get_earnings_dates_from_edgar_api(ticker)
        for idx_val in df_rev_all_columns.index.values:
            trimmed_dates_df = actual_dates_df[actual_dates_df.index >= idx_val]
            if len(trimmed_dates_df) == 0:
                continue

            actual_qtr_date = trimmed_dates_df.index.values[-1]
            df_rev_all_columns.rename(index={idx_val: actual_qtr_date}, inplace=True)

        df_rev_all_columns.sort_index(ascending=False, inplace=True)

        return df_rev_all_columns
    except Exception as e:
        print(f"Error: {e}")
        raise e


def r_and_rsquared(df, x, y):
    corr = df[x].corr(df[y], method="pearson")
    return corr, corr**2
