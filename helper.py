import datetime

import requests
import pandas as pd
import numpy as np
from sec_cik_mapper import StockMapper

from constants import (
    EODHD_API_TOKEN,
    EODHD_URL_FOR_FINANCIALS
)


# NOTE: uncomment this for the first time you run the app
# import pip
# pip.main(["install", "openpyxl"])


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


def get_start_date_from_sm_file(uploaded_file_obj):
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


def assign_reporting_date_to_df_date_row_helper(given_date, rev_df, default_date):
    truncated_reporting_dates = rev_df[rev_df.index >= given_date]
    return (
        truncated_reporting_dates.index.values[-1]
        if len(truncated_reporting_dates.index.values) > 0
        else default_date
    )


def assign_reporting_date_to_df_date_row(df, revenue_df):
    default_date = df.index.values[0]
    df["TempDateCol"] = df.index.values
    df["Quarter Date"] = df["TempDateCol"].apply(
        lambda val: assign_reporting_date_to_df_date_row_helper(
            val, revenue_df, default_date
        )
    )
    df.drop("TempDateCol", axis=1, inplace=True)
    return df


def add_is_quarter_date_valid_column_helper(given_date, most_recent_quarter):
    return given_date <= most_recent_quarter


def add_is_quarter_date_valid_column(df, revenue_df):
    most_recent_quarter = revenue_df.index.values[0]
    df["TempDateCol"] = df.index.values
    df["is_date_valid"] = df["TempDateCol"].apply(
        lambda val: add_is_quarter_date_valid_column_helper(val, most_recent_quarter)
    )
    df.drop("TempDateCol", axis=1, inplace=True)
    return df


def compute_cumsums_and_means_by_quarterly_date(df, cols_of_interest=None):
    unique_quarter_dates = list(df["Quarter Date"].unique())
    for metric in cols_of_interest:
        metric_vals_cumsums_by_quarter = []
        for unique_quarter_date in unique_quarter_dates:
            output = (
                df[df["Quarter Date"] == unique_quarter_date][metric]
                .cumsum()[::-1]
                .values.tolist()
            )
            metric_vals_cumsums_by_quarter.extend(output)

        df[f"{metric} CumSum"] = metric_vals_cumsums_by_quarter

    return df


def collapse_df_to_quarterly(df):
    unique_quarter_dates = list(df["Quarter Date"].unique())[1:]
    data_dict = {c: [] for c in df.columns if c.endswith(" CumSum")}
    for metric in list(data_dict.keys()):
        metric_vals_cumsums_by_quarter = []
        for unique_quarter_date in unique_quarter_dates:
            output = df[(df["Quarter Date"] == unique_quarter_date)][metric][
                ::-1
            ].values.tolist()
            metric_vals_cumsums_by_quarter.append(max(output))
        data_dict[metric] = metric_vals_cumsums_by_quarter

    data_dict["Date"] = unique_quarter_dates
    df_cumsums_quarterly = pd.DataFrame(data_dict)
    df_cumsums_quarterly.set_index("Date", inplace=True)
    df_cumsums_quarterly.sort_index(ascending=False, inplace=True)

    return df_cumsums_quarterly


def add_growth_rates_to_quarterly_cumsums_df(df):
    for col in df.columns:
        df[f"{col} growth (%)"] = (
            (df[col] - df[col].shift(-4)) / df[col].shift(-4)
        ) * 100
    df.dropna(inplace=True)

    return df


def merge_rev_columns_to(src_df, rev_df):
    """
    TODO: Check if there's a full quarter that is being loaded from the excel sheet
        If not then chop it off. For example, for quarter ending 2019-06-30, the sheet uploaded would have to start
        at 2019-04-01 otherwise the data for the quarter ending on 2019-06-30 would be incomplete in which case we would chop it off!
    """
    merged_src_df = src_df.merge(
        rev_df, how="inner", left_on=src_df.index, right_on=rev_df.index
    )
    merged_src_df.rename({"key_0": "Date"}, axis=1, inplace=True)
    merged_src_df.set_index("Date", inplace=True)
    plot_index_vals = []
    for i in merged_src_df.index.date:
        _, month, day = str(i).split("-")
        plot_index_vals.append(f"{month}-{day}")

    merged_src_df["Plot Index"] = plot_index_vals
    return merged_src_df


def get_stats_df(df):
    cols_to_compute = [c for c in df.columns if c.endswith("growth (%)")]
    data_dict = {"Input": [], "Correlation": [], "r_squared": []}

    for col in cols_to_compute:
        corr, r2 = r_and_rsquared(df, col, "Rev % Change")
        data_dict["Input"].append(col)
        data_dict["Correlation"].append(corr)
        data_dict["r_squared"].append(r2)

    df = pd.DataFrame(data_dict)
    df.set_index("Input", inplace=True)

    return df


def qtd_data_for_chart(df, revenue_df, cols_of_interest):
    df_qtd_for_approaching_quarter = df[df.is_date_valid == False]
    df_qtd_for_approaching_quarter = df_qtd_for_approaching_quarter[cols_of_interest]
    prior_year_start_date = df_qtd_for_approaching_quarter.index.date[
        -1
    ] + datetime.timedelta(days=-366)
    prior_year_start_date_np = np.datetime64(str(prior_year_start_date))
    prior_year_end_date_np = np.datetime64(str(revenue_df.index.date[3]))

    df_prior_quarter_for_tracking = df[
        (df.index >= prior_year_start_date_np) & ((df.index <= prior_year_end_date_np))
    ]
    df_prior_quarter_for_tracking = df_prior_quarter_for_tracking[cols_of_interest]
    index_dates = []
    for i in df_prior_quarter_for_tracking.index.date:
        _, month, day = str(i).split("-")
        index_dates.append(f"{month}-{day}")

    index_dates2 = []
    for i in df_qtd_for_approaching_quarter.index.date:
        _, month, day = str(i).split("-")
        index_dates2.append(f"{month}-{day}")

    for metric in cols_of_interest:
        df_qtd_for_approaching_quarter[f"{metric} CumSum"] = (
            df_qtd_for_approaching_quarter[metric].cumsum()[::-1].values.tolist()
        )
        df_prior_quarter_for_tracking[f"{metric} CumSum"] = (
            df_prior_quarter_for_tracking[metric].cumsum()[::-1].values.tolist()
        )

    df_qtd_for_approaching_quarter["Plot Index"] = index_dates2
    df_prior_quarter_for_tracking["Plot Index"] = index_dates

    return df_qtd_for_approaching_quarter, df_prior_quarter_for_tracking
