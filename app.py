import sys
import datetime

import pandas as pd
from urllib.error import URLError
import plotly.graph_objects as go
import streamviz
import streamlit as st
import numpy as np

from helper import (
    get_cik_from_ticker,
    get_earnings_dates_from_edgar_api,
    get_start_date_from_sm_file,
    determine_quarter,
    get_eod_api_data_by_ticker,
    r_and_rsquared,
)


METRICS = [
    "Observed Sales",
    "Observed Transactions",
    "Observed Customers",
    "Sales per Customer",
    "Average Transaction Value",
    "Transactions per Customer",
]
SM_METRICS = ["Observed Sales", "Observed Transactions", "Observed Customers"]
SM_METRIC_COLOR_GRADIENT_MAP = {
    "Observed Sales": ["#0000FF", "#0041C2", "#1E90FF", "#79BAEC", "#82CAFF"],
    "Observed Transactions": ["#004225", "#005C29", "#228C22", "#32CD32", "#32CD32"],
    "Observed Customers": ["#680C07", "#900D09", "#A91B0D", "#B90E0A", "#E3242B"],
}
SW_METRICS = ["Visits", "Unique Visitors", "Total Page Views"]
SW_METRIC_COLOR_GRADIENT_MAP = {
    "Visits": ["#290916", "#710193", "#AF69EF", "#9867C5", "#BE93D4"],
    "Unique Visitors": ["#E11584", "#FD5DA8", "#F699CD", "#FDA4BA", "#FEC5E5"],
    "Total Page Views": ["#004225", "#005C29", "#228C22", "#32CD32", "#32CD32"],
}
SW_METRICS_COLORS = ["blue", "green", "firebrick", "orange", "purple"]
SW_METRICS_COLORS_QTD = ["lightblue", "lightgreen", "tomato"]
SM_METRICS_COLORS = ["blue", "green", "firebrick", "orange", "purple"]
SM_METRICS_COLORS_QTD = ["lightblue", "lightgreen", "tomato"]
CUMSUM_METRICS = [
    "Observed Sales",
    "Observed Transactions",
    "Observed Customers",
    "Visits",
    "Unique Visitors",
    "Total Page Views",
]
MEAN_METRICS = [
    "Sales per Customer",
    "Average Transaction Value",
    "Transactions per Customer",
]
SIMILAR_WEB_API_KEY = "7d015188e27e4ee1867f64a18213cbf5"


df = None
ticker = None
revenue_dict = None
normalized_eod_df = None
final_df = None
company_website = None
metric_to_df_map = {}
final_similar_web_df = None
sm_uploaded_file = None
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
ticker = None


st.set_page_config(page_title="Backtester", page_icon="📊", layout="wide")
st.sidebar.markdown("### Backtest Parameters")


def preprocess_df(df, df_type=None):
    if df_type is None:
        raise ValueError("df_type is either 'sm' or 'sw'")

    df.rename({df.columns[0]: "Date"}, axis=1, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(ascending=False, inplace=True)
    return df[SM_METRICS] if df_type == "sm" else df[SW_METRICS]


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


def generate_page():
    with st.sidebar.form("my_form"):
        sm_uploaded_file = st.file_uploader(
            "Choose a Second Measure file", type=["csv"]
        )
        sw_uploaded_file = st.file_uploader("Choose a SimilarWeb file", type=["xlsx"])
        ticker = st.text_input("ticker")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if sm_uploaded_file is not None:
            sm_uploaded_file.seek(0)
            df_sm = pd.read_csv(sm_uploaded_file)
            df_sm = preprocess_df(df_sm, df_type="sm")

        if sw_uploaded_file is not None:
            df_sw = pd.read_excel(sw_uploaded_file, sheet_name="Total")
            df_sw = preprocess_df(df_sw, df_type="sw")

        if sw_uploaded_file is None or sm_uploaded_file is None:
            raise RuntimeError("Upload a file for both SimilarWeb and Second Measure")

        revenue_df = get_eod_api_data_by_ticker(ticker, df_sm)

        df_sm = assign_reporting_date_to_df_date_row(df_sm, revenue_df)
        df_sw = assign_reporting_date_to_df_date_row(df_sw, revenue_df)

        df_sm = add_is_quarter_date_valid_column(df_sm, revenue_df)
        df_sw = add_is_quarter_date_valid_column(df_sw, revenue_df)

        df_sm = compute_cumsums_and_means_by_quarterly_date(df_sm, SM_METRICS)
        df_sw = compute_cumsums_and_means_by_quarterly_date(df_sw, SW_METRICS)

        df_sm_cumsums_by_quarter = collapse_df_to_quarterly(df_sm)
        df_sw_cumsums_by_quarter = collapse_df_to_quarterly(df_sw)

        df_sm_cumsums_by_quarter = add_growth_rates_to_quarterly_cumsums_df(
            df_sm_cumsums_by_quarter
        )
        df_sw_cumsums_by_quarter = add_growth_rates_to_quarterly_cumsums_df(
            df_sw_cumsums_by_quarter
        )

        df_sm_cumsums_by_quarter_with_rev = merge_rev_columns_to(
            df_sm_cumsums_by_quarter, revenue_df
        )
        df_sw_cumsums_by_quarter_with_rev = merge_rev_columns_to(
            df_sw_cumsums_by_quarter, revenue_df
        )

        sm_stats_df = get_stats_df(df_sm_cumsums_by_quarter_with_rev)
        sw_stats_df = get_stats_df(df_sw_cumsums_by_quarter_with_rev)

        df_sm_tracking_approaching_qtd, df_sm_tracking_prior_qtd = qtd_data_for_chart(
            df_sm, revenue_df, SM_METRICS
        )
        df_sw_tracking_approaching_qtd, df_sw_tracking_prior_qtd = qtd_data_for_chart(
            df_sw, revenue_df, SW_METRICS
        )

        summary_tab, qtd_tab = st.tabs(["Summary", "QTD"])
        with summary_tab:
            st.markdown("### Correlations")
            summary_row_sw_corr = st.columns(3)
            summary_row_sw_r2 = st.columns(3)
            summary_row_sm_corr = st.columns(3)
            summary_row_sm_r2 = st.columns(3)
            idx = 0
            for col in (
                summary_row_sw_corr
                + summary_row_sw_r2
                + summary_row_sm_corr
                + summary_row_sm_r2
            ):
                tile = col.container(height=120)
                if idx == 0:
                    tile.metric(
                        "Visits % Change ${r^2}$",
                        round(
                            sw_stats_df[
                                (sw_stats_df.index == "Visits CumSum growth (%)")
                            ]["r_squared"],
                            4,
                        ),
                    )
                elif idx == 1:
                    tile.metric(
                        "Unique Visitors % Change ${r^2}$",
                        round(
                            sw_stats_df[
                                (
                                    sw_stats_df.index
                                    == "Unique Visitors CumSum growth (%)"
                                )
                            ]["r_squared"],
                            4,
                        ),
                    )
                elif idx == 2:
                    tile.metric(
                        "Total Page Views % Change ${r^2}$",
                        round(
                            sw_stats_df[
                                (
                                    sw_stats_df.index
                                    == "Total Page Views CumSum growth (%)"
                                )
                            ]["r_squared"],
                            4,
                        ),
                    )
                elif idx == 3:
                    tile.metric(
                        "Visits % Change Corr.",
                        round(
                            sw_stats_df[
                                (sw_stats_df.index == "Visits CumSum growth (%)")
                            ]["Correlation"],
                            4,
                        ),
                    )
                elif idx == 4:
                    tile.metric(
                        "Unique Visitors % Change Corr.",
                        round(
                            sw_stats_df[
                                (
                                    sw_stats_df.index
                                    == "Unique Visitors CumSum growth (%)"
                                )
                            ]["Correlation"],
                            4,
                        ),
                    )
                elif idx == 5:
                    tile.metric(
                        "Total Page Views % Change Corr.",
                        round(
                            sw_stats_df[
                                (
                                    sw_stats_df.index
                                    == "Total Page Views CumSum growth (%)"
                                )
                            ]["Correlation"],
                            4,
                        ),
                    )
                elif idx == 6:
                    tile.metric(
                        "Obs. Sales % Change ${r^2}$",
                        round(
                            sm_stats_df[
                                (
                                    sm_stats_df.index
                                    == "Observed Sales CumSum growth (%)"
                                )
                            ]["r_squared"],
                            4,
                        ),
                    )
                elif idx == 7:
                    tile.metric(
                        "Obs. Txns % Change ${r^2}$",
                        round(
                            sm_stats_df[
                                (
                                    sm_stats_df.index
                                    == "Observed Transactions CumSum growth (%)"
                                )
                            ]["r_squared"],
                            4,
                        ),
                    )
                elif idx == 8:
                    tile.metric(
                        "Obs. Custs. % Change ${r^2}$",
                        round(
                            sm_stats_df[
                                (
                                    sm_stats_df.index
                                    == "Observed Customers CumSum growth (%)"
                                )
                            ]["r_squared"],
                            4,
                        ),
                    )
                elif idx == 9:
                    tile.metric(
                        "Obs. Sales % Change Corr.",
                        round(
                            sm_stats_df[
                                (
                                    sm_stats_df.index
                                    == "Observed Sales CumSum growth (%)"
                                )
                            ]["Correlation"],
                            4,
                        ),
                    )
                elif idx == 10:
                    tile.metric(
                        "Obs. Txns Visitors % Change Corr.",
                        round(
                            sm_stats_df[
                                (
                                    sm_stats_df.index
                                    == "Observed Transactions CumSum growth (%)"
                                )
                            ]["Correlation"],
                            4,
                        ),
                    )
                elif idx == 11:
                    tile.metric(
                        "Obs. Custs. % Change Corr.",
                        round(
                            sm_stats_df[
                                (
                                    sm_stats_df.index
                                    == "Observed Customers CumSum growth (%)"
                                )
                            ]["Correlation"],
                            4,
                        ),
                    )

                idx += 1

            st.divider()
            growth_tab, absolute_tab = st.tabs(["Growth", "Absolute"])
            with growth_tab:
                data = []
                for i, metric in enumerate(SW_METRICS):
                    trace = go.Scatter(
                        x=df_sw_cumsums_by_quarter_with_rev.index,
                        y=df_sw_cumsums_by_quarter_with_rev[
                            f"{metric} CumSum growth (%)"
                        ],
                        mode="lines+markers",
                        name=f"{metric}",
                        line=dict(color=SW_METRIC_COLOR_GRADIENT_MAP[metric][i]),
                    )
                    data.append(trace)

                for i, metric in enumerate(SM_METRICS):
                    trace = go.Scatter(
                        x=df_sw_cumsums_by_quarter_with_rev.index,
                        y=df_sm_cumsums_by_quarter_with_rev[
                            f"{metric} CumSum growth (%)"
                        ],
                        mode="lines+markers",
                        name=f"{metric}",
                        line=dict(color=SM_METRIC_COLOR_GRADIENT_MAP[metric][i]),
                    )
                    data.append(trace)

                trace = go.Scatter(
                    x=df_sw_cumsums_by_quarter_with_rev.index,
                    y=df_sm_cumsums_by_quarter_with_rev[f"Rev % Change"],
                    mode="lines+markers",
                    name=f"Revenue Growth",
                    line=dict(color="lightblue"),
                )
                data.append(trace)

                layout = go.Layout(
                    title="Summary",
                    xaxis=dict(title="Quarterly Dates"),
                    yaxis=dict(title="Y"),
                    hovermode="closest",
                )
                fig = go.Figure(data=data, layout=layout)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with st.expander("Second Measure Table"):
                    st.dataframe(df_sm_cumsums_by_quarter_with_rev)
                with st.expander("Similar Web Table"):
                    st.dataframe(df_sw_cumsums_by_quarter_with_rev)

            idx = 0
            with absolute_tab:
                data = []
                for i, metric in enumerate(SM_METRICS):
                    trace = go.Scatter(
                        x=df_sw_cumsums_by_quarter_with_rev.index,
                        y=df_sm_cumsums_by_quarter_with_rev[f"{metric} CumSum"],
                        mode="lines+markers",
                        name=f"{metric}",
                        line=dict(color=SM_METRIC_COLOR_GRADIENT_MAP[metric][i]),
                    )
                    data.append(trace)

                layout = go.Layout(
                    title="Absolute - Second Measure",
                    xaxis=dict(title="Quarterly Dates"),
                    yaxis=dict(title="Y"),
                    hovermode="closest",
                )
                fig = go.Figure(data=data, layout=layout)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with st.expander("Second Measure Table"):
                    st.dataframe(df_sm_cumsums_by_quarter_with_rev)

                data = []
                for i, metric in enumerate(SW_METRICS):
                    trace = go.Scatter(
                        x=df_sw_cumsums_by_quarter_with_rev.index,
                        y=df_sw_cumsums_by_quarter_with_rev[f"{metric} CumSum"],
                        mode="lines+markers",
                        name=f"{metric}",
                        line=dict(color=SW_METRIC_COLOR_GRADIENT_MAP[metric][i]),
                    )
                    data.append(trace)

                layout = go.Layout(
                    title="Absolute - Similar Web",
                    xaxis=dict(title="Quarterly Dates"),
                    yaxis=dict(title="Y"),
                    hovermode="closest",
                )
                fig = go.Figure(data=data, layout=layout)
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with st.expander("Similar Web Table"):
                    st.dataframe(df_sw_cumsums_by_quarter_with_rev)

        with qtd_tab:
            data = []
            for i, metric in enumerate(SW_METRICS):
                trace = go.Scatter(
                    x=[
                        i
                        for i in range(
                            len(df_sw_tracking_approaching_qtd["Plot Index"])
                        )
                    ][::-1],
                    y=df_sw_tracking_approaching_qtd[f"{metric} CumSum"],
                    mode="lines+markers",
                    name=f"{metric}",
                    line=dict(color=SW_METRICS_COLORS[i]),
                )
                data.append(trace)
                trace2 = go.Scatter(
                    x=[i for i in range(len(df_sw_tracking_prior_qtd["Plot Index"]))][
                        ::-1
                    ],
                    y=df_sw_tracking_prior_qtd[f"{metric} CumSum"],
                    mode="lines+markers",
                    name=f"{metric}",
                    line=dict(color=SW_METRICS_COLORS_QTD[i]),
                )
                data.append(trace2)

            layout = go.Layout(
                title="Similar Web",
                xaxis=dict(title="QTD"),
                yaxis=dict(title="Y"),
                hovermode="closest",
            )
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            with st.expander("Similar Web Table"):
                st.dataframe(df_sw_tracking_approaching_qtd)

            data = []
            for i, metric in enumerate(SM_METRICS):
                trace = go.Scatter(
                    x=[
                        i
                        for i in range(
                            len(df_sm_tracking_approaching_qtd["Plot Index"])
                        )
                    ][::-1],
                    y=df_sm_tracking_approaching_qtd[f"{metric} CumSum"],
                    mode="lines+markers",
                    name=f"{metric}",
                    line=dict(color=SM_METRICS_COLORS[i]),
                )
                data.append(trace)
                trace2 = go.Scatter(
                    x=[i for i in range(len(df_sm_tracking_prior_qtd["Plot Index"]))][
                        ::-1
                    ],
                    y=df_sm_tracking_prior_qtd[f"{metric} CumSum"],
                    mode="lines+markers",
                    name=f"{metric}",
                    line=dict(color=SM_METRICS_COLORS_QTD[i]),
                )
                data.append(trace2)

            layout = go.Layout(
                title="Second Measure",
                xaxis=dict(title="QTD"),
                yaxis=dict(title="Y"),
                hovermode="closest",
            )
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            with st.expander("Second Measure Table"):
                st.dataframe(df_sm_tracking_approaching_qtd)


generate_page()
