import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

from helper import (
    assign_reporting_date_to_df_date_row,
    add_is_quarter_date_valid_column,
    compute_cumsums_and_means_by_quarterly_date,
    collapse_df_to_quarterly,
    get_eod_api_data_by_ticker,
    add_growth_rates_to_quarterly_cumsums_df,
    merge_rev_columns_to,
    get_stats_df,
    qtd_data_for_chart,
)
from constants import (
    SM_METRICS,
    SM_METRIC_COLOR_GRADIENT_MAP,
    SW_METRICS,
    SW_METRIC_COLOR_GRADIENT_MAP,
    SW_METRICS_COLORS,
    SW_METRICS_COLORS_QTD,
    SM_METRICS_COLORS,
    SM_METRICS_COLORS_QTD,
)


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


def clean_summary_df_column_names(df):
    new_df = df.copy()
    columns_to_be_cleaned = [c for c in new_df.columns if "CumSum" in c]
    for column_to_clean in columns_to_be_cleaned:
        new_column_name = column_to_clean.replace(" CumSum", "")
        new_df.rename({column_to_clean: new_column_name}, axis=1, inplace=True)

    return new_df


def clean_qtd_df_column_names(df):
    new_df = df.copy()
    columns_to_be_cleaned = [c for c in new_df.columns if "CumSum" in c]
    for column_to_clean in columns_to_be_cleaned:
        new_column_name = column_to_clean.replace(" CumSum", " Cumulative")
        new_df.rename({column_to_clean: new_column_name}, axis=1, inplace=True)

    return new_df


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

        cleaned_df_summary_sm = clean_summary_df_column_names(
            df_sm_cumsums_by_quarter_with_rev
        )
        cleaned_df_summary_sw = clean_summary_df_column_names(
            df_sw_cumsums_by_quarter_with_rev
        )

        sm_stats_df = get_stats_df(df_sm_cumsums_by_quarter_with_rev)
        sw_stats_df = get_stats_df(df_sw_cumsums_by_quarter_with_rev)

        df_sm_tracking_approaching_qtd, df_sm_tracking_prior_qtd = qtd_data_for_chart(
            df_sm, revenue_df, SM_METRICS
        )
        df_sw_tracking_approaching_qtd, df_sw_tracking_prior_qtd = qtd_data_for_chart(
            df_sw, revenue_df, SW_METRICS
        )

        cleaned_qtd_df_sm = clean_qtd_df_column_names(df_sm_tracking_approaching_qtd)
        cleaned_qtd_df_sw = clean_qtd_df_column_names(df_sw_tracking_approaching_qtd)

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
                    st.dataframe(cleaned_df_summary_sm)
                with st.expander("Similar Web Table"):
                    st.dataframe(cleaned_df_summary_sw)

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
                    st.dataframe(cleaned_df_summary_sm)

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
                    st.dataframe(cleaned_df_summary_sw)

        with qtd_tab:
            data = []
            sw_approaching_label = str(df_sw_tracking_approaching_qtd.index.date[0])
            sw_prior_qtd_label = str(df_sw_tracking_prior_qtd.index.date[0])
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
                    name=f"{metric} {sw_approaching_label}",
                    line=dict(color=SW_METRICS_COLORS[i]),
                )
                data.append(trace)
                trace2 = go.Scatter(
                    x=[i for i in range(len(df_sw_tracking_prior_qtd["Plot Index"]))][
                        ::-1
                    ],
                    y=df_sw_tracking_prior_qtd[f"{metric} CumSum"],
                    mode="lines+markers",
                    name=f"{metric} {sw_prior_qtd_label}",
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
                st.dataframe(cleaned_qtd_df_sw)

            data = []
            sm_approaching_label = str(df_sm_tracking_approaching_qtd.index.date[0])
            sm_prior_qtd_label = str(df_sm_tracking_prior_qtd.index.date[0])
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
                    name=f"{metric} {sm_approaching_label}",
                    line=dict(color=SM_METRICS_COLORS[i]),
                )
                data.append(trace)
                trace2 = go.Scatter(
                    x=[i for i in range(len(df_sm_tracking_prior_qtd["Plot Index"]))][
                        ::-1
                    ],
                    y=df_sm_tracking_prior_qtd[f"{metric} CumSum"],
                    mode="lines+markers",
                    name=f"{metric} {sm_prior_qtd_label}",
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
                st.dataframe(cleaned_qtd_df_sm)


if __name__ == '__main__':
    generate_page()
