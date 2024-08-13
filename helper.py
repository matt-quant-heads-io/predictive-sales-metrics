import os

import requests
import pandas as pd 

# NOTE: uncomment this for the first time you run the app
# import pip
# pip.main(["install", "openpyxl"])

EODHD_API_TOKEN = "667849cef19004.34002384"
EODHD_URL_FOR_FINANCIALS = 'https://eodhd.com/api/fundamentals/{ticker}.US?api_token={api_token}&fmt=json' 
SIMILAR_WEB_API_KEY = ""
#TODO: change this to get the path automatically
PATH_TO_INPUT_DATA = "/Users/matt/predictive-sales-metrics/input_data"


def get_eod_api_data_by_ticker(ticker, num_quarters_to_get_financials_for):
    try:
        req_url = EODHD_URL_FOR_FINANCIALS.format(ticker=ticker, api_token=EODHD_API_TOKEN)
        json_data = requests.get(req_url).json()
        
        web_url = json_data["General"]["WebURL"]
        data = json_data["Financials"]
        filing_dates = set()
        rev_data =[]
        for idx, (date, data_dict) in enumerate(data["Income_Statement"]["quarterly"].items()):
            # print(f"data_dict: {data_dict}")
            if date not in filing_dates:
                filing_dates.add(date)
                rev_data.append(data_dict["totalRevenue"])

        filing_dates = list(filing_dates)
        filing_dates.sort(reverse=True)

        rev_data = list(rev_data)
        rev_data.sort(reverse=True)
        df = pd.DataFrame({"Month": filing_dates[:num_quarters_to_get_financials_for], "Revenue": rev_data[:num_quarters_to_get_financials_for]})
        df["Month"] = pd.to_datetime(df["Month"])
        df.set_index("Month", inplace=True)
        # print(f"df: {df.head()}")
        return df, web_url

        
        
    except Exception as e:
        print(f"Error: {e}")
        return {}


def get_similarweb_traffic_stats(domain, api_key, start_date, end_date, granularity="monthly"):
    url = "https://api.similarweb.com/batch/v4/request-report"

    payload = {
        "metrics": ["all_traffic_visits", "global_rank", "desktop_new_visitors", "mobile_average_visit_duration"],
        "filters": {
            "domains": [domain],
            "countries": ["WW"],
            "include_subdomains": True
        },
        "granularity": granularity,
        "start_date": start_date,#"2022-06",
        "end_date": end_date,#"2023-06",
        "response_format": "csv",
        "delivery_method": "download_link"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "api-key": api_key
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print(f"response.json():\n{response.json()}")
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"


# def process_second_measure_input_data():

#     for file in os.listdir(PATH_TO_INPUT_DATA):
#         if ".xlsx" not in file:
#             continue

#         ticker_to_process = file.split("_")[0]
#         print(f"path: {PATH_TO_INPUT_DATA}/{file}")
#         df = pd.read_excel(f"{PATH_TO_INPUT_DATA}/{file}")
#         print(f"df: \n{df.head()}")

def process_second_measure_input_data(uploaded_file_obj, metrics=["Observed Sales"]):
    ticker = uploaded_file_obj.name.split("_")[0]
    if ".xlsx" in uploaded_file_obj.name or ".xls" in uploaded_file_obj.name:
        df = pd.read_excel(uploaded_file_obj)
    elif ".csv" in uploaded_file_obj.name:
        df = pd.read_csv(uploaded_file_obj)

    # df = df[["Month","entity_id","Name","Brand ID","Observed Sales","Observed Transactions","Observed Customers","Sales per Customer","Average Transaction Value","Transactions per Customer","Share of Sales","Share of Transactions"]]
    df = df[["Month", "Name","Brand ID"] + metrics]

    # Take unique list of brand ID
    brand_ids = list(df["Brand ID"].unique())
    dates = list(df["Month"].unique())
    """
    Observed Sales
    Observed Transactions
    Observed Customers
    Sales per Customer
    Average Transaction Value
    Transactions per Customer
    Share of Sales
    Share of Transactions
    """

    merged_data_dict = {"Month": dates}
    for brand_id in brand_ids:
        df_by_brand = df[df["Brand ID"] == brand_id]
        if brand_id not in merged_data_dict:
            merged_data_dict[brand_id] = []
        for metric_val in df_by_brand[metrics[0]].values:
            merged_data_dict[brand_id].append(metric_val)

        print(f"df_by_brand: {df_by_brand}")
    print(f"brand_ids: {brand_ids}")

    # Use list to buld out dataframe where the column are the brands (i.e. the unique ids)

        # Observed Sales 
        # Merge all of the data on Month column 
    for k, v in merged_data_dict.items():
        print(f"{k}={len(v)}")
    df_metrics_by_brand = pd.DataFrame(merged_data_dict)
    df_metrics_by_brand["Month"] = pd.to_datetime(df_metrics_by_brand["Month"])

    df_metrics_by_brand.set_index("Month", inplace=True)

    return df_metrics_by_brand, ticker

    
    
    for file in os.listdir(PATH_TO_INPUT_DATA):
        if ".xlsx" not in file:
            continue

        ticker_to_process = file.split("_")[0]
        print(f"path: {PATH_TO_INPUT_DATA}/{file}")
        df = pd.read_excel(f"{PATH_TO_INPUT_DATA}/{file}")
        print(f"df: \n{df.head()}")






# def get_similarweb_traffic_stats(domain, api_key, start_date, end_date, granularity="monthly"):
#     url = "https://api.similarweb.com/batch/v4/request-report"
#     # endpoint = f"{base_url}/{domain}/total-traffic-and-engagement/visits"
   
#     headers = {
#         "accept": "application/json",
#         "content-type": "content-type: application/json"
#     }
   
#     params = {
#         "start_date": "2024-01",
#         "end_date": "2024-03",
#         "country": "world",
#         "granularity": "monthly"
#     }

#     data = {
#         "metrics": [
#             "all_traffic_visits",
#             "global_rank",
#             "desktop_new_visitors",
#             "mobile_average_visit_duration"
#         ],
#         "filters": {
#             "domains": [
#                 domain
#             ],
#             "countries": [
#             "WW"
#             ],
#             "include_subdomains": True
#         },
#         "granularity": granularity,
#         "start_date": start_date,#"2022-06",
#         "end_date": end_date,#"2023-06",
#         "response_format": "csv",
#         "delivery_method": "download_link"
#         }
   

#     response = requests.post(url, data=data, headers=headers, params=params)
   
#     if response.status_code == 200:
#         print(f"response.json():\n{response.json()}")
#         return response.json()
#     else:
#         return f"Error: {response.status_code}, {response.text}"







# curl --request POST \
#      --url https://api.similarweb.com/batch/v4/request-report \
#      --header 'accept: application/json' \
#      --header 'content-type: application/json' \
#      --data '
# {
#   "metrics": [
#     "all_traffic_visits",
#     "global_rank",
#     "desktop_new_visitors",
#     "mobile_average_visit_duration"
#   ],
#   "filters": {
#     "domains": [
#       "facebook.com"
#     ],
#     "countries": [
#       "WW"
#     ],
#     "include_subdomains": true
#   },
#   "granularity": "monthly",
#   "start_date": "2022-06",
#   "end_date": "2023-06",
#   "response_format": "csv",
#   "delivery_method": "download_link"
# }
# '