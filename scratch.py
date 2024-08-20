import requests
import pandas as pd

from sec_edgar_api import EdgarClient
from sec_cik_mapper import StockMapper

# Initialize a stock mapper instance
def get_cik_from_ticker(ticker):
    mapper = StockMapper()
    cik = mapper.ticker_to_cik[ticker.upper()]
    return cik

def get_earnings_dates_from_edgar_api(ticker):
    cik = get_cik_from_ticker(ticker)
    edgar = EdgarClient(user_agent="<msf122133@gmail.com")
    res = edgar.get_submissions(cik=cik)
    # print(f"res: \n{list(res.keys())}")

    # print(res["filings"].keys())
    # print(res["filings"]["recent"])
    recent_filings = res["filings"]["recent"]["primaryDocDescription"] #["primaryDocument"]
    print(res["filings"])
    recent_report_dates = res["filings"]["recent"]["reportDate"]
    relevant_indexes_for_report_dates = []
    for i, recent_filing in enumerate(recent_filings):
        recent_filing_lower_case = recent_filing.lower()
        if recent_filing_lower_case == "10-q" or recent_filing_lower_case == "10-k":
            relevant_indexes_for_report_dates.append(i)

    relevant_dates = []
    for j, report_date in enumerate(recent_report_dates):
        if j in relevant_indexes_for_report_dates:
            relevant_dates.append(report_date)

    report_dates = pd.DataFrame({})
    report_dates["report_dates"] = pd.to_datetime(relevant_dates)
    report_dates["dummy_column"] = [i for i in range(len(relevant_dates))] 
    report_dates.set_index("report_dates", inplace=True)
    report_dates.sort_index(ascending=True)

    return report_dates


ticker  = "AAPL"
dates = get_earnings_dates_from_edgar_api(ticker)
# dates = [str(d) for d in dates]
with open("/home/ubuntu/predictive-sales-metrics/test_dates_apple.txt", "w") as f:
    for d in list(dates.index.values):
        f.write(str(d)+"\n")
print(f"dates: {dates}")

