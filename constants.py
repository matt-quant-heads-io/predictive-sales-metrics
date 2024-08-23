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
EODHD_URL_FOR_FINANCIALS = (
    "https://eodhd.com/api/fundamentals/{ticker}.US?api_token={api_token}&fmt=json"
)