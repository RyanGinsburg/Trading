
from datetime import date, timedelta

stocks = ["AAPL", "AMD", "GOOGL", "META", "MSFT", "NVDA", "SPY", "TSLA"]
today_str = date.today()
yesterday_str = today_str - timedelta(days=1)
yesterday_str = yesterday_str.strftime("%Y-%m-%d")
last_known_dates = {ticker: yesterday_str for ticker in stocks}

print(last_known_dates)