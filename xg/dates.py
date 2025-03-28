from datetime import datetime
import pandas_market_calendars as mcal  # type: ignore

def get_market_days(start_year, end_year, exchange='NYSE'):
    market_calendar = mcal.get_calendar(exchange)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    schedule = market_calendar.schedule(start_date=start_date, end_date=end_date)
    return schedule.index.to_list()

def get_start_date_for_ending_today(market_days, future_days, end_date):
    if end_date not in market_days:
        raise ValueError(f"{end_date.date()} is not a market day.")
    end_index = market_days.index(end_date)
    if end_index < future_days:
        raise ValueError("Not enough market days before the end date.")
    return market_days[end_index - future_days]

# Parameters
future_days = 30
today = datetime(2025, 3, 26)
market_days = get_market_days(2015, 2026)

# Get start date
today = today.replace(hour=0, minute=0, second=0, microsecond=0)
try:
    start_date = get_start_date_for_ending_today(market_days, future_days, today)
    print(f"To get {future_days} market days ending on {today.date()}, start on {start_date.date()}")
except ValueError as e:
    print(f"Error: {e}")
