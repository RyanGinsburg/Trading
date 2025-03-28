from datetime import datetime, timedelta
import pandas_market_calendars as mcal  # type: ignore

def get_market_days(start_year, end_year, exchange='NYSE'):
    market_calendar = mcal.get_calendar(exchange)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    schedule = market_calendar.schedule(start_date=start_date, end_date=end_date)
    return schedule.index.to_list()

def dates():
    dates = [
        datetime(2025, 2, 11),
        # datetime(2024, 1, 9),
        # datetime(2024, 5, 2),
    ]
    result_dates = []
    for start in dates:
        future_dates = []
        current_date = start
        for _ in range(future_days + 1):
            while current_date not in market_days:
                current_date += timedelta(days=1)
            future_dates.append(current_date)
            current_date += timedelta(days=1)
        result_dates.append(future_dates)
    return result_dates

# Parameters
future_days = 30
market_days = get_market_days(2015, 2026)

# Run and print result
if __name__ == "__main__":
    result = dates()
    for i, date_list in enumerate(result):
        print(f"\nMarket days starting from {date_list[0].date()}:")
        for d in date_list:
            print(d.date())
