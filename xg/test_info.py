import json

# Configurable
json_path = "json/trading_database_export.json"
expected_groups = 1
expected_days_per_group = 31
stocks = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMD', 'ORCL',
    'UNH', 'LLY', 'JNJ', 'MRK', 'ABBV', 'ABT', 'AMGN',
    'JPM', 'BRK.B', 'V', 'MA', 'BAC', 'WFC',
    'AMZN', 'TSLA', 'MCD', 'HD',
    'XOM', 'CVX', 'COP',
    'NFLX', 'DIS', 'CMCSA',
    'KO', 'PEP'
]

def find_resume_point(data):
    for stock in stocks:
        if stock not in data:
            print(f"Resume at stock: {stock}, group: 1, date: 0")
            return stock, 1, 0

        groups = data[stock]
        for group_index, group in enumerate(groups, start=1):
            if "data" not in group or not isinstance(group["data"], list):
                print(f"Resume at stock: {stock}, group: {group_index}, date: 0")
                return stock, group_index, 0
            if len(group["data"]) < expected_days_per_group:
                print(f"Resume at stock: {stock}, group: {group_index}, date: {len(group['data'])}")
                return stock, group_index, len(group["data"])

        if len(groups) < expected_groups:
            print(f"Resume at stock: {stock}, group: {len(groups)+1}, date: 0")
            return stock, len(groups)+1, 0

    print("All stocks fully processed.")
    return None, None, None

def main():
    with open(json_path, "r") as f:
        data = json.load(f)

    stock, group, date = find_resume_point(data)

    if stock:
        print(f"\nSet your script parameters to:\nstock_start_point = '{stock}'\ngroup = {group}\ndate = {date}")
    else:
        print("✅ All data has been processed!")

if __name__ == "__main__":
    main()
