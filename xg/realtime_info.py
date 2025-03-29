import json

def get_last_dates(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    print("📅 Last available date per stock:\n")
    for stock, groups in data.items():
        if not groups or not groups[0]['data']:
            print(f"{stock}: ⚠️ No data found")
            continue

        last_entry = groups[0]['data'][-1]
        last_date = last_entry.get('date', 'Unknown')
        print(f"{stock}: {last_date}")

if __name__ == "__main__":
    json_path = 'json/trading_database_export.json'  # 🔁 Replace this with your actual path
    get_last_dates(json_path)
