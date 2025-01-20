import sqlite3
import json
import os

def export_trading_algo_to_json():
    """
    Exports all information from the trading_algo database to a JSON file, 
    with unnecessary 'columns' information removed and stocks sorted alphabetically.
    """
    db_name = 'trading_algo.db'
    output_folder = 'json'
    output_filename = os.path.join(output_folder, 'trading_database_export.json')
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Dictionary to store all database information
    database_contents = {}
    
    # Process each table
    for table in tables:
        table_name = table[0]
        
        # Get all data from the table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # Convert rows to list of dictionaries
        table_data = []
        for row in rows:
            row_dict = {}
            for i, value in enumerate(row):
                row_dict[column_names[i]] = value
            table_data.append(row_dict)

        # Parse stock and date from table name (e.g., "AAPL_2023_02_06")
        stock, date = table_name.split('_', 1)
        if stock not in database_contents:
            database_contents[stock] = []
        
        # Add table data
        database_contents[stock].append({
            "date": date,
            "data": sorted(table_data, key=lambda x: x["date"])  # Sort data by date within each range
        })
    
    conn.close()
    
    # Sort stocks alphabetically and sort data by date
    sorted_database_contents = {
        stock: sorted(data_list, key=lambda x: x["date"])
        for stock, data_list in sorted(database_contents.items())
    }
    
    # Write to JSON file
    with open(output_filename, 'w') as f:
        json.dump(sorted_database_contents, f, indent=2)
    
    print(f"Trading database exported to {output_filename}")

def export_optimization_history_to_json():
    """
    Exports the optimization_history database to a JSON file organized by stock and version.
    """
    db_name = 'optimization_history.db'
    output_folder = 'json'
    output_filename = os.path.join(output_folder, 'optimization_history_export.json')
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Query all data from optimization_history table
    cursor.execute("SELECT stock, version, parameters FROM optimization_history")
    rows = cursor.fetchall()
    
    # Organize data by stock and version
    history_data = {}
    for stock, version, parameters in rows:
        if stock not in history_data:
            history_data[stock] = {}
        history_data[stock][str(version)] = json.loads(parameters)  # Convert parameters JSON string to dict

    conn.close()
    
    # Sort the data alphabetically by stock and numerically by version
    sorted_history_data = {
        stock: {k: history_data[stock][k] for k in sorted(history_data[stock], key=lambda x: int(x))}
        for stock in sorted(history_data)
    }
    
    # Write the sorted data to JSON
    with open(output_filename, 'w') as f:
        json.dump(sorted_history_data, f, indent=2)
    
    print(f"Optimization history exported to {output_filename}")

def export_all_databases():
    """
    Exports both trading_algo.db and optimization_history.db to JSON files in the json folder.
    """
    export_trading_algo_to_json()
    export_optimization_history_to_json()

if __name__ == "__main__":
    export_all_databases()
