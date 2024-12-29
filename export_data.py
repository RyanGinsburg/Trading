import sqlite3
import json
from datetime import datetime

def export_database_to_json():
    """
    Exports all information from every table in the trading_algo database to a JSON file
    """
    conn = sqlite3.connect('trading_algo.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Dictionary to store all database information
    database_contents = {}
    
    # Process each table
    for table in tables:
        table_name = table[0]
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_info = {col[1]: col[2] for col in columns}  # column name: data type
        
        # Get all data from the table
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        # Convert rows to list of dictionaries
        table_data = []
        column_names = [col[1] for col in columns]
        for row in rows:
            row_dict = {}
            for i, value in enumerate(row):
                # Convert datetime objects to strings
                if isinstance(value, datetime):
                    value = value.isoformat()
                row_dict[column_names[i]] = value
            table_data.append(row_dict)
        
        # Store table information
        database_contents[table_name] = {
            "columns": column_info,
            "data": table_data
        }
    
    conn.close()
    
    # Write to JSON file
    output_filename = 'trading_database_export.json'
    with open(output_filename, 'w') as f:
        json.dump(database_contents, f, indent=2)
    
    print(f"Database exported to {output_filename}")
    print(f"Exported {len(tables)} tables")

if __name__ == "__main__":
    export_database_to_json()