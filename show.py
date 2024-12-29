import sqlite3
import pandas as pd
import json

def export_database_to_single_json(db_name, output_file="database_export.json"):
    """
    Exports the contents of all tables in the database to a single JSON file.
    
    :param db_name: Name of the database file
    :param output_file: Name of the output JSON file
    """
    # Connect to the database
    conn = sqlite3.connect(db_name)

    # Get all table names
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)

    if tables.empty:
        print("No tables found in the database.")
        conn.close()
        return

    # Dictionary to store all tables' data
    database_data = {}

    # Loop through tables and collect content
    for table_name in tables['name']:
        # Fetch table content
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            # Convert DataFrame to a list of dictionaries and store it
            database_data[table_name] = df.to_dict(orient="records")
        else:
            print(f"Table '{table_name}' is empty and will not be included in the JSON file.")

    # Write the collected data to a JSON file
    with open(output_file, "w") as json_file:
        json.dump(database_data, json_file, indent=4)

    print(f"Exported all tables to {output_file}.")

    # Close the connection
    conn.close()

# Example usage
database_name = "trading_algo.db"
export_database_to_single_json(database_name)
