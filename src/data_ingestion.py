# src/data_ingestion.py
import pandas as pd
import sqlite3

def load_data_from_sqlite(db_path, table_name):
    """
    Loads data from a SQLite database into a Pandas DataFrame.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to read from.

    Returns:
        pandas.DataFrame: The loaded data, or None if an error occurs.
    """
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(db_path)

        # Construct the SQL query
        query = f"SELECT * FROM {table_name};"

        # Read the data into a Pandas DataFrame
        df = pd.read_sql_query(query, conn)

        print(f"Data loaded successfully from table '{table_name}' in database '{db_path}'")
        return df

    except sqlite3.Error as e:
        print(f"Error loading data from SQLite: {e}")
        print(f"An error occurred: {e}")  # Print the error message
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    finally:
        # Ensure the connection is closed, even if errors occur
        if conn:
            conn.close()



if __name__ == "__main__":
    # Example usage:
    db_path = 'bmarket.db'  # Replace with the actual path to your database file
    table_name = 'bank_marketing'  # Replace with the name of the table you want to read
    
    df = load_data_from_sqlite(db_path, table_name)

    if df is not None:
        # Print the first few rows of the DataFrame to verify the data
        print("First 5 rows of the data:")
        print(df.head())

        # Print the shape of the DataFrame (number of rows and columns)
        print("\nShape of the data:")
        print(df.shape)

        # Print the column names and their data types
        print("\nColumn information:")
        print(df.info())

