# src/data_processing.py
import pandas as pd
import numpy as np

def handle_missing_values(df, strategy=None, columns=None):
    """
    Handles missing values in a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        strategy (str): The strategy for handling missing values ('mean', 'median', 'drop', 'ffill').
            If None, no action is taken.
        columns (list, optional):  List of columns to apply the strategy to.
            If None, apply to all columns with missing values.

    Returns:
        pandas.DataFrame: The DataFrame with missing values handled.
    """
    if strategy is None:
        return df

    df_processed = df.copy() # Create a copy to avoid modifying original dataframe.
    
    if columns is None:
        columns_with_missing = df_processed.columns[df_processed.isnull().any()].tolist()
    else:
        columns_with_missing = [col for col in columns if col in df_processed.columns and df_processed[col].isnull().any()] # ensure the cols exist

    if not columns_with_missing:
        print("No missing values to handle in the specified columns.")
        return df_processed

    for col in columns_with_missing:
        if strategy == 'mean':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            print(f"Missing values in column '{col}' filled with mean.")
        elif strategy == 'median':
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            print(f"Missing values in column '{col}' filled with median.")
        elif strategy == 'drop':
             df_processed.dropna(subset=[col], inplace=True) # drop rows where there are missing values in specified column
             print(f"Rows with missing values in column '{col}' dropped.")
        elif strategy == 'ffill':
            df_processed[col] = df_processed[col].ffill()
            print(f"Missing values in column '{col}' filled with forward fill.")
        else:
            print(f"Warning: Invalid missing value strategy '{strategy}'. No action taken for column '{col}'.")
            # No action, retain original missing values

    return df_processed


def convert_data_types(df, data_types):
    """
    Converts the data types of columns in a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        data_types (dict): A dictionary where keys are column names and values are the
            desired data types (e.g., {'col1': 'int64', 'col2': 'category'}).

    Returns:
        pandas.DataFrame: The DataFrame with converted data types.
    """
    df_processed = df.copy()
    for col, data_type in data_types.items():
        if col in df_processed.columns:
            try:
                df_processed[col] = df_processed[col].astype(data_type)
                print(f"Column '{col}' converted to type '{data_type}'.")
            except ValueError:
                print(f"Error: Cannot convert column '{col}' to type '{data_type}'. Skipping.")
        else:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping type conversion.")
    return df_processed



def remove_columns(df, columns):
    """
    Removes specified columns from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): A list of column names to remove.

    Returns:
        pandas.DataFrame: The DataFrame with the specified columns removed.
    """
    df_processed = df.copy()
    columns_to_drop = [col for col in columns if col in df_processed.columns] # only drop columns that exist
    if not columns_to_drop:
        print("No columns to remove.")
        return df_processed
    df_processed = df_processed.drop(columns=columns_to_drop)
    print(f"Columns '{columns_to_drop}' removed.")
    return df_processed



if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [6, 7, 8, 9, 10],
        'C': [11.1, 12.2, 13.3, np.nan, 15.5],
        'D': ['a', 'b', 'c', 'd', 'e'],
        'E': [True, False, True, False, True]
    })

    print("Original Data:")
    print(data)

    # Handle missing values
    data_processed = handle_missing_values(data, strategy='median', columns=['A', 'C'])
    print("\nData after handling missing values:")
    print(data_processed)

    # Convert data types
    data_processed = convert_data_types(data_processed, data_types={'A': 'float64', 'B': 'int64', 'D': 'category', 'E': 'bool'})
    print("\nData after converting data types:")
    print(data_processed)

    # Remove columns
    data_processed = remove_columns(data_processed, columns=['B', 'D'])
    print("\nData after removing columns:")
    print(data_processed)
