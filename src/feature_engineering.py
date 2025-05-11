# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_age_feature(df, date_column, birth_date_column):
    """
    Creates an 'age' feature from a date column and a birth date column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        date_column (str): The name of the column containing the reference date.
        birth_date_column (str): The name of the column containing the birth date.

    Returns:
        pandas.DataFrame: The DataFrame with the new 'age' column.
    """
    df_processed = df.copy() # don't modify the original
    if date_column not in df_processed.columns or birth_date_column not in df_processed.columns:
        raise ValueError(f"Columns '{date_column}' or '{birth_date_column}' not found in DataFrame.")

    try:
        df_processed[date_column] = pd.to_datetime(df_processed[date_column])
        df_processed[birth_date_column] = pd.to_datetime(df_processed[birth_date_column])
        df_processed['age'] = (df_processed[date_column] - df_processed[birth_date_column]).dt.days // 365
        print(f"Age feature created from '{date_column}' and '{birth_date_column}'.")
        return df_processed
    except Exception as e:
        print(f"Error creating age feature: {e}")
        raise

def create_interaction_feature(df, col1, col2, new_col_name):
    """
    Creates an interaction feature by multiplying two columns.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        new_col_name (str): The name of the new interaction column.

    Returns:
        pandas.DataFrame: The DataFrame with the new interaction column.
    """
    df_processed = df.copy()
    if col1 not in df_processed.columns or col2 not in df_processed.columns:
        raise ValueError(f"Columns '{col1}' or '{col2}' not found in DataFrame.")

    try:
        df_processed[new_col_name] = df_processed[col1] * df_processed[col2]
        print(f"Interaction feature '{new_col_name}' created from '{col1}' and '{col2}'.")
        return df_processed
    except Exception as e:
        print(f"Error creating interaction feature: {e}")
        raise



def apply_log_transformation(df, column):
    """
    Applies a log transformation to a specified column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the column to transform.

    Returns:
        pandas.DataFrame: The DataFrame with the log-transformed column.
    """
    df_processed = df.copy()
    if column not in df_processed.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    if (df_processed[column] <= 0).any():
        print(f"Warning: Column '{column}' contains non-positive values.  Log transformation will add 1 before transforming.")
        df_processed[column] = df_processed[column] + 1

    try:
        df_processed[column] = np.log(df_processed[column])
        print(f"Log transformation applied to column '{column}'.")
        return df_processed
    except Exception as e:
        print(f"Error applying log transformation to column '{column}': {e}")
        raise



def encode_categorical_variable(df, column, encoding_type='one-hot'):
    """
    Encodes a categorical variable using one-hot or label encoding.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The name of the column to encode.
        encoding_type (str): The type of encoding ('one-hot' or 'label').

    Returns:
        pandas.DataFrame: The DataFrame with the encoded column(s).
    """
    df_processed = df.copy()
    if column not in df_processed.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if encoding_type == 'one-hot':
        try:
            encoded_df = pd.get_dummies(df_processed, columns=[column], prefix=[column])
            print(f"One-hot encoding applied to column '{column}'.")
            return encoded_df
        except Exception as e:
            print(f"Error applying one-hot encoding to column '{column}': {e}")
            raise
    elif encoding_type == 'label':
        try:
            le = LabelEncoder()
            df_processed[column + '_encoded'] = le.fit_transform(df_processed[column])
            df_processed.drop(columns=[column], inplace=True)
            print(f"Label encoding applied to column '{column}'.")
            return df_processed
        except Exception as e:
            print(f"Error applying label encoding to column '{column}': {e}")
            raise
    else:
        raise ValueError(f"Invalid encoding type '{encoding_type}'.")



if __name__ == "__main__":
    # Example Usage
    data = pd.DataFrame({
        'date': ['2023-01-15', '2023-02-20', '2023-03-25', '2023-04-30', '2023-05-10'],
        'birth_date': ['1990-05-01', '1985-10-15', '2000-02-01', '1995-07-22', '1988-12-05'],
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 0, 200, 300, 5],
        'category': ['x', 'y', 'x', 'z', 'y'],
        'label': ['a', 'b', 'a', 'c', 'b']
    })

    print("Original Data:")
    print(data)

    # Create age feature
    data_processed = create_age_feature(data, 'date', 'birth_date')
    print("\nData after creating age feature:")
    print(data_processed)

    # Create interaction feature
    data_processed = create_interaction_feature(data_processed, 'A', 'B', 'A_times_B')
    print("\nData after creating interaction feature:")
    print(data_processed)

    # Apply log transformation
    data_processed = apply_log_transformation(data_processed, 'C')
    print("\nData after log transformation:")
    print(data_processed)

    # Encode categorical variable
    data_processed = encode_categorical_variable(data_processed, 'category', encoding_type='one-hot')
    print("\nData after one-hot encoding:")
    print(data_processed)

    data_processed_label_encoded = encode_categorical_variable(data, 'label', encoding_type='label')
    print("\nData after label encoding:")
    print(data_processed_label_encoded)
