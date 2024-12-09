import pandas as pd


def split_csv(file_path, train_percent=70, val_percent=15, test_percent=15):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Check if the percentages sum to 100
    total_percent = train_percent + val_percent + test_percent
    if total_percent != 100:
        raise ValueError("The percentages must sum to 100.")

    # Calculate the number of rows for each split
    total_rows = len(df)
    train_size = int(total_rows * train_percent / 100)
    val_size = int(total_rows * val_percent / 100)
    test_size = total_rows - train_size - val_size  # Remainder goes to test set

    # Split the DataFrame into train, validation, and test sets sequentially
    train_data = df[:train_size]
    val_data = df[train_size : train_size + val_size]
    test_data = df[train_size + val_size :]

    # Return the splits
    return train_data, val_data, test_data


# Example usage
#sort_csv_chronologically("output.csv")
