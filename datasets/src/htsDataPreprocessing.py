import pandas as pd
import re
from pathlib import Path
from datetime import datetime, timedelta


def list_hts_files(directory_path):
    # Create a Path object for the specified directory
    dir_path = Path(directory_path)

    extension = ".hts"

    # Return a list of filenames with the specified extension (only files)
    return [
        file.name
        for file in dir_path.iterdir()
        if file.is_file() and file.suffix == extension
    ]


# getting date based on date id and delta
def get_date(startDate, startDateId, currentDateId):
    dateDelta = int(currentDateId) - int(startDateId)

    return startDate + timedelta(days=dateDelta)


# extract start and end date from headers
def extract_dates(data):
    # Regex to find dates in the format YYYY MM DD
    date_pattern = r"(\d{4}) (\d{2}) (\d{2})"
    matches = re.findall(date_pattern, data)

    # Convert matched dates to datetime objects
    dates = [datetime.strptime(" ".join(match), "%Y %m %d") for match in matches]

    return dates


def seconds_to_timestamp(seconds):
    from datetime import timedelta

    return str(timedelta(seconds=seconds))


# Function to update or add an entry into a df
def update_or_add_entry(df, params, new_data):
    mask = pd.Series([True] * len(df))
    for key, value in params.items():
        mask &= df[key] == value
    if mask.any():
        df.loc[mask, new_data.keys()] = list(
            new_data.values()
        )  # Convert values to list
    else:
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    return df


def process_hts_headers(file_path):

    startDate = None
    endDate = None
    startDateId = None

    with open(file_path, "r") as file:
        for line in file:
            # parses hts headers for start and end date
            if line.startswith("H2"):
                startDate, endDate = extract_dates(line)
            # parses  for the first day start id
            if not line.startswith("H") and not line.startswith("99"):
                parts = line.split()
                if float(parts[3]) == 0:
                    startDateId = parts[2]
                    break

    return startDate, endDate, startDateId


def process_hts_body(file_path, df, startDate, startDateId):
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith("H") and not line.startswith("99"):
                parts = line.split()
                date = get_date(startDate, startDateId, parts[2])
                time = seconds_to_timestamp(float(parts[3]))
                new_data = {
                    "date": date,
                    "time": time,
                    "x": parts[5],
                    "y": parts[6],
                    "z": parts[7],
                }
                params = {"date": date, "time": time}
                df = update_or_add_entry(df, params, new_data)
    return df


def read_hts_file(file_path, df):

    startDate, endDate, startDateId = process_hts_headers(file_path)

    return process_hts_body(file_path, df, startDate, startDateId)


def read_all_hts_files(dirPath):

    htsFilesInDir = list_hts_files(dirPath)

    df = pd.DataFrame(columns=["date", "time", "x", "y", "z"])

    numberOfFiles = len(htsFilesInDir)
    progressCounter = 0

    for file in htsFilesInDir:
        htsFilePath = dirPath + file
        progressCounter += 1
        print(f"{htsFilePath}___{(progressCounter/numberOfFiles)*100}%")

        df = read_hts_file(htsFilePath, df)

    return df


if __name__ == "__main__":
    # Initialize an empty DataFrame with the required columns

    # Call `read_hts` with the path to your `.hts` file and the empty DataFrame
    df = read_all_hts_files("datasets/sampleData/")
    print(df)

    # Assuming `df` is your DataFrame
    df = df.sort_values(by=["date", "time"]).reset_index(drop=True)

    # Save DataFrame `df` to a CSV file
    df.to_csv("output.csv", index=False)
