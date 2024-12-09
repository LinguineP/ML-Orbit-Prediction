import pandas as pd

# Load data from CSV files
df1 = pd.read_csv("D:\\fax\\master\\op-ml\\omnidata\\cleanedData\\cleaned1hour.csv")
df2 = pd.read_csv("D:\\fax\\master\\op-ml\\omnidata\\cleanedData\\cleaned1min.csv")

# Convert Timestamp to datetime
df1["Timestamp"] = pd.to_datetime(df1["Timestamp"])
df2["Timestamp"] = pd.to_datetime(df2["Timestamp"])

# Merge the dataframes on Timestamp
merged_df = pd.merge(df2, df1, on="Timestamp", how="inner")

# Display the merged dataframe
print(merged_df)

# Optionally, save the merged dataframe to a new CSV file
merged_df.to_csv(
    "D:\\fax\\master\\op-ml\\omnidata\\omniComplete\\omniComplete.csv", index=False
)
