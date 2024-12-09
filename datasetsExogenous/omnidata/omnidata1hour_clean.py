import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv(
    ".\\omnidata\\downloaded_files\\omni1hour.lst",
    delim_whitespace=True,  # Handles both tabs and multiple spaces
    header=None,  # No header in the file
)

# Optionally assign column names for clarity
df.columns = ["Year", "Day", "Hour", "Kp_index", "Lyman_alpha"]

# Plot Lyman_alpha before alteration
plt.figure(figsize=(12, 6))
plt.plot(df["Lyman_alpha"], label="Lyman_alpha Before Alteration")

# Add Minute column and expand for all 60 minutes
df["Minute"] = 0
df_expanded = pd.concat([df.assign(Minute=m) for m in range(60)]).sort_values(
    ["Year", "Day", "Hour", "Minute"]
)

# Convert "Year" and "Day" to a proper date format
df_expanded["Date"] = pd.to_datetime(
    df_expanded["Year"].astype(str) + df_expanded["Day"].astype(str), format="%Y%j"
)

# Create a datetime column for interpolation
df_expanded["Datetime"] = pd.to_datetime(
    df_expanded["Date"].astype(str)
    + " "
    + df_expanded["Hour"].astype(str)
    + ":"
    + df_expanded["Minute"].astype(str),
    format="%Y-%m-%d %H:%M",
)

dfp = df_expanded.copy()


# Set the index to Datetime for interpolation
df_expanded.set_index("Datetime", inplace=True)

# Interpolate missing values for each minute
df_expanded["Kp_index"] = df_expanded["Kp_index"].interpolate(method="nearest")
df_expanded["Lyman_alpha"] = df_expanded["Lyman_alpha"].interpolate(method="nearest")

# Reset the index and drop unnecessary columns
df_expanded.reset_index(drop=True, inplace=True)
df_expanded = df_expanded[["Year", "Day", "Hour", "Minute", "Kp_index", "Lyman_alpha"]]

if False:
    plt.figure(figsize=(12, 6))
    plt.plot(dfp["Lyman_alpha"], label="Lyman_alpha Before Alteration")

    # Plot Lyman_alpha after alteration on the same plot
    plt.plot(df_expanded["Lyman_alpha"], label="Lyman_alpha After Alteration")
    plt.xlabel("Time")
    plt.ylabel("Lyman_alpha")
    plt.title("Lyman_alpha Before and After Alteration")
    plt.legend()
    plt.show()

df = df_expanded
if True:
    df["Timestamp"] = (
        pd.to_datetime(df["Year"] * 1000 + df["Day"], format="%Y%j")
        + pd.to_timedelta(df["Hour"], unit="h")
        + pd.to_timedelta(df["Minute"], unit="m")
    )

    df = df.drop(columns=["Year", "Day", "Hour", "Minute"])

    df = df[["Timestamp"] + [col for col in df.columns if col != "Timestamp"]]

df.to_csv(
    "D:\\fax\\master\\op-ml\\omnidata\\cleanedData\\cleaned1hour.csv", index=False
)
