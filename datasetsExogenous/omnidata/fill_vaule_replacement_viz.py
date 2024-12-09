import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the file with a flexible separator (tabs or spaces)
df = pd.read_csv(
    ".\\omnidata\\files_copy\\omni_min_vh1min.lst",
    delim_whitespace=True,  # Handles both tabs and multiple spaces
    header=None,  # No header in the file
)

# Optionally assign column names for clarity
df.columns = [
    "Year",
    "Day",
    "Hour",
    "Minute",
    "Timeshift",
    "Field_Magnitude_nT",
    "Speed_km_per_s",
    "Proton_Density_n_per_cc",
    "Proton_Temperature_K",
    "Flow_Pressure_nPa",
    "Alfven_Mach_Number",
]


# Subset of the first 100 rows
subset = df.iloc[:100].copy()

# List of placeholder values to replace
placeholder_values = [99999.9, 999.99, 9999999.0, 99.99, 999.9]

# Replace placeholders with NaN
subset.replace(placeholder_values, np.nan, inplace=True)

# Apply Linear Interpolation
linear_filled = subset.copy()
linear_filled.interpolate(method="linear", axis=0, inplace=True)

# Apply Forward Fill
forward_filled = subset.copy()
forward_filled.ffill(axis=0, inplace=True)


# List of columns to visualize (you can exclude non-numeric ones if needed)
columns_to_plot = [
    "Speed_km_per_s",
    "Proton_Density_n_per_cc",
    "Proton_Temperature_K",
    "Flow_Pressure_nPa",
    "Field_Magnitude_nT",
]

if True:  # change true/false based on weather you want the plot or not
    # Loop through each column and plot
    for col in columns_to_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(
            subset.index,
            subset[col],
            label=f"Original {col} (with NaN)",
            linestyle="dotted",
            color="gray",
        )
        plt.plot(
            linear_filled.index,
            linear_filled[col],
            label=f"Linear Interpolation {col}",
            linestyle="solid",
        )
        plt.plot(
            forward_filled.index,
            forward_filled[col],
            label=f"Forward Fill {col}",
            linestyle="dashed",
        )
        plt.xlabel("Row Index")
        plt.ylabel(col)
        plt.title(f"Comparison of Interpolation Methods for {col}")
        plt.legend()
        plt.show()


# after seeing all the graphs i decided that linear fills look closer
