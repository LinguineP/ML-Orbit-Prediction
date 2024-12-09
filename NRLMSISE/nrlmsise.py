import pandas as pd
import numpy as np
import pymsis
import pyproj


wgs84 = pyproj.Proj(proj="latlong", datum="WGS84")
cartesian = pyproj.Proj(proj="geocent", datum="WGS84")


# Function for coordinate transformation
def cartesian_to_geodetic(x, y, z):
    lon, lat, alt = pyproj.transform(cartesian, wgs84, x, y, z, radians=False)
    return lat, lon, alt / 1000.0  # Altitude in kilometers


if __name__ == "__main__":

    df = pd.read_csv("D:\\fax\\master\\op-ml\\NRLMSISE\\output.csv")

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

    df[["latitude", "longitude", "altitude"]] = df.apply(
        lambda row: pd.Series(cartesian_to_geodetic(row["x"], row["y"], row["z"])),
        axis=1,
    )  # executes the coordinates conversion over the whole dataframe

    densities = []
    for index, row in df.iterrows():
        time = np.array([row["datetime"]], dtype="datetime64[s]")
        density_data = pymsis.calculate(
            time,
            row["longitude"],
            row["latitude"],
            row["altitude"],
            geomagnetic_activity=-1,
        )
        densities.append(density_data[0, 0])
    df["total_mass_density"] = densities

    # Keep only the required columns
    output_df = df[["datetime", "total_mass_density"]]

    # Save the results to a CSV file
    output_df.to_csv("total_mass_density_out.csv", index=False)
    print("Results saved to total_mass_density_out.csv")
