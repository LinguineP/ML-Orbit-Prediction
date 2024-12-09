import pyshtools as pysh


import pandas as pd
import numpy as np
import pyproj


wgs84 = pyproj.Proj(proj="latlong", datum="WGS84")
cartesian = pyproj.Proj(proj="geocent", datum="WGS84")


def cartesian_to_geodetic(x, y, z):
    earth_radius = 6371000
    lon, lat, alt = pyproj.transform(cartesian, wgs84, x, y, z, radians=False)
    return lat, lon, earth_radius + alt


if __name__ == "__main__":

    df = pd.read_csv("D:\\fax\\master\\op-ml\\NRLMSISE\\output.csv")

    df["Timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])

    df[["latitude", "longitude", "radius"]] = df.apply(
        lambda row: pd.Series(cartesian_to_geodetic(row["x"], row["y"], row["z"])),
        axis=1,
    )  # executes the coordinates conversion over the whole dataframe

    GOCO9s_dataset = pysh.datasets.Earth.GOCO06S()

    perturbation_r = []
    perturbation_theta = []
    perturbation_phi = []
    for index, row in df.iterrows():

        pertrubation_data = GOCO9s_dataset.expand(
            lat=row["longitude"],
            lon=row["latitude"],
            r=row["radius"],
        )  # gravity vector components [r(Radial component), theta((Polar or Latitudinal component)), phi((Azimuthal or Longitudinal component)) ]
        perturbation_r.append(pertrubation_data[0])
        perturbation_theta.append(pertrubation_data[1])
        perturbation_phi.append(pertrubation_data[2])
    df["pertrubation_r"] = perturbation_r
    df["pertrubation_theta"] = perturbation_theta
    df["pertrubation_phi"] = perturbation_phi

    # Keep only the required columns
    output_df = df[
        ["Timestamp", "pertrubation_r", "pertrubation_theta", "pertrubation_phi"]
    ]

    # Save the results to a CSV file
    output_df.to_csv("gravity_pertrubations_out.csv", index=False)
