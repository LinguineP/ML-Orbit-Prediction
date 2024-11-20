import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


def tranform_to_timestamp(in_data):
    df = pd.DataFrame(in_data)
    df["ds"] = df["date"] + " " + df["time"]
    df = df.drop(["date", "time"], axis=1)
    df = df[["ds", "x", "y", "z"]]

    return df


def separate_components(in_data):
    df_x = pd.DataFrame(in_data)
    df_y = pd.DataFrame(in_data)
    df_Z = pd.DataFrame(in_data)
    df_x = df_x.drop(["y", "z"], axis=1)
    df_y = df_y.drop(["x", "z"], axis=1)
    df_z = df_Z.drop(["y", "x"], axis=1)
    df_x = df_x.rename(columns={"x": "y"})
    df_z = df_z.rename(columns={"z": "y"})
    return df_x, df_y, df_z


def get_rows_before_n(df: pd.DataFrame, n: int):
    """
    Returns a tuple containing:
    1. The last 1000 rows before the row at index `n`.
    2. The contents of the `n`th row.

    Parameters:
    df (pd.DataFrame): The DataFrame from which rows will be selected.
    n (int): The index of the row at which to stop and get the 1000 rows before it.

    Returns:
    tuple: A tuple where:
        - The first element is the DataFrame of the last 1000 rows before `n`.
        - The second element is a Series representing the `n`th row.
    """
    # Ensure there are no out-of-bounds errors
    start_index = max(n - 1000, 0)  # Start at index 0 if `n - 1000` is less than 0

    # Select the last 1000 rows before `n`
    train_set = df.iloc[start_index:n]

    # Select the row at index `n`
    prediction_target = df.iloc[n]

    # Return a tuple (last 1000 rows, nth row)
    return train_set, prediction_target


def get_prediction_series(df, target_value):
    matching_row = df[df["ds"] == target_value]
    if not matching_row.empty:
        return (
            matching_row.squeeze()
        )  # Returns a Series if one row matches, otherwise remains a DataFrame
    else:
        return None  # Return None if no match is found


def plot_ds_y_and_yhat(df1, df2):
    """
    Plot `ds` and `y` from `df1` and `ds` and `yhat` from `df2`, ensuring that `ds` is in datetime format.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame containing `ds` and `y` columns.
    - df2 (pd.DataFrame): Second DataFrame containing `ds` and `yhat` columns.
    """
    # Convert `ds` column to datetime in both DataFrames
    df1["ds"] = pd.to_datetime(df1["ds"])
    df2["ds"] = pd.to_datetime(df2["ds"])

    plt.figure(figsize=(10, 6))

    # Plot `ds` vs `y` from df1
    plt.plot(df1["ds"], df1["y"], label="Actual (y)", color="blue", marker="o")

    # Plot `ds` vs `yhat` from df2

    plt.plot(
        df2["ds"],
        df2["yhat"],
        label="Predicted (yhat)",
        color="red",
        linestyle="--",
        marker="x",
    )

    # Labeling
    plt.xlabel("Date and Time (ds)")
    plt.ylabel("Values")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.grid()
    plt.show()


def plot_ds_absError(df1):
    """
    Plot `ds` and `y` from `df1` and `ds` and `yhat` from `df2`, ensuring that `ds` is in datetime format.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame containing `ds` and `y` columns.
    - df2 (pd.DataFrame): Second DataFrame containing `ds` and `yhat` columns.
    """
    # Convert `ds` column to datetime in both DataFrames
    df1["timeStamp"] = pd.to_datetime(df1["timeStamp"])

    plt.figure(figsize=(10, 6))

    # Plot `ds` vs `y` from df1
    plt.plot(
        df1["timeStamp"],
        df1["absoluteError"],
        label="Actual (y)",
        color="blue",
        marker="o",
    )

    # Plot `ds` vs `yhat` from df2

    # Labeling
    plt.xlabel("Date and Time (ds)")
    plt.ylabel("Values")
    plt.title("abserror")
    plt.legend()
    plt.grid()
    plt.show()


def train_predict_1000(data, targetRow):

    trainSet, predictionTarget = get_rows_before_n(
        data, targetRow
    )  # takes 1000 rows before n and n

    timeStamp = predictionTarget.loc["ds"]

    model = Prophet(growth="flat")

    # seasonalities params aquired from furier transform, experimentaly came to this configuration
    model.add_seasonality(
        name="s0",
        period=0.064007,
        fourier_order=5,
        prior_scale=None,
        mode="additive",
    )
    model.add_seasonality(
        name="s1",
        period=0.073360,
        fourier_order=1,
        prior_scale=None,
        mode="additive",
    )

    model.add_seasonality(
        name="s2",
        period=0.073384,
        fourier_order=1,
        prior_scale=None,
        mode="additive",
    )

    model.add_seasonality(
        name="s3",
        period=0.073408,
        fourier_order=1,
        prior_scale=None,
        mode="additive",
    )

    model.fit(trainSet)

    future = model.make_future_dataframe(freq="3min", periods=500)

    forecast = model.predict(future)

    # data extraction
    targetPrediction = get_prediction_series(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], timeStamp
    )

    predictedValue = targetPrediction.loc["yhat"]

    realValue = predictionTarget.loc["y"]

    # plot_ds_y_and_yhat(trainSet, forecast)

    return timeStamp, realValue, predictedValue


if __name__ == "__main__":

    in_data = pd.read_csv("D:\\fax\\master\\op-ml\\prophetLarets\\train_data.csv")

    timeSeriesData = tranform_to_timestamp(in_data)

    data_x, data_y, data_z = separate_components(timeSeriesData)

    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(
        columns=["timeStamp", "realValue", "predictedValue", "absoluteError"]
    )

    stopIndex = 1100

    for index, row in data_x.iloc[1000:].iterrows():
        if index == stopIndex:
            break
        timeStamp, realValue, predictedValue = train_predict_1000(data_x, index)
        absoluteError = abs(realValue - predictedValue)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "timeStamp": timeStamp,
                        "realValue": realValue,
                        "predictedValue": predictedValue,
                        "absoluteError": absoluteError,
                    },
                    index=[0],
                ),
            ]
        )

    df.to_csv("dfDump")
    plot_ds_absError(df)
