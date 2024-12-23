import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
import time
import logging

logging.disable(logging.CRITICAL)


def transform_to_timestamp(in_data):
    df = pd.DataFrame(in_data)
    df["ds"] = df["date"] + " " + df["time"]
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m-%d %H:%M:%S")
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")
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
    Plot absolute error over time and other comparisons with proper layout.

    Parameters:
    - df1 (pd.DataFrame): DataFrame containing columns:
        - timeStamp: datetime or string representing time.
        - absoluteError: actual error values.
        - realValue: ground truth values.
        - predictedValue: predicted values.
    """

    # Ensure `timeStamp` is in datetime format
    df1["timeStamp"] = pd.to_datetime(df1["timeStamp"])

    # Plot absolute error over time
    plt.figure(figsize=(10, 6))
    plt.plot(
        df1["timeStamp"],
        df1["absoluteError"],
        label="Absolute Error",
        color="blue",
        marker="o",
    )
    plt.xlabel("Date and Time")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error Over Time")
    plt.legend()
    plt.grid()
    plt.show()

    # Print a sample of the dataframe for debugging
    print(df1.head())

    # Plot realValue vs predictedValue for additional insight
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # First plot: Ground Truth vs Predicted
    ax[0, 0].plot(
        df1["timeStamp"],
        df1["realValue"],
        label="Ground Truth",
        color="blue",
        linewidth=2,
    )
    ax[0, 0].plot(
        df1["timeStamp"],
        df1["predictedValue"],
        label="Predicted",
        color="lightblue",
        linestyle="--",
        linewidth=2,
    )
    ax[0, 0].set_title("Ground Truth vs Predicted")
    ax[0, 0].set_xlabel("Time")
    ax[0, 0].set_ylabel("Values")
    ax[0, 0].legend()

    # Add other subplots as needed, here's an example:
    # Example: Ground Truth vs Absolute Error
    ax[0, 1].plot(
        df1["timeStamp"],
        df1["absoluteError"],
        label="Absolute Error",
        color="red",
        linewidth=2,
    )
    ax[0, 1].set_title("Absolute Error Over Time")
    ax[0, 1].set_xlabel("Time")
    ax[0, 1].set_ylabel("Absolute Error")
    ax[0, 1].legend()

    # Customize the rest of the subplots based on your needs

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
def warm_start_params(m):
    """
    Retrieve parameters from a trained model in the format used to initialize a new Stan model.
    Note that the new Stan model must have these same settings:
        n_changepoints, seasonality features, mcmc sampling
    for the retrieved parameters to be valid for the new model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ['delta', 'beta']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res


def train_predict_1000(prevModel,data, targetRow,isZ=False):


    trainSet, predictionTarget = get_rows_before_n(
        data, targetRow
    )

    last_timestamp = trainSet.iloc[-1]['ds']

    timeStamp = predictionTarget.loc["ds"]
    
    if isZ:
        model=init_prophet_model_z()    
    else:
        model=init_prophet_model_xy()
    
    
    logging.getLogger('prophet').setLevel(logging.WARNING)
    if prevModel!=None:
        
        model.fit(trainSet,init=warm_start_params(prevModel))
    else:
        
        model.fit(trainSet)

    future = model.make_future_dataframe(freq="3min", periods=100)

    forecast = model.predict(future)

    # data extraction
    targetPrediction = get_prediction_series(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], timeStamp
    )

    predictedValue = targetPrediction.loc["yhat"]

    realValue = predictionTarget.loc["y"]

    # plot_ds_y_and_yhat(trainSet, forecast)

    return timeStamp, realValue, predictedValue,model,last_timestamp

def init_prophet_model_xy():
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
    
    return model

def init_prophet_model_z():
    model = Prophet(growth="flat")

    model.add_seasonality(
        name="s0",
        period=0.68384,
        fourier_order=10,
        prior_scale=None,
        mode="additive",
    )
    
    return model




if __name__ == "__main__":

    in_data = pd.read_csv("/home/pavle/op-ml/prophetLarets/train_data.csv")

    timeSeriesData = transform_to_timestamp(in_data)

    data_x, data_y, data_z = separate_components(timeSeriesData)

    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(
        columns=["timeStamp", "realValue", "predictedValue", "absoluteError"]
    )

    stopIndex = 1100
    
    model=None
    
    
    

    for index, row in data_x.iloc[1000:].iterrows():
        if index == stopIndex:
            break
        timeStamp, realValue, predictedValue,model,_= train_predict_1000(model,data_y, index)
        
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

    df.to_csv("dfDump",index=False)
    plot_ds_absError(df)
