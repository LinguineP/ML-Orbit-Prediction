import pandas as pd
import numpy as np


from mango import scheduler, Tuner
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from scipy.stats import uniform


import logging

logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

train_df = None
test_df = None


# loss function
def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)


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


def plot_series_dataset():
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(6)

    plt.plot(train_df["ds"], train_df["y"], linewidth=4, label="Train Series")
    plt.plot(test_df["ds"], test_df["y"], linewidth=4, label="Test Series")

    plt.legend(fontsize=25)
    plt.ylabel("Value", fontsize=25)
    plt.xticks([])
    plt.show()


def objective_function(args_list):
    global train_df, test_df

    params_evaluated = []
    results = []

    for params in args_list:
        try:
            model = Prophet(**params)
            model.fit(train_df)
            future = model.make_future_dataframe(freq="3min", periods=10)
            forecast = model.predict(future)
            predictions_tuned = forecast.tail(Test_size)
            error = mape(test_df["y"], predictions_tuned["yhat"])

            params_evaluated.append(params)
            results.append(error)
        except:
            # print(f"Exception raised for {params}")
            # pass
            params_evaluated.append(params)
            results.append(25.0)  # Giving high loss for exceptions regions of spaces

        # print(params_evaluated, mse)
    return params_evaluated, results


if __name__ == "__main__":

    in_data = pd.read_csv("D:\\fax\\master\\op-ml\\prophetLarets\\train_data.csv")

    timeSeriesData = tranform_to_timestamp(in_data)

    data_x, data_y, data_z = separate_components(timeSeriesData)

    # Create an empty DataFrame with the specified columns
    Test_size = int(40)

    train_df = data_x.head(len(data_x) - Test_size)
    test_df = data_x.tail(Test_size)

    param_space = dict(
        growth=["flat"],
        n_changepoints=range(0, 60, 3),
        changepoint_range=uniform(0.5, 0.5),
        seasonality_prior_scale=uniform(0.01, 20.0),
        changepoint_prior_scale=uniform(0.001, 1.0),
    )

    conf_Dict = dict()
    conf_Dict["initial_random"] = 10
    conf_Dict["num_iteration"] = 100

    tuner = Tuner(param_space, objective_function, conf_Dict)
    results = tuner.minimize()
    print("best parameters:", results["best_params"])
    print("best loss:", results["best_objective"])

    # plot_ds_absError(df)
