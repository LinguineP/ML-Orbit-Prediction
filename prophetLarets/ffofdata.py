import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def transform_to_timestamp(in_data):
    df = pd.DataFrame(in_data)
    df["ds"] = df["date"] + " " + df["time"]
    df = df.drop(["date", "time"], axis=1)
    df = df[["ds", "x", "y", "z"]]
    return df


def separate_components(in_data):
    df_x = pd.DataFrame(in_data)
    df_y = pd.DataFrame(in_data)
    df_z = pd.DataFrame(in_data)
    df_x = df_x.drop(["y", "z"], axis=1)
    df_y = df_y.drop(["x", "z"], axis=1)
    df_z = df_z.drop(["y", "x"], axis=1)
    df_x = df_x.rename(columns={"x": "y"})
    df_z = df_z.rename(columns={"z": "y"})
    return df_x, df_y, df_z


in_data = pd.read_csv("D:\\fax\\master\\op-ml\\prophetLarets\\train_data.csv")


timeSeriesData = transform_to_timestamp(in_data)
data_x, data_y, data_z = separate_components(timeSeriesData)


data_y["ds"] = pd.to_datetime(data_y["ds"])
data_x["elapsed_time"] = (data_x["ds"] - data_x["ds"].iloc[0]).dt.total_seconds()


time = data_x["elapsed_time"].values
signal = data_x["y"].values


d = time[1] - time[0]


fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=d)
amplitude = np.abs(fft_result)

frequencies_per_day = frequencies * 86400


positive_freqs = frequencies_per_day[frequencies_per_day > 0]
positive_amplitude = amplitude[frequencies_per_day > 0]


periods_in_days = 1 / positive_freqs


plt.plot(positive_freqs, positive_amplitude)
plt.xlabel("Frequency (cycles per day)")
plt.ylabel("Amplitude")
plt.title("Frequency Spectrum")
plt.show()


threshold = 0.1 * max(positive_amplitude)
significant_freqs = positive_freqs[positive_amplitude > threshold]
significant_amplitudes = positive_amplitude[positive_amplitude > threshold]
significant_periods = 1 / significant_freqs

# Display the significant frequencies, their amplitudes, and periods
for i, freq in enumerate(significant_freqs):
    print(
        f"Frequency: {freq:.4f} cycles/day, Period: {significant_periods[i]:.6f} days, Amplitude: {significant_amplitudes[i]:.4f}"
    )


fourier_order = len(significant_freqs)
print(f"\nSuggested Fourier Order: {fourier_order}")

"""
Frequency: 13.6179 cycles/day, Period: 0.073433 days, Amplitude: 30530177137.2358
Frequency: 13.6224 cycles/day, Period: 0.073408 days, Amplitude: 141652642016.6460
Frequency: 13.6269 cycles/day, Period: 0.073384 days, Amplitude: 53536837997.4647
Frequency: 13.6314 cycles/day, Period: 0.073360 days, Amplitude: 22549421557.0259
Frequency: 15.6233 cycles/day, Period: 0.064007 days, Amplitude: 213006889964.0411

Suggested Fourier Order: 5
"""
