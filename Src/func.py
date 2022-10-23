import numpy as np
import pandas as pd
import math

# approximating points


def approximate(n, m):
    return m * math.ceil(n / m)


# getting maximum frequency
def f_max(magnitude=[], time=[]):
    sample_period = time[1]-time[0]
    n_samples = len(time)
    fft_magnitudes = np.abs(np.fft.fft(magnitude))
    fft_frequencies = np.fft.fftfreq(n_samples, sample_period)
    fft_clean_frequencies_array = []
    for i in range(len(fft_frequencies)):
        if fft_magnitudes[i] > 100:
            fft_clean_frequencies_array.append(fft_frequencies[i])
    f_maximum = max(fft_clean_frequencies_array)
    return f_maximum


# sampling csv file
def nyquist_sampling(time, signal_magnitude, rate):
    step = time[1]-time[0]
    fm = f_max(signal_magnitude, time)
    T = 1/fm
    df = pd.DataFrame({"time": time, "Amplitude": signal_magnitude})
    sample_rate = rate * fm
    sampling_period = 1/sample_rate
    time = np.arange(T/4, df['time'].iloc[-1], sampling_period)
    amplitude = []
    for time_point in time:
        amplitude.append(
            df.iloc[int(round(approximate(time_point, step), 10)/step)]['Amplitude'])
    signal = [time, amplitude]
    return signal


# reconstruction
def sinc_interpolation(input_magnitude, input_time, original_time):

    if len(input_magnitude) != len(input_time):
        print('not same')

    # Find the period
    if len(input_time) != 0:
        T = input_time[1] - input_time[0]

    # the equation
    sincM = np.tile(original_time, (len(input_time), 1)) - \
        np.tile(input_time[:, np.newaxis], (1, len(original_time)))
    output_magnitude = np.dot(input_magnitude, np.sinc(sincM/T))
    return output_magnitude
