from os import access
from func import *
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as pp
from scipy.interpolate import interp1d
from scipy import signal
from scipy import interpolate
import math
import plotly.graph_objects as go

st.markdown("""

<style>
.css-9s5bis.edgvbvh3{
    visibility: hidden;
}
.css-1q1n0ol.egzxvld0{
    visibility: hidden;
}
.css-9ycgxx.exg6vvm12{
    visibility: hidden;
}
.css-1aehpvj.euu6i2w0{
    visibility: hidden;

</style>""", unsafe_allow_html=True)

if "added_csv_function" not in st.session_state:
    st.session_state["added_csv_function"] = []


def drawsignal(magnitude, frequency, snr_dB=1000000):

    # sampling_freq = st.slider(
    #     label="Sample freq.:", min_value=2, max_value=500, step=1)
    time = np.linspace(0, 1, 1000)
    signal = magnitude*np.sin(2*np.pi*frequency*time)
    signal = fSNR(snr_dB, signal)
    fig1 = plt.figure()
    # plt.subplot(2,1,1)
    plt.plot(time, signal)
    st.plotly_chart(fig1)
    return signal


def fSNR(snr_dB, signal):
    power = signal ** 2
    siganl_average_power = np.mean(power)
    signalpower_db = 10*np.log10(power)
    siganl_average_power_dB = 10*np.log10(siganl_average_power)
    noise_dB = siganl_average_power_dB - snr_dB
    noise_watts = 10**(noise_dB / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_watts), len(signal))
    noise_signal = signal + noise
    return noise_signal


tab1, tab2 = st.tabs(["Upload CSV", "Generate Signal"])

with tab1:
    st.header("Upload CSV")
    file = st.sidebar.file_uploader("Open signal", type={"csv", "txt"})
    remove_csv_list = []
    index = 0
    for signal in st.session_state["added_csv_function"]:
        remove_csv_list.append([str(index+1)+') amp: ' +
                               str(signal[0]), 'freq: '+str(signal[1])])
        index += 1

    remove_from_csv = st.sidebar.selectbox("remove Function", remove_csv_list)
    remove_from_csv_button = st.sidebar.button("remove function")
    index = 0
    if (remove_from_csv_button):
        for value in remove_csv_list:
            if (remove_from_csv == value):
                st.session_state['added_csv_function'].pop(index)
                break
            index += 1

    add_function_csv_check_box = st.checkbox("add Signal ")
    add_function_csv_button = 0
    if (add_function_csv_check_box):
        add_function_csv_amplituide = st.slider(
            "Amplituide", min_value=0, max_value=100)
        add_function_csv_frequency = st.slider(
            "Ferquency", min_value=0, max_value=100)
        add_function_csv_button = st.button("Add Signal")
    else:
        st.session_state["added_csv_function"] = []


if file is not None:
    df = pd.read_csv(file)
    csv_data = df.to_numpy()
    x_csv_data = csv_data[:, 0]
    y_csv_data = csv_data[:, 1]
    if (add_function_csv_button):
        st.session_state["added_csv_function"].append(
            [add_function_csv_amplituide, add_function_csv_frequency])
        st.experimental_rerun()
    if (add_function_csv_check_box):
        for value in st.session_state['added_csv_function']:
            y_csv_data += value[0]*np.sin(value[1]*x_csv_data)

    CsvSNR_checkbox, CsvSNR = st.sidebar.columns(2)
    CsvSNR_checkbox = CsvSNR_checkbox.checkbox("Add CSV SNR")
    if CsvSNR_checkbox:
        CsvSNR = st.sidebar.slider(
            label="Add Csv SNR", min_value=0, max_value=100, step=1)
        df[df.columns[1]] = fSNR(CsvSNR, df[df.columns[1]])
    fig_Nyquist_sampling = pp.line(df, x=df.columns[0],
                                   y=df.columns[1], title="Original signal")
    sample_points = st.slider(
        label="Factor * Fmax:", min_value=0.25, max_value=100.0, step=0.1)
    ys = nyquist_sampling(df[df.columns[0]], df[df.columns[1]], sample_points)
    fig_Nyquist_sampling.add_scatter(x=ys[0], y=ys[1], mode='markers')
    st.plotly_chart(fig_Nyquist_sampling, use_container_width=True)

    ts = ys[0]
    magn = ys[1]

    time_og = np.linspace(ts[0], ts[-1], 2000)
    fig_recostruction = go.Figure()
    fig_recostruction.add_trace(
        go.Line(x=time_og, y=sinc_interpolation(magn, ts, time_og)))

    st.plotly_chart(fig_recostruction, use_container_width=True)

    # st.download_button("Save CSV", fig_recostruction.to_csv(), mime='text/csv')

    # save = st.button("save")
    #     if save:
    #         if curr_tap.source == "generate":
    #             curr_tap.set_attributes(magnitude=magnitude, time=time, amplitude=amplitude, frequency=freq,
    #                                     noise_check_box=noise_check_box, snr=snr, sampling_rate=sampling_rate)
    #         if curr_tap.source == "csv":
    #             curr_tap.set_attributes(time=curr_tap.time, magnitude=curr_tap.magnitude,
    #                                     noise_check_box=noise_check_box, source="csv", snr=snr, sampling_rate=sampling_rate)

with tab2:
    st.header("Generate Signal")

    Magnitude, Frequency = st.sidebar.columns(2)
    Magnitude = st.sidebar.slider(
        label="Magnitude:", min_value=0.0, max_value=100.0, step=0.5)
    Frequency = st.sidebar.slider(
        label="Frequency:", min_value=0.0, max_value=100.0, step=0.5)

    snr_checkbox, SNR = st.sidebar.columns(2)
    snr_checkbox = snr_checkbox.checkbox("Add SNR")

    if snr_checkbox:
        SNR = SNR.number_input("SNR", min_value=0, step=1, max_value=1000)
        drawsignal(Magnitude, Frequency, SNR)
    else:
        drawsignal(Magnitude, Frequency)

    if (remove_from_csv_button):
        st.experimental_rerun()
