from pickle import TRUE
from turtle import onclick, onkeyrelease, width
from typing_extensions import Self
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as pp
from scipy.interpolate import interp1d
from scipy import signal
import scipy as sc
import math
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout='wide')


st.markdown("""
<style>
.css-18ni7ap.e8zbici2
{
    visibility:hidden;
}
.css-qri22k.egzxvld0
{
    visibility:hidden;
}
.css-9s5bis.edgvbvh3
{
    visibility:hidden;
}
.block-container.css-18e3th9.egzxvld2
{
    padding: 1rem 1rem 1rem 1rem;
}
.css-10trblm.e16nr0p30
{
    text-align:center;
    font-family: Arial;
}
.css-hxt7ib.e1fqkh3o4
{
    padding: 0.5rem 1rem 0rem 1rem;
}
.css-81oif8.effi0qh3
{
    font-size:16px;
}
.css-1p46ort.effi0qh3 {
    font-size: 1px;
    color: rgb(49, 51, 63);
    display: flex;
    visibility: hidden;
    margin-bottom: 0rem;
    height: auto;
    min-height: 0rem;
    vertical-align: middle;
    flex-direction: row;
    -webkit-box-align: center;
    align-items: center;
}
.css-k8kh4s {
    font-size: 1px;
    color: rgba(49, 51, 63, 0.4);
    display: flex;
    visibility: hidden;
    margin-bottom: 0rem;
    height: auto;
    min-height: 0rem;
    vertical-align: middle;
    flex-direction: row;
    -webkit-box-align: center;
    align-items: center;
}

</style>
""", unsafe_allow_html=True)


if 'freqsample' not in st.session_state:
    st.session_state['freqsample'] = 0

if 'table' not in st.session_state:
    st.session_state['table'] = []


if 'fig_main' not in st.session_state:
    st.session_state['fig_main'] = go.Figure()


if 'fig_hidden' not in st.session_state:
    st.session_state['fig_hidden'] = go.Figure()


if 'SNR' not in st.session_state:
    st.session_state['SNR'] = 10000

if 'sampling_choice' not in st.session_state:
    st.session_state['sampling_choice'] = 'Sampling frequency'


def f_max(magnitude=[], time=[]):
    sample_period = time[1]-time[0]
    n_samples = len(time)
    fft_magnitudes = np.abs(np.fft.fft(magnitude))
    fft_frequencies = np.fft.fftfreq(n_samples, sample_period)
    fft_clean_frequencies_array = []
    for i in range(len(fft_frequencies)):
        if fft_magnitudes[i] > 1:
            fft_clean_frequencies_array.append(fft_frequencies[i])
    f_maximum = max(fft_clean_frequencies_array)
    return f_maximum


def nySample(time, amplitude, fs):
    if len(time) == len(amplitude):
        points_per_indices = int((len(time) / time[-1]) / fs)
        if points_per_indices == 0:
            points_per_indices = 1
        amplitude = amplitude[::points_per_indices]
        time = time[::points_per_indices]
        return time, amplitude


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


def update_signal(magnitude, frequency):
    for i in range(len(st.session_state['time'])):
        st.session_state['signal_drawn'][i] += magnitude * \
            np.sin(2*np.pi*frequency*st.session_state['time'][i])


def noise(snr):

    SNR = 10.0**(snr/10.0)
    p1 = st.session_state['signal_drawn'].var()
    n = p1/SNR
    noise = sc.sqrt(n)*sc.randn(len(st.session_state['signal_drawn']))
    # signal after Noise
    st.session_state['magnitude'] = st.session_state['signal_drawn']+noise
    st.session_state['fig_main'] = pp.line(
        x=st.session_state['time'], y=st.session_state['magnitude'])


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def draw():
    st.session_state['Fmax'] = f_max(
        st.session_state['magnitude'],  st.session_state['time'])

    if (st.session_state['sampling_choice'] != 'Sampling frequency'):
        Actual_frequency = st.session_state['freqsample'] * \
            st.session_state['Fmax']
    else:
        Actual_frequency = st.session_state['freqsample']

    sampleTime, sampleAmplitude = nySample(
        st.session_state['time'], st.session_state['magnitude'], Actual_frequency)

    st.session_state['fig_main'].add_trace(go.Scatter(
        x=sampleTime, y=sampleAmplitude, mode='markers', visible=True, marker=dict(color="red"), name='Sample Points'))

    time_og = np.linspace(time[0], time[-1], 2000)
    y = sinc_interpolation(sampleAmplitude, sampleTime, time_og)

    st.session_state['fig_main'].add_trace(go.Scatter(
        x=time_og, y=y, mode='lines', marker=dict(color="black"), visible=True, name='Reconstructed'))

    st.session_state['fig_main'].update_layout(xaxis_range=[-0.001, 1.001], yaxis_range=[(-1*max(max(st.session_state['magnitude']), max(y))-1), (max(max(st.session_state['magnitude']), max(y))+1)],
                                               xaxis_title="Time(seconds)", yaxis_title="Amplitude", font=dict(family='Arial', size=18))

    st.session_state['fig_hidden'] = pp.scatter(
        x=sampleTime, y=sampleAmplitude)

    st.session_state['fig_hidden'].add_trace(go.Scatter(
        x=time_og, y=y, mode='lines', marker=dict(color="black"), visible=True, name='Reconstructed'))

    st.session_state['fig_hidden'].update_layout(xaxis_range=[-0.001, 1.001], yaxis_range=[(-1*max(max(st.session_state['magnitude']), max(y))-1), (max(max(st.session_state['magnitude']), max(y))+1)],
                                                 xaxis_title="Time(seconds)", yaxis_title="Amplitude", font=dict(family='Arial', size=18))

    df = pd.DataFrame({"x": time_og, "y": sinc_interpolation(
        sampleAmplitude, sampleTime, time_og)})
    csv = convert_df(df)
    st.sidebar.download_button(
        label="Save", data=csv, file_name='Signal.csv', mime='text/csv')


def change():
    if 'time' not in st.session_state:
        st.session_state['time'] = []

    if 'signal_drawn' not in st.session_state:
        st.session_state['signal_drawn'] = []

    if 'magnitude' not in st.session_state:
        st.session_state['magnitude'] = []
    del st.session_state['time']
    del st.session_state['signal_drawn']
    del st.session_state['magnitude']
    st.session_state['table'].clear()


st.title("Sampling Studio")
file = st.sidebar.file_uploader(
    "Open signal", type={"csv", "txt", "csv.xls"}, label_visibility="hidden", on_change=change, key="file")


hide = st.checkbox(label="Hide Original Signal", value=False)


col1, col2 = st.sidebar.columns(2)
added_magnitude = col1.number_input(label="Signal Magnitude:", step=1, value=0)
added_frequency = col2.number_input(label="Signal Frequency:", step=1, value=0)

add_btn = st.sidebar.button('Add')

SNR = st.sidebar.slider(label="SNR", min_value=0,
                        step=1, max_value=100, value=100)

undo_signals = st.sidebar.multiselect(
    "Remove signals", options=st.session_state['table'], label_visibility="hidden")

remove_btn = st.sidebar.button('Remove')

st.session_state['sampling_choice'] = st.sidebar.selectbox(
    'Sample Using:', ('Sampling frequency', 'Factor*Fmax'))
if (st.session_state['sampling_choice'] == 'Sampling frequency'):
    st.session_state['freqsample'] = st.sidebar.slider(
        label="Sampling frequency:", min_value=2, max_value=100, step=1, label_visibility="hidden")
else:
    st.session_state['freqsample'] = st.sidebar.slider(
        label="Factor*Fmax:", min_value=0.25, max_value=10.0, step=0.1, label_visibility="hidden")


def functions():

    if add_btn:
        update_signal(added_magnitude, added_frequency)
        st.session_state['table'].append([added_magnitude, added_frequency])
        st.experimental_rerun()

    if remove_btn:
        for item in undo_signals:
            update_signal(-1.0*item[0], item[1])
            for item2 in st.session_state['table']:
                if item == item2:
                    st.session_state['table'].remove(item2)
        st.experimental_rerun()

    noise(SNR)
    draw()

    if hide:
        st.plotly_chart(
            st.session_state['fig_hidden'], use_container_width=True)
    else:
        st.plotly_chart(st.session_state['fig_main'], use_container_width=True)


if file is not None:
    File = pd.read_csv(file)
    csv_data = File.to_numpy()
    time = csv_data[:, 0]
    magnitude = csv_data[:, 1]

    if 'time' not in st.session_state:
        st.session_state['time'] = time

    if 'signal_drawn' not in st.session_state:
        st.session_state['signal_drawn'] = magnitude

    if 'magnitude' not in st.session_state:
        st.session_state['magnitude'] = magnitude

    if 'Fmax' not in st.session_state:
        st.session_state['Fmax'] = f_max(
            st.session_state['magnitude'],  st.session_state['time'])

    functions()
else:
    File = pd.read_csv(
        r'C:\Users\LAPTOP\OneDrive\المستندات\GitHub\DSP_Task1_-20-\Docs\signal3.csv')
    csv_data = File.to_numpy()
    time = csv_data[:, 0]
    magnitude = csv_data[:, 1]

    if 'time' not in st.session_state:
        st.session_state['time'] = time

    if 'signal_drawn' not in st.session_state:
        st.session_state['signal_drawn'] = magnitude

    if 'magnitude' not in st.session_state:
        st.session_state['magnitude'] = magnitude

    if 'Fmax' not in st.session_state:
        st.session_state['Fmax'] = f_max(
            st.session_state['magnitude'],  st.session_state['time'])

    functions()
