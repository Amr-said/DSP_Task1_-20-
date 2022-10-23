from os import access
from finalfunc import *
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as pp
from scipy import signal
from scipy import interpolate
import math
import plotly.graph_objects as go
import scipy as sc
import math

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


if 'time' not in st.session_state:
        st.session_state['time']=np.linspace(0,1,1000)

if 'signal_drawn' not in st.session_state:
  st.session_state['signal_drawn']=np.zeros(len( st.session_state['time'])) 


if 'sampled_signal_drawn' not in st.session_state:
  st.session_state['sampled_signal_drawn']=[]

if 'freqsample' not in st.session_state:
  st.session_state['freqsample']=0     

if 'table' not in st.session_state:
  st.session_state['table']=[]   

if 'fig' not in st.session_state:
  st.session_state['fig']=pp.line()  

if 'fig2' not in st.session_state:
  st.session_state['fig2']=pp.line() 
  
if 'fig3' not in st.session_state:
  st.session_state['fig3']=pp.line()    

if 'fig4' not in st.session_state:
  st.session_state['fig4']=pp.line()

if "added_csv_function" not in st.session_state:
    st.session_state["added_csv_function"] = []

def approximate(n, m):
    return m * math.ceil(n / m)


def f_max(magnitude=[], time=[]):
    sample_period = time[1]-time[0]
    n_samples = len(time)
    fft_magnitudes = np.abs(np.fft.fft(magnitude))
    fft_frequencies = np.fft.fftfreq(n_samples, sample_period)
    fft_clean_frequencies_array = []
    for i in range(len(fft_frequencies)):
        if fft_magnitudes[i] > 0.001:
            fft_clean_frequencies_array.append(fft_frequencies[i])
    f_maximum = max(fft_clean_frequencies_array)
    return f_maximum


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



def update_signal(magnitude,frequency):    
    for i in range(len(st.session_state['time'])):
     st.session_state['signal_drawn'][i] += magnitude*np.sin(2*np.pi*frequency*st.session_state['time'][i])
    

def update_signal2(magnitude,frequency):        
    y= magnitude*np.sin(2*np.pi*frequency*st.session_state['time'])
    st.session_state['fig2'].add_scatter(x=st.session_state['time'], y=y,name="frequency:"+str(frequency))  


def noise(snr,add):
    if add:
      SNR=10.0**(snr/10.0)      
      p1=st.session_state['signal_drawn'].var()
      n=p1/SNR
      noise=sc.sqrt(n)*sc.randn(len(st.session_state['signal_drawn']))   
      mixed=st.session_state['signal_drawn']+noise   #signal after Noise
      st.session_state['fig']=pp.line( x=st.session_state['time'], y=mixed)
    else:
     st.session_state['fig']=pp.line( x=st.session_state['time'], y=st.session_state['signal_drawn'])
     

def clear_signal():    
    st.session_state['signal_drawn']=np.zeros(len(st.session_state['time']))
    for item in st.session_state['table']:
         st.session_state['table'].remove(item) 


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
                                   y=df.columns[1], title="Original signal")
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

    col3, col4 = st.columns([3,1],gap="small")
    with col3:
     added_magnitude=st.slider(label="Signal Magnitude:",step=1)
     added_frequency=st.slider(label="Signal Frequency:",step=1)
     
    with col4:
     snr_checkbox=st.checkbox("Add SNR")
     if snr_checkbox:
       SNR=st.slider(label="SNR",min_value=0,step=1,max_value=1000)
       noise(SNR,True)
     else :noise(0,False)    
        
     add_btn=st.button('Add')
     if add_btn :
      update_signal(added_magnitude,added_frequency)
      update_signal2(added_magnitude,added_frequency)
      st.session_state['table'].append([added_magnitude,added_frequency])
      st.experimental_rerun()  
    
    undo_signals=st.multiselect("Remove signals", options=st.session_state['table'])
    remove_btn=st.button('Remove')
    if remove_btn :
      for item in undo_signals:  
       update_signal(-1.0*item[0],item[1])
       for item2 in st.session_state['table']:
        if item==item2:
           st.session_state['table'].remove(item2) 
      st.experimental_rerun() 


    clear=st.button('Clear All')
    if clear :
      clear_signal()
      st.experimental_rerun()


    st.subheader("Original signal:")
    st.session_state['fig'].update_layout(width=1000,height=500)
    st.plotly_chart(st.session_state['fig'],use_container_width=True)


    with st.expander("Choose sampling rate:"):
        st.session_state['freqsample']=st.slider(label="Sampling rate:",min_value=2,max_value=100,step=1) 
#  sampling(st.session_state['freqsample'])


    # samples=nyquist_sampling(st.session_state['time'], st.session_state['signal_drawn'], freqsample)
    # originaltime=np.linspace(0,1,len( st.session_state['time']))
    # interpolation=sinc_interpolation(samples,  originaltime,st.session_state['time'])
    # st.session_state['fig3'].add_scatter(x=st.session_state['time'], y=interpolation,mode='lines')

    
    st.subheader("Original signal + sampled signal:")
    st.session_state['fig3'].update_layout(width=1000,height=500)
    st.plotly_chart(st.session_state['fig3'],use_container_width=True)


    st.subheader("Added signals:")
    st.session_state['fig2'].update_layout(width=1000,height=500)
    st.plotly_chart(st.session_state['fig2'],use_container_width=True)