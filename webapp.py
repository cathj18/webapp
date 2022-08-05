#sound = AudioSegment.from_wav(path+name)
            # sound = sound.set_channels(1)
            # sound.export(path+name, format="wav")

            # y, _ = librosa.load(path+name, sr = 16000)
import streamlit as st
import os
import sys
import numpy as np
import librosa
import librosa.display
import sklearn
# from sklearn import metrics._dist_metrics
# from sklearn.neighbors import _dist_metrics
#import plotly.express as px
#import matplotlib.pyplot as plt
#import sound
import pickle
from pydub import AudioSegment
#import SessionState
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go
import streamlit.components.v1 as components  # Import Streamlit
import pandas as pd

def extract_features(data):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
    try:
        arr = dict()
        arr['bandwidth'] = librosa.feature.spectral_bandwidth(y=data).mean()
        arr['flatness'] = librosa.feature.spectral_flatness(y=data).mean()
        arr['centroid'] = librosa.feature.spectral_centroid(y=data).mean()
        arr['rolloff'] = librosa.feature.spectral_rolloff(y=data).mean()
        arr['contrast'] = librosa.feature.spectral_contrast(y=data).mean()
        arr['mfcc'] = librosa.feature.mfcc(y=data).mean()
        arr['zcr'] = librosa.feature.zero_crossing_rate(y=data).mean()
        arr['stft'] = librosa.feature.chroma_stft(y=data).mean()
        arr['cqt'] = librosa.feature.chroma_cqt(y=data).mean()
        arr['cens'] = librosa.feature.chroma_cens(y=data).mean()
        arr['rms'] = librosa.feature.rms(y=data).mean()
        arr['tonnetz'] = librosa.feature.tonnetz(y=data).mean()
        arr['poly'] = librosa.feature.poly_features(y=data).mean()
        arr['mel_spec'] = librosa.feature.melspectrogram(y=data).mean()
    except Exception as e:
        print("Error encountered while opening file")
        return None
    data = pd.DataFrame()
    data = data.append(arr, ignore_index=True)
    return data


def display_results(uploaded_file=None, flag='uploaded'):

    if flag=='uploaded':
        audio_bytes = uploaded_file.getvalue()
    elif flag=='recorded':
        audio_bytes = open(uploaded_file, 'rb').read()
    # sound = AudioSegment.from_wav(uploaded_file)
    # sound = sound.set_channels(1)
    # sound.export(uploaded_file,format='wav')
    st.subheader('Sample of the submitted audio')
    st.audio(audio_bytes, format='audio/wav')
    y, _ = librosa.load(uploaded_file, sr = 16000)
    input_features = extract_features(y)
    input = input_features[set(input_features.columns)-set(['contrast','centroid','rolloff','zcr','bandwidth','cqt','stft'])]
    output = int(model.predict(input)[0])
    st.text(output)
    if(output==1):
        st.text("You may be depressed.")
    else:
        st.text("You are likely not depressed.")

def load_model():
    loaded_model = pickle.load(open('Downloads/KNN_dep.sav', 'rb'))
    return loaded_model

data_load_state = st.text('Loading data...')
model = load_model()
data_load_state.text("Done!")

st.subheader('Please submit your audio')
#session_state = SessionState.get(name='', path=None)

with st.form(key='uploader'):
    uploaded_file = st.file_uploader("Choose a file... (Try to keep the audio short 5-6 seconds and upload as a .wav file)")
    submit_button_upl = st.form_submit_button(label='Submit the uploaded audio')

# import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Sample rate
seconds = 15  # Duration of recording

# if st.button('Record'):
#   with st.spinner(f'Recording for 5 seconds ....'):
#       try:
#         myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
#         sd.wait()  # Wait until recording is finished
#         write("audio1.wav", fs, myrecording)  # Save as WAV file
#         # session_state.path = sound.record()
#       except:
#           pass
#   st.success("Recording completed")

if st.button('Submit the recorded audio'):
    filename = "audio1.wav"
    display_results(filename, flag='recorded')
    os.remove(filename)

if (uploaded_file is None and submit_button_upl):
  st.subheader("Something's not right, please refresh the page and retry!")

elif uploaded_file and submit_button_upl:
    display_results(uploaded_file, flag="uploaded")
