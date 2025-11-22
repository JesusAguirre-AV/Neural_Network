import os
import librosa
import librosa.display
import IPython.display as ipd
import lpath
import numpy as np
import matplotlib.pyplot as plt

"""This is an optional part of the code, meant to generate a visualisation of the spectrogram, may need some
modifications before using but only meant to provide visual aid and will not be part of the grad. Code was based on 
'Music classification with Neural Networks' doc under Project 3"""
def plot_spectrogram(title, y, sr, hop_length, y_axis = "linear"):
   plt.figure(figsize=(10,6))
   librosa.display.specshow(y, sr = sr, hop_length = hop_length, x_axis = "time", y_axis = y_axis)
   #plt.colorbar(format="%+2.f")
   plt.title(title)
   plt.show()


def extract_and_plot(audio_data, frameSize, hopSize, title):
   audio, sample_rate = librosa.load(audio_data)
   stft_audio = librosa.stft(audio, n_fft = frameSize, hop_length = hopSize)
   y_audio = np.abs(stft_audio) ** 2
   plot_spectrogram(title+' linear', y_audio, sample_rate, hopSize)
   y_log_audio = librosa.power_to_db(y_audio)
   plot_spectrogram(title+' log', y_log_audio, sample_rate, hopSize)
   plot_spectrogram(title+' log and y_axis log', y_log_audio, sample_rate, hopSize, y_axis = "log")




frameSize = 2048
hopSize = 512


audio1 = lpath+"/country.00000.au"
extract_and_plot(audio1, frameSize, hopSize,'country.00000')


audio2 = lpath+"/classical.00000.au"
extract_and_plot(audio2, frameSize, hopSize,'classical.00000')