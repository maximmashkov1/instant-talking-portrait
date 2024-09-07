import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import winsound
from scipy.interpolate import interp1d
import librosa
import librosa.display
import noisereduce as nr

from scipy.ndimage import gaussian_filter
def low_pass_filter(tensor, framerate, cutoff_freq):
  filter_length = int(framerate / cutoff_freq)
  filter_length = filter_length if filter_length % 2 == 1 else filter_length + 1 
  filter_kernel = np.exp(-0.5 * np.linspace(-filter_length // 2, filter_length // 2, filter_length)**2 / (filter_length / 3)**2)
  filter_kernel /= np.sum(filter_kernel)
  padded_tensor = np.pad(tensor, ((filter_length // 2, filter_length // 2), (0, 0), (0, 0)), mode='edge')
  filtered_tensor = np.zeros_like(tensor)
  for v in range(tensor.shape[1]):
    filtered_tensor[:, v, 0] = np.convolve(padded_tensor[:, v, 0], filter_kernel, mode='valid')
    filtered_tensor[:, v, 1] = np.convolve(padded_tensor[:, v, 1], filter_kernel, mode='valid')

  return filtered_tensor


def filter_frequencies(signal, sample_rate, low_pass, high_pass=8):
    freq_signal = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/sample_rate)
    freq_signal[np.abs(freq) < low_pass] = 0
    freq_signal[np.abs(freq) > high_pass] = 0
    filtered_signal = np.fft.ifft(freq_signal)
    return np.real(filtered_signal)


def modeled_mouth_to_complete(modeled, base):
  modeled = modeled.copy()
  jaw = modeled[:, -1].flatten()
  modeled = modeled[:, :-1] #N, 24
  modeled = modeled.reshape(modeled.shape[0], -1, 2) #N, 12, 2
  anim=np.repeat(base.copy()[np.newaxis, :], modeled.shape[0], axis=0)
  
  right = modeled.copy()
  right[:, :, 0] *= -1

  print("right:",right.shape)
  print("anim:",anim.shape)
  print("modeled:",modeled.shape)
  
  anim[:,0]+=modeled[:,0]
  anim[:,1]+=modeled[:,1]
  anim[:,2]+=modeled[:,2]
  anim[:,3]+=modeled[:,3]
  anim[:,4]+=right[:,2]
  anim[:,5]+=right[:,1]
  anim[:,6]+=right[:,0]
  anim[:,7]+=right[:,6]
  anim[:,8]+=right[:,5]
  anim[:,9]+=modeled[:,4]
  anim[:,10]+=modeled[:,5]
  anim[:,11]+=modeled[:,6]
  anim[:,12]+=modeled[:,7]
  anim[:,13]+=modeled[:,8]
  anim[:,14]+=modeled[:,9]
  anim[:,15]+=right[:,8]
  anim[:,16]+=right[:,7]
  anim[:,17]+=right[:,11]
  anim[:,18]+=modeled[:,10]
  anim[:,19]+=modeled[:,11]
  return anim, jaw


def animate_vertices(vertices, framerate):

    if vertices.ndim != 3 or vertices.shape[2] != 2:
        raise ValueError("The vertices array must be of shape (F, V, 2).")

    F, V, _ = vertices.shape

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(vertices[:, :, 0]), np.max(vertices[:, :, 0]))
    ax.set_ylim(np.min(vertices[:, :, 1]), np.max(vertices[:, :, 1]))

    # Plot the initial positions
    scat = ax.scatter(vertices[0, :, 0], vertices[0, :, 1])

    def update(frame):
        # Update the positions of the vertices
        scat.set_offsets(vertices[frame])

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=F, interval=1000/framerate, repeat=False)

    # Show the animation
    plt.show()


def animate_edge_movement(vertices, framerate):
    from matplotlib.animation import FuncAnimation
    edges = np.array([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11),(11,0),(12,13), (13, 14), (14,15), (15,16), (16,17), (17, 18), (18, 19), (19, 12), (19, 14)])
    F, V, _ = vertices.shape
    E, _ = edges.shape
    
    fig, ax = plt.subplots()
    
    # Plot initial vertex positions
    scatter = ax.scatter(vertices[0, :, 0], vertices[0, :, 1])

    # Plot initial edges
    lines = []
    for edge in edges:
        line, = ax.plot(vertices[0, edge, 0], vertices[0, edge, 1], 'k-', lw=2)
        lines.append(line)

    ax.set_xlim(np.min(vertices[:, :, 0]) - 1, np.max(vertices[:, :, 0]) + 1)
    ax.set_ylim(np.min(vertices[:, :, 1]) - 1, np.max(vertices[:, :, 1]) + 1)
    
    def update(frame):
        # Update vertex positions
        scatter.set_offsets(vertices[frame])

        # Update edge positions
        for i, edge in enumerate(edges):
            lines[i].set_data(vertices[frame, edge, 0], vertices[frame, edge, 1])
        
        return scatter, *lines

    anim = FuncAnimation(fig, update, frames=F, interval=1000/framerate, blit=True)
    plt.show()



import threading

def get_mel_spectrogram(y, sr, target_sr = 16000, n_mels=64, framerate=30, tempo_multiplier=1, sr_multiplier=1, play=False):
    hop_length=int(target_sr/framerate)
    sr *= sr_multiplier

    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.95)  

    y_resampled = librosa.resample(y_denoised, orig_sr=sr, target_sr=target_sr)

    y_resampled = y_resampled / np.max(np.abs(y_resampled))

    if tempo_multiplier != 1:
       y_resampled = librosa.effects.time_stretch(y_resampled,rate=tempo_multiplier)
    

    mel_spectrogram = librosa.feature.melspectrogram(y=y_resampled, sr=target_sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    if play:
      import sounddevice as sd
      plt.figure()
      librosa.display.specshow(mel_spectrogram)
      plt.colorbar()
      plt.show()
      sd.play(y_resampled, target_sr)
      sd.wait()  


    return mel_spectrogram.astype(np.float32)




import sys

def resample_keypoints(data, new_size):
    original_size = data.shape[0]
    resampled_data = np.zeros((new_size, data.shape[1]))
    original_indices = np.linspace(0, original_size - 1, num=original_size)
    new_indices = np.linspace(0, original_size - 1, num=new_size)

    for i in range(data.shape[1]):
        interp_func = interp1d(original_indices, data[:, i], kind='linear')
        resampled_data[:, i] = interp_func(new_indices)
    return resampled_data


def estimate_f0_and_power(y, sr, target_sr = 16000, framerate=30, tempo_multiplier=1, sr_multiplier=1, play=False):
    hop_length=int(target_sr/framerate)
    sr *= sr_multiplier

    #y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.90)  

    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    y_resampled = librosa.util.normalize(y_resampled)

    if tempo_multiplier != 1:
       y_resampled = librosa.effects.time_stretch(y_resampled,rate=tempo_multiplier)

    f0, voiced_flag, voiced_probs = librosa.pyin(y_resampled, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length = 2048, hop_length=hop_length, fill_na=None)
    f0_smooth = gaussian_filter(f0, sigma=0.5)
    
    ######################

    frames = librosa.util.frame(np.pad(y_resampled,(1024, 1024)), frame_length=2048, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    energy_db = librosa.power_to_db(energy, ref=np.max)
    energy_db_smooth = gaussian_filter(energy_db, sigma=0.5)
    
    if play:
      import sounddevice as sd
      plt.plot(f0_smooth)
      plt.plot(energy_db_smooth)
      plt.plot(voiced_flag)
      sd.play(y_resampled, target_sr)
      plt.show()
      

    out = np.hstack((f0_smooth.reshape(-1, 1), energy_db_smooth.reshape(-1, 1), voiced_flag.reshape(-1, 1)))
    return out