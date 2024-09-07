from communication import send_audio
import librosa

if __name__=='__main__':

    waveform, sr = librosa.load('audio.wav', sr=None, mono=True)
    send_audio(waveform, sr)
    # or   
    send_audio('audio.wav')