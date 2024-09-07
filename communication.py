import socket
import librosa
import numpy as np
import soundfile as sf
import io

SERVER_HOST = '127.0.0.1'
BUFFER_SIZE = 4096 

def receive_audio(port=49165, parent=None):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, port))
    server_socket.listen(1)
    server_socket.settimeout(0.2) 

    try:
        while not parent.close_socket:
            try:
                client_socket, addr = server_socket.accept()
            except socket.timeout:
                continue

            audio_data = b""
            while True:
                chunk = client_socket.recv(BUFFER_SIZE)
                if not chunk:
                    break
                audio_data += chunk

            audio_stream = io.BytesIO(audio_data)
            audio, sr = librosa.load(audio_stream, sr=None, mono=True)

            client_socket.close()

            yield audio, sr

    finally:
        server_socket.close()

def send_audio(audio_input, sample_rate=None, port=49165):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((SERVER_HOST, port))

    if isinstance(audio_input, np.ndarray):
        if sample_rate is None:
            raise ValueError("Sample rate must be provided for waveform input.")
        
        buffer = io.BytesIO()
        
        sf.write(buffer, audio_input, sample_rate, format='WAV')
        buffer.seek(0) 

        while True:
            chunk = buffer.read(BUFFER_SIZE)
            if not chunk:
                break
            client_socket.sendall(chunk)

    elif isinstance(audio_input, str):
        with open(audio_input, "rb") as audio_file:
            while True:
                chunk = audio_file.read(BUFFER_SIZE)
                if not chunk:
                    break
                client_socket.sendall(chunk)
    
    else:
        raise ValueError("audio_input must be either a file path or a numpy int16 waveform array.")

    client_socket.close()

