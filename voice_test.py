import pyaudio
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import time

def test_microphone():
    """
    Simple test to verify microphone functionality
    """
    print("Testing microphone...")
    
    # Audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 3
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Print available input devices
    print("\nAvailable input devices:")
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"Device {i}: {device_info['name']}")
    
    # Select default input device
    print("\nUsing default input device for recording...")
    
    try:
        # Open audio stream
        stream = audio.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         frames_per_buffer=CHUNK)
        
        print("Recording 3 seconds of audio. Please speak...")
        
        # Record audio
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        print("Recording complete.")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Convert frames to numpy array
        audio_data = b''.join(frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate energy
        energy = np.mean(np.abs(audio_data))
        print(f"Average audio energy: {energy}")
        
        # Plot the waveform
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(audio_data)
        plt.title("Audio Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        
        # Plot the spectrum
        fft_data = fft(audio_data)
        fft_data = np.abs(fft_data[:len(fft_data)//2])
        freqs = np.linspace(0, RATE/2, len(fft_data))
        
        plt.subplot(2, 1, 2)
        plt.plot(freqs, fft_data)
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.show()
        
        # Basic command detection simulation
        if energy < 100:
            print("Detected: silence")
        elif np.argmax(fft_data) < len(fft_data) // 4:
            print("Detected: low frequency sound (like 'stop')")
        else:
            print("Detected: high frequency sound (like 'go')")
        
        print("Microphone test completed successfully!")
        
    except Exception as e:
        print(f"Error during microphone test: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your microphone is connected and enabled")
        print("2. Check your system sound settings")
        print("3. Try installing the following dependencies:")
        print("   pip install pyaudio numpy scipy matplotlib")
    
    # Clean up
    audio.terminate()

if __name__ == "__main__":
    test_microphone()