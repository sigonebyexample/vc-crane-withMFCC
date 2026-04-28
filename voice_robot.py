import os
import json
import time
import threading
import webbrowser
import numpy as np
from scipy.fftpack import dct
from scipy.signal import resample
from scipy.io import wavfile
import warnings
from http.server import HTTPServer, SimpleHTTPRequestHandler

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.010
N_MFCC = 13
PRE_EMPHASIS = 0.97
NFFT = 512
CONFIDENCE_THRESHOLD = 50.0

def read_wave_file(filepath, target_sr=SAMPLE_RATE):
    sr, audio = wavfile.read(filepath)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio.astype(np.float32) / max_val
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, num_samples)
    return audio

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framing(signal, sample_rate, frame_size, frame_stride):
    frame_len = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    sig_len = len(signal)
    num_frames = int(np.ceil(float(np.abs(sig_len - frame_len)) / frame_step)) + 1
    pad_len = int((num_frames - 1) * frame_step + frame_len)
    pad_signal = np.pad(signal, (0, pad_len - sig_len), 'constant')
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32)]
    frames *= np.hamming(frame_len)
    return frames

def magspec(frames, nfft=512):
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = (mag_frames ** 2) / nfft
    return pow_frames

def mel_filterbank(nfilt, nfft, sample_rate, low_freq=0, high_freq=None):
    if high_freq is None:
        high_freq = sample_rate / 2
    low_mel = 2595 * np.log10(1 + low_freq / 700)
    high_mel = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)
    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = bin[m - 1]
        f_m = bin[m]
        f_m_plus = bin[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return fbank

def compute_delta(mfcc_feat, N=2):
    num_frames = mfcc_feat.shape[0]
    delta = np.zeros_like(mfcc_feat)
    for t in range(num_frames):
        numerator = np.zeros(mfcc_feat.shape[1])
        denominator = 0
        for n in range(1, N+1):
            if t + n < num_frames and t - n >= 0:
                numerator += n * (mfcc_feat[t+n] - mfcc_feat[t-n])
                denominator += n**2
        delta[t] = numerator / (2 * denominator) if denominator > 0 else 0
    return delta

def extract_features_from_signal(signal):
    signal = pre_emphasis(signal)
    frames = framing(signal, SAMPLE_RATE, FRAME_SIZE, FRAME_STRIDE)
    pow_frames = magspec(frames, 512)
    nfilt = 26
    fbank = mel_filterbank(nfilt, 512, SAMPLE_RATE)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :13]
    delta1 = compute_delta(mfcc, 2)
    delta2 = compute_delta(delta1, 2)
    features = []
    for feat in [mfcc, delta1, delta2]:
        features.append(np.mean(feat, axis=0))
        features.append(np.std(feat, axis=0))
    return np.concatenate(features)

MODEL_PATH = "models/voice_model_improved.npz"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/voice_model_enhanced.npz"

data = np.load(MODEL_PATH, allow_pickle=True)
X_train = data["X_train"]
y_train = data["y_train"]
mean = data["mean"]
std = data["std"]
class_names = data["class_names"].tolist()
K = int(data["k"]) if "k" in data else 5
n_classes = len(class_names)

def predict_command(features_norm):
    distances = np.sqrt(np.sum((X_train - features_norm) ** 2, axis=1))
    nearest_idx = np.argsort(distances)[:K]
    nearest_distances = distances[nearest_idx]
    nearest_labels = y_train[nearest_idx]
    weights = 1.0 / (nearest_distances + 1e-10)
    weights = weights / weights.sum()
    scores = np.zeros(n_classes)
    for label, weight in zip(nearest_labels, weights):
        scores[label] += weight
    chap_id = class_names.index("chap") if "chap" in class_names else None
    if chap_id is not None:
        current_best = np.max(scores)
        current_best_id = np.argmax(scores)
        if current_best_id != chap_id and scores[chap_id] > current_best * 0.4:
            scores[chap_id] *= 1.4
    pred_id = np.argmax(scores)
    confidence = scores[pred_id] * 100
    return class_names[pred_id], confidence

COMMAND_FILE = "command.wav"
STATUS_FILE = "command.json"

def process_commands():
    print("Monitoring command.wav file...")
    while True:
        if os.path.exists(COMMAND_FILE):
            try:
                signal = read_wave_file(COMMAND_FILE, SAMPLE_RATE)
                features = extract_features_from_signal(signal)
                features_norm = (features - mean) / (std + 1e-10)
                command, confidence = predict_command(features_norm)
                with open(STATUS_FILE, "w") as f:
                    json.dump({
                        "command": command,
                        "confidence": confidence,
                        "timestamp": time.time()
                    }, f)
                print(f"Command: {command} (confidence: {confidence:.1f}%)")
                os.remove(COMMAND_FILE)
            except Exception as e:
                print(f"Error: {e}")
        time.sleep(0.5)

def start_server():
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("", 8000), handler)
    print("Server started on port 8000. Address: http://localhost:8000")
    webbrowser.open("http://localhost:8000")
    httpd.serve_forever()

if __name__ == "__main__":
    if not os.path.exists("index.html"):
        print("index.html not found! Please place it in the project folder.")
        exit(1)
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    process_commands()
