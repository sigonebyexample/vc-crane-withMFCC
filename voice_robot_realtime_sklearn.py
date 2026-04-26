import tkinter as tk
import numpy as np
import threading
import queue
import warnings
import sounddevice as sd
from scipy.fftpack import dct
from scipy.signal import resample
import joblib
import time

warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_RATE = 16000
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.010
N_MFCC = 13
PRE_EMPHASIS = 0.97
NFFT = 512

BUFFER_DURATION = 1.5
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.6
CONFIDENCE_THRESHOLD = 0.15

NOISE_FLOOR = 0.005
VAD_MULTIPLIER = 2.5

model = joblib.load("models/sklearn_model.joblib")
scaler = joblib.load("models/scaler.joblib")
le = joblib.load("models/label_encoder.joblib")
print("Model loaded.")
print(f"Classes: {le.classes_}")

def pre_emphasis(signal, coeff=PRE_EMPHASIS):
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

def magspec(frames, nfft=NFFT):
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
    if num_frames < 5:
        return np.zeros_like(mfcc_feat)
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
    if len(signal) < int(FRAME_SIZE * SAMPLE_RATE):
        return None
    signal = pre_emphasis(signal)
    frames = framing(signal, SAMPLE_RATE, FRAME_SIZE, FRAME_STRIDE)
    pow_frames = magspec(frames, NFFT)
    nfilt = 26
    fbank = mel_filterbank(nfilt, NFFT, SAMPLE_RATE)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :N_MFCC]
    delta1 = compute_delta(mfcc, N=2)
    delta2 = compute_delta(delta1, N=2)
    features = []
    for feat in [mfcc, delta1, delta2]:
        features.append(np.mean(feat, axis=0))
        features.append(np.std(feat, axis=0))
    return np.concatenate(features)

def predict_with_sklearn(features):
    features_scaled = scaler.transform([features])
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0]
        pred_id = np.argmax(proba)
        confidence = proba[pred_id]
        
        print(f"   Probabilities:")
        sorted_idx = np.argsort(proba)[::-1]
        for i in sorted_idx:
            bar = "█" * int(proba[i] * 40)
            print(f"      {le.inverse_transform([i])[0]:8s}: {proba[i]:.3f}  {bar}")
    else:
        pred_id = model.predict(features_scaled)[0]
        confidence = 1.0
    return pred_id, confidence

command_queue = queue.Queue()

def calibrate_noise_floor():
    print("Calibrating environment noise...")
    audio = sd.rec(int(1.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    energy = np.sqrt(np.mean(audio**2))
    threshold = energy * VAD_MULTIPLIER
    print(f"   Noise energy: {energy:.6f}")
    print(f"   Speech threshold: {threshold:.6f}")
    print(f"   Tip: If too sensitive, increase VAD_MULTIPLIER")
    return threshold

def audio_thread_func():
    global NOISE_FLOOR
    
    vad_threshold = calibrate_noise_floor()
    
    buffer_size = int(SAMPLE_RATE * BUFFER_DURATION)
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    is_speaking = False
    speech_frames = []
    silence_counter = 0
    speech_counter = 0
    silence_threshold_frames = int(SILENCE_DURATION * SAMPLE_RATE / 1024)
    min_speech_frames = int(MIN_SPEECH_DURATION * SAMPLE_RATE / 1024)
    
    print("\nListening system active.")
    print(f"Speech threshold: {vad_threshold:.6f}")
    print(f"Minimum speech duration: {MIN_SPEECH_DURATION} seconds")
    print("Speak (not too quietly)...\n")
    
    def callback(indata, frames, time, status):
        nonlocal audio_buffer, is_speaking, speech_frames, silence_counter, speech_counter
        
        audio_buffer = np.roll(audio_buffer, -frames)
        audio_buffer[-frames:] = indata[:, 0]
        energy = np.sqrt(np.mean(indata**2))
        
        if energy > vad_threshold:
            if not is_speaking:
                speech_frames = list(audio_buffer[:-frames])
                is_speaking = True
                speech_counter = 0
                print(f"\nSpeech detected (energy: {energy:.6f})")
            
            speech_frames.extend(indata[:, 0])
            speech_counter += 1
            silence_counter = 0
            
        else:
            if is_speaking:
                speech_frames.extend(indata[:, 0])
                silence_counter += 1
                
                if silence_counter > silence_threshold_frames:
                    total_speech_duration = speech_counter * 1024 / SAMPLE_RATE
                    
                    if speech_counter >= min_speech_frames:
                        print(f"End of speech (duration: {total_speech_duration:.2f}s)")
                        print("Processing...")
                        
                        audio_segment = np.array(speech_frames[-int(SAMPLE_RATE * BUFFER_DURATION):])
                        features = extract_features_from_signal(audio_segment)
                        
                        if features is not None:
                            pred_id, conf = predict_with_sklearn(features)
                            command = le.inverse_transform([pred_id])[0]
                            
                            if conf >= CONFIDENCE_THRESHOLD:
                                print(f"   Command: {command.upper()} (confidence: {conf:.3f})")
                                command_queue.put(command)
                            else:
                                print(f"   Low confidence ({conf:.3f}) - command ignored")
                        else:
                            print("   Feature extraction failed")
                    else:
                        print(f"Short noise ignored ({total_speech_duration:.2f}s)")
                    
                    is_speaking = False
                    speech_frames = []
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=1024):
        threading.Event().wait()

class VoiceRobotApp:
    def __init__(self, master):
        self.master = master
        master.title("Voice Command Robot")
        master.geometry("800x600")
        master.resizable(False, False)
        
        self.canvas = tk.Canvas(master, width=800, height=600, bg='black')
        self.canvas.pack()
        
        self.colors = {
            "jelo": "#00BFFF", "aghab": "#FF8C00",
            "rast": "#32CD32", "chap": "#FFD700", "ist": "#808080"
        }
        
        self.x, self.y = 400, 300
        self.robot = self.canvas.create_oval(380, 280, 420, 320, fill='red', outline='white')
        
        self.label = tk.Label(master, text="Ready - Speak", 
                              font=("Arial", 14), fg="white", bg="black")
        self.label.place(x=10, y=10)
        
        self.master.after(100, self.check_queue)
    
    def check_queue(self):
        try:
            cmd = command_queue.get_nowait()
            self.label.config(text=f"Command: {cmd.upper()}")
            self.move(cmd)
            self.master.after(2000, lambda: self.label.config(text="Ready - Speak"))
        except queue.Empty:
            pass
        self.master.after(100, self.check_queue)
    
    def move(self, cmd):
        color = self.colors.get(cmd, "white")
        self.canvas.create_oval(self.x-5, self.y-5, self.x+5, self.y+5, fill=color, outline="")
        
        if cmd == "jelo": self.y -= 15
        elif cmd == "aghab": self.y += 15
        elif cmd == "rast": self.x += 15
        elif cmd == "chap": self.x -= 15
        
        self.x = max(20, min(780, self.x))
        self.y = max(20, min(580, self.y))
        self.canvas.coords(self.robot, self.x-20, self.y-20, self.x+20, self.y+20)

if __name__ == "__main__":
    audio_thread = threading.Thread(target=audio_thread_func, daemon=True)
    audio_thread.start()
    
    time.sleep(1.5)
    
    root = tk.Tk()
    app = VoiceRobotApp(root)
    root.mainloop()
