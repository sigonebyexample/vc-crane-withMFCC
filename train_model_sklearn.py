import os
import numpy as np
import warnings
from scipy.fftpack import dct
from scipy.signal import resample
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = "dataset"
SAMPLE_RATE = 16000
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.010
N_MFCC = 13
PRE_EMPHASIS = 0.97
NFFT = 512

MODEL_TYPE = 'rf'

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

print("Extracting MFCC+Delta+DeltaDelta features...")
X = []
y = []
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"Classes found: {class_names}")

for class_name in class_names:
    class_path = os.path.join(DATA_DIR, class_name)
    files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    print(f"Processing class '{class_name}' with {len(files)} files...")
    for file in files:
        filepath = os.path.join(class_path, file)
        try:
            signal = read_wave_file(filepath, SAMPLE_RATE)
            features = extract_features_from_signal(signal)
            X.append(features)
            y.append(class_name)
        except Exception as e:
            print(f"  Error in {filepath}: {e}")

X = np.array(X)
y = np.array(y)
print(f"\nTotal samples: {len(X)}")
print(f"Feature dimensions: {X.shape[1]}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Class encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if MODEL_TYPE == 'rf':
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model_name = "Random Forest"
else:
    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42
    )
    model_name = "SVM"

print(f"\nTraining {model_name}...")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data: {acc*100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sklearn_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(le, "models/label_encoder.joblib")

mfcc_config = {
    'sample_rate': SAMPLE_RATE,
    'frame_size': FRAME_SIZE,
    'frame_stride': FRAME_STRIDE,
    'n_mfcc': N_MFCC,
    'pre_emphasis': PRE_EMPHASIS,
    'nfft': NFFT,
    'feature_dim': X.shape[1]
}
joblib.dump(mfcc_config, "models/mfcc_config.joblib")

print("\nModel and config saved in models folder.")
print("=" * 60)
print("Training complete. You can now run voice_robot_realtime_sklearn.py")
