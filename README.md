```bash
git clone https://github.com/sigonebyexample/vc-crane-withMFCC.git
cd vc-crane-withMFCC
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
git add .
git commit -m "Initial commit: voice-controlled crane with MFCC"
git push origin main
```

---

### README.md
```markdown
# 🎤 VC-Crane with MFCC

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.43%25-brightgreen.svg)]()

A real-time voice-controlled industrial crane simulation using **MFCC feature extraction** and **Weighted KNN / Random Forest** machine learning models. Built in 3 days — demonstrating that AI is accessible to everyone.

---

## ✨ Features

- 🎙️ Real-time voice recognition with adaptive VAD (Voice Activity Detection)
- 🏗️ Interactive crane simulation with HTML5 Canvas
- 📡 Two operating modes: File-based and Real-time (microphone)
- 🧠 Multiple ML models: Hand-coded KNN, Weighted KNN, Random Forest, SVM
- 📊 96.43% accuracy on test data
- 🗣️ Persian language command recognition

---

## 🎮 Demo

Open `crane_demo.html` in any browser to control the crane:

| Command | Persian | Action |
|---------|---------|--------|
| Forward | جلو (Jelo) | Hook moves UP ⬆️ |
| Back | عقب (Aghab) | Hook moves DOWN ⬇️ |
| Right | راست (Rast) | Arm EXTENDS ➡️ |
| Left | چپ (Chap) | Arm RETRACTS ⬅️ |
| Stop | ایست (Ist) | Everything STOPS ⏹️ |

---

## 🧠 How It Works

```
🎤 Voice → 🔊 Audio (16kHz WAV) → 📊 78 MFCC Features → 🧠 ML Model → 🏗️ Crane Movement
```

### Pipeline

1. **Record:** 50 samples per command recorded with phone microphone
2. **Extract:** 78-dimensional MFCC features (13 MFCC + 13 Delta + 13 Delta-Delta)
3. **Train:** Multiple models evaluated (KNN, Weighted KNN, Random Forest, SVM)
4. **Predict:** Real-time inference in under 100ms

---

## 📦 Installation

```bash
git clone https://github.com/sigonebyexample/vc-crane-withMFCC.git
cd vc-crane-withMFCC
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train Model
```bash
python train_model_sklearn.py
```

### Real-Time Voice Control
```bash
python voice_robot_realtime_sklearn.py
```

### Interactive Demo
Open `crane_demo.html` in your browser.

---

## 📊 Results

| Model                 | Accuracy |
|-----------------------|----------|
| KNN (hand-coded, K=3) | 92.26%   |
| Weighted KNN (K=5)    | 96.43%   |
| Random Forest         | 96.43%   |
| SVM (RBF)             | 94.00%   |

| Class          | Accuracy |
|----------------|----------|
| Aghab (Back)   | 97.22%   |
| Chap (Left)    | 96.43%   |
| Ist (Stop)     | 87.88%   |
| Jelo (Forward) | 100%     |
| Rast (Right)   | 100%     |

---

## 🔧 Technologies

- **Python 3.8+**
- **scikit-learn** — Random Forest, SVM, StandardScaler
- **NumPy / SciPy** — MFCC implementation, signal processing
- **sounddevice** — Real-time audio capture
- **HTML5 Canvas** — Crane visualization
- **Joblib** — Model serialization

---

## 🚧 Challenges Solved

| Problem                                    | Solution                                            |
|--------------------------------------------|-----------------------------------------------------|
| File format incompatibility (RIFF error)   | Switched from `wave` to `scipy.io.wavfile`          |
| Covariance matrix singularity              | Z-Score + Euclidean distance for outlier removal    |
| Domain mismatch (train vs test microphone) | Data augmentation: noise, pitch shift, time stretch |
| Windows environment conflicts              | Migrated to Linux + venv                            |
| Browser microphone access (HTTP/HTTPS)     | File-based fallback via Google Drive                |

---

## 📝 Lessons Learned

1. **Microphone matching matters** — 96% accuracy on training data means nothing if test hardware differs
2. **Weighted KNN at K=5** hits the sweet spot between accuracy and generalization
3. **Data augmentation is essential** — 250 samples became 840, significantly improving robustness
4. **"Ist" (Stop) is the hardest class** — low phonetic diversity, easily confused with silence
5. **VAD needs calibration** — adaptive threshold based on ambient noise is minimum viable

---

## 👤 Author

**sigonebyexample**

[github.com/sigonebyexample](https://github.com/sigonebyexample)
