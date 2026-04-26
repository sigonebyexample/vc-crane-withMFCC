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

| Command | Persian     | Action              |
|---------|-------------|---------------------|
| Forward | جلو (Jelo)  | Hook moves UP ⬆️    |
| Back    | عقب (Aghab) | Hook moves DOWN ⬇️  |
| Right   | راست (Rast) | Arm EXTENDS ➡️      |
| Left    | چپ (Chap)   | Arm RETRACTS ⬅️     |
| Stop    | ایست (Ist)  | Everything STOPS ⏹️ |

---

## 🧠 How It Works
