# 🚀 Deep Learning Final Project - FPA_1

## 👥 Team Members
- Haider Shabbir Daresalamwala
- Manali Shetye
- Shreya Gupta

🔗 **GitHub Repository**: [Deep Learning Final Project](https://github.com/haiderdares/DeepLearningFinalProject)

---

## 📊 Project Overview
This project explores the application of **deep learning models** to three diverse datasets across different modalities: **medical imaging, autonomous driving, and music classification**. Each dataset presents unique challenges and opportunities to leverage Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and advanced deep learning architectures.

---

## 🧠 Dataset 1: **Alzheimer Brain Scans** 🏥

### 🔍 Problem Definition
- **Dataset:** Brain Structure Dataset
- **Goal:** Classify different anatomical structures in brain MRI scans.
- **Problem Type:** Multi-Class Classification
- **Target Labels:** Different brain structures (e.g., cerebellum, hippocampus, corpus callosum, ventricles, etc.)

### ⚠️ Challenges
- High inter-class similarity among brain structures.
- Requires **advanced image segmentation** and feature extraction.
- Variability in MRI scan quality and noise.

### 🏆 Why It’s Useful?
This dataset is valuable for **automated medical image analysis**, aiding in neuroscience research, early disease detection (**Alzheimer’s, tumors**), and surgical planning.

### 🛠️ Preprocessing & Feasibility
✔️ **Sufficient Data** → Enough samples for training and validation.
✔️ **Class Imbalance Handling** → Augmentation & resampling techniques applied.
✔️ **Preprocessing** → Contrast enhancement, noise reduction, skull stripping.
✔️ **Tensor-Ready Format** → Converted into NumPy arrays and PyTorch tensors.

🛠️ **Best Models:** CNNs & Vision Transformers (ViTs) 📈

---

## 🚗 Dataset 2: **Self-Driving Cars - Road Object Detection** 🛑

### 🔍 Problem Definition
- **Dataset:** Berkeley DeepDrive (BDD100K)
- **Goal:** Detect and classify objects in driving scenes, including vehicles, pedestrians, traffic signs, and lane markings.
- **Problem Type:** Object Detection (Supervised Learning)
- **Target Labels:** Bounding boxes for multiple object classes (cars, buses, pedestrians, traffic signs, etc.)

### ⚠️ Challenges
- **High variability** in lighting, weather conditions, and occlusions.
- Requires **real-time inference** for autonomous driving.
- Multi-object detection & tracking complexity.

### 🏆 Why It’s Useful?
This dataset is crucial for training **YOLO, Faster R-CNN, DETR, or Vision Transformers**, enabling robust perception in self-driving cars 🚗💨.

### 🛠️ Preprocessing & Feasibility
✔️ **Large-Scale Dataset** → 100,000 diverse driving scenes.
✔️ **Realistic Class Distribution** → Minor imbalances handled via augmentation.
✔️ **Preprocessing** → Annotated bounding boxes, OpenCV & TensorFlow/PyTorch processing.
✔️ **Tensor-Ready Format** → Supports real-time loading via TensorFlow/PyTorch DataLoader.

🛠️ **Best Models:** YOLOv8, Faster R-CNN, DETR, ViTs 📌

---

## 🎵 Dataset 3: **Music Genre Classification** 🎶

### 🔍 Problem Definition
- **Dataset:** GTZAN Music Genre Classification
- **Goal:** Classify music tracks into their respective genres based on audio features.
- **Problem Type:** Multi-Class Classification
- **Target Labels:** 10 music genres (Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock)

### ⚠️ Challenges
- **High intra-class similarity** → Songs within the same genre can vary widely.
- **Background noise & instrument overlap** → Extracting meaningful features is complex.
- Requires **spectrogram transformation** or feature extraction (**MFCCs, chroma, tempo**).

### 🏆 Why It’s Useful?
- Enables the development of **personalized music recommendation systems** 🎧.
- Can integrate **multi-modal features** (lyrics analysis, deep audio embeddings) for improved classification.

### 🛠️ Preprocessing & Feasibility
✔️ **Balanced Dataset** → 1,000 tracks (30-second clips) with equal class representation.
✔️ **Preprocessing** → Raw audio files converted to spectrograms using **Librosa, TensorFlow, PyTorch**.
✔️ **Tensor-Ready Format** → Transformed into waveforms, spectrograms, or feature tensors.

🛠️ **Best Models:** CNNs + Spectrograms, LSTMs, Self-Supervised Learning (SimCLR, CLIP) 🎛️

---

## 🔮 **Future Scope & Enhancements** 🚀

1️⃣ **Personalized Music Recommendations** 🎼 → Real-time, user-specific recommendations based on listening history, mood analysis, and preferences.
2️⃣ **Multi-Modal Feature Integration** 📡 → Combining **lyrics analysis, user behavior data, and deep audio embeddings** for enhanced accuracy.
3️⃣ **Advanced Segmentation for Medical Imaging** 🏥 → Using **3D CNNs & Transformers** for improved brain structure classification.
4️⃣ **Autonomous Driving with Sensor Fusion** 🚘 → Integrating **LiDAR & RADAR** with **camera data** for robust real-time object detection.

---

## 🛠️ **Tech Stack & Frameworks Used** 🖥️

- **🧠 Deep Learning:** TensorFlow, PyTorch
- **📊 Data Preprocessing:** OpenCV, NumPy, Pandas
- **🎵 Audio Processing:** Librosa, Mel Spectrograms, MFCCs
- **🚗 Computer Vision:** YOLOv8, Faster R-CNN, ViTs
- **📈 Self-Supervised Learning:** SimCLR, CLIP

---

## 📌 How to Run the Code? 🚀
1️⃣ Clone the repository:
   ```bash
   git clone https://github.com/haiderdares/DeepLearningFinalProject.git
   cd DeepLearningFinalProject
   ```
2️⃣ Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the models:
   ```bash
   python train_model.py --dataset <dataset_name>
   ```

---

## 📢 **Contributions & Contact** ✉️

💡 **Want to contribute?** Feel free to fork the repo and submit a pull request!

📩 **Questions?** Contact us via GitHub Issues or reach out to the team members.

🚀 **Happy Coding!** 🎯

