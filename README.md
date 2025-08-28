
# 🧠 Attention Detection in Video Frames Using Deep Learning

This project presents a research pipeline for detecting **human attention** from video frames using computer vision and deep learning. The system processes videos, labels extracted frames using heuristics or manual annotation, trains models based on these labels, and predicts attention levels on new video content.



---

## 📁 Project Structure

```
.
├── Dataset_LBL.ipynb                  # Preprocessing & heuristic labeling notebook
├── HL_ModelDevelopment.ipynb         # Model training using Human-Labeled data
├── CL_ModelDevelopment.ipynb         # Model training using Computer-Labeled data
├── Proposed_Application.ipynb        # End-to-end video analysis pipeline
├── all_attention_results.json        # Heuristic labeling result storage
├── HL_all_attention_results.csv      # Final dataset used for HL model training
├── All_extracted_frames.zip          # Raw dataset (video frames)
├── video2.mp4                        # Input video for final application
├── models/                           # Contains trained model weights
├── visualized_results/               # Prediction + CAM visualization output
```

---

## 🧪 Project Overview

This research project explores **binary classification** of attention (`Attentive` or `Not Attentive`) in humans from static images extracted from video streams.

### 🔍 Key Steps:
1. **Dataset Preparation** using heuristics and manual labeling.
2. **Deep Learning Model Training** using CNN architecture and pre-trained models like RESNET18 & VGG19.
3. **Video Processing Pipeline** to make video-level predictions.

---

## 🛠️ Environment Setup

This project was developed in **Anaconda Navigator** with **Jupyter Notebook** interface on **Windows**.

You can also download all project files (prepared dataset, extracted frames, trained models, and results) from this Google Drive folder: [Click Here to Download](https://drive.google.com/drive/folders/1Z2xrf3_nkEdeDJGBDD118uUU5PdK7BLJ?usp=drive_link)

### Required Libraries

Install the following libraries in a Conda environment with CUDA support:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib torchcam opencv-python mediapipe deepface tensorflow
```

Refer to this [Medium guide](https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef) to configure PyTorch with CUDA.

---

## 📂 Dataset and Preprocessing

### 🔹 Step 1: Extract Frames

Unzip the dataset:

```
All_extracted_frames.zip
```

Place the extracted folder in the same directory where `Dataset_LBL.ipynb` expects it.

You can also download the prepared dataset directly from Google Drive: [Click Here to Download prepared Dataset](https://drive.google.com/file/d/15m80hDRqcxGaInznH8pF02JnKE3AkCcE/view?usp=drive_link)

### 🔹 Step 2: Label the Dataset

Run `Dataset_LBL.ipynb` to:
- Apply computer vision techniques (using Mediapipe, OpenCV, etc.)
- Label frames using heuristic logic
- Save outputs to:
  - `all_attention_results.json` (intermediate features)
  - `HL_all_attention_results.csv` (final annotated dataset for training)

---

## 🧠 Model Training

### CL_ModelDevelopment.ipynb (Computer-Labeled)
- Trains CustomCNN, ResNet18, VGG19 model on data labeled by algorithm.
- Fast prototyping and experimentation.

### HL_ModelDevelopment.ipynb (Human-Labeled)
- Trains CustomCNN, ResNet18, VGG19 on manually verified labels from team members.
- Model offers better generalization and accuracy.

#### 🔒 Skip Training
Trained model weights are stored in the `models/` directory. You can load them directly to skip retraining.
[Click Here to Download .pth file of the train model](https://drive.google.com/drive/folders/1ByZNkTHDyd8ONBN-TZwNza6fRbwj7RCt?usp=drive_link)

---

## 🎥 Video Prediction Pipeline

### Proposed_Application.ipynb
- Accepts a video file (`video1.mp4`)
- Extracts frames, loads the best trained model (HL or CL)
- Predicts attention for each frame
- Flags the **entire video** as `Attentive` or `Not Attentive` based on majority vote

### 📊 Output
- Frame-wise predictions
- Summary decision for the video
- Stored results in `visualized_results/` folder

---

## 🧾 Example Output

```
Frame_123: Attentive (0.78)
Frame_124: Not Attentive (0.32)
...
Final Video Decision: NOT ATTENTIVE (41% attentive frames)
```

---

## 📌 Key Libraries Used

- `torch`, `torchvision`: Deep learning
- `torchcam`: CAM visualizations (SmoothGradCAM++)
- `opencv-python`: Frame extraction and video handling
- `mediapipe`: Face and pose detection
- `deepface`: Facial feature analysis
- `matplotlib`: Visualization

---

## 📈 Potential Improvements

- Use transformer-based vision models (e.g., ViT)
- Add temporal analysis with 3D CNN or LSTM
- Improve labeling automation using gaze tracking
- Deploy as a web or real-time desktop application

---

## 👨‍🔬 Authors

**Ayaz Marediya - amarediya@algomau.ca**,  
**Divy Goswami - digoswami@algomau.ca**,
**Kartik Prajapati - karprajapati@algomau.ca**,
**Krish Patel - krishkpatel@algomau.ca**,
**Parth Sathiya - psathiya@algomau.ca**,
**Prof. Rashid Hussain Khokhar - rashid.khokhar@algomau.ca**,

---

## 📄 License

This research project is free to use under the MIT License.

---

## 📌 Acknowledgements

Thanks to our team for manual annotation and support. This work was inspired by the growing need to analyze attentiveness in online education, virtual meetings, and safety-critical environments.
