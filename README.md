# 🧬 Skin Lesion Classification using Deep Learning (HAM10000 Dataset)

## 🩺 Project Overview
This project focuses on **automated skin lesion classification** using the **HAM10000 dataset**, leveraging multiple deep learning architectures such as **DenseNet121**, **DenseNet169**, **CNN**, and hybrid models.  
The aim is to develop a **highly accurate and interpretable system** capable of classifying skin lesions into seven diagnostic categories.

---

## 🎯 Objective
Build a model that can:
- Accurately classify multiple types of skin lesions from dermatoscopic images.
- Handle **dataset imbalance** effectively.
- Provide **explainable predictions** through **Grad-CAM visualizations**.

---

## 📊 Dataset Details — HAM10000
The **HAM10000 ("Human Against Machine with 10000 training images")** dataset includes **10,015 dermatoscopic images** of common pigmented skin lesions.

### 🧩 Class Distribution:
| Label | Class Name | Approx. Samples |
|:------|:------------|----------------:|
| 0 | Actinic keratoses (akiec) | ~327 |
| 1 | Basal cell carcinoma (bcc) | ~514 |
| 2 | Benign keratosis-like lesions (bkl) | ~1,099 |
| 3 | Dermatofibroma (df) | ~115 |
| 4 | Melanocytic nevi (nv) | ~6,700 |
| 5 | Vascular lesions (vasc) | ~142 |
| 6 | Melanoma (mel) | ~1,113 |

🧠 **Observation:**  
Severe class imbalance — the *nv* class dominates, while *df* and *vasc* are rare.

---

## 🧮 Handling Class Imbalance
To counter imbalance and improve model robustness:

- **Data Augmentation:** rotations, zooms, flips, and shifts (especially on minority classes).  
- **Stratified Splitting:** ensured balanced representation in train/test sets.  
- **Weighted Loss:** gave higher importance to underrepresented lesion types.

---

## ⚙️ Models Implemented

### 1️⃣ DenseNet121  
Fine-tuned ImageNet-pretrained model.  
✅ **Accuracy:** 93.4%

### 2️⃣ DenseNet169  
Deeper model variant with slightly improved stability.  
✅ **Accuracy:** 93.7%

### 3️⃣ Custom CNN  
Baseline convolutional model trained from scratch.  
✅ **Accuracy:** 93.7%

### 4️⃣ CNN + DenseNet121 (Hybrid Model) — 🏆 *Best Performer*  
Combines CNN feature extraction and DenseNet transfer learning.  
✅ **Best Accuracy:** **93.8%**

| Class | Accuracy |
|:------|:----------|
| 0 | 95.09% |
| 1 | 93.14% |
| 2 | 87.14% |
| 3 | 99.80% |
| 4 | 82.29% |
| 5 | 93.44% |
| 6 | 100.00% |

### 5️⃣ DenseNet121 + EfficientNetB3 (Hybrid)
High-capacity model but computationally heavy, risk of overfitting.  
✅ **Accuracy:** 88.2%

---

## 🧠 Grad-CAM Visualization for Model Interpretability

### 🔍 Purpose
To make the model’s predictions explainable and verify whether it focuses on actual lesion areas instead of background noise.

### ⚙️ Implementation
- Loaded the **trained CNN + DenseNet121 hybrid model (`best_model.h5`)** without re-training.  
- Identified the **final convolutional layer** to compute class-specific gradients.  
- Generated **Grad-CAM heatmaps** for each of the 7 lesion classes:  
  `['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']`.

### 🖼️ Example Outputs
Each Grad-CAM result shows three panels:
1. **Original Image**
2. **Grad-CAM Heatmap**
3. **Superimposed Image (Original + Heatmap)**

Below are some representative outputs from the `/gradcam_examples/` directory:

| Lesion Class | Grad-CAM Visualization |
|:--------------|:-----------------------|
| Melanoma (mel) | ![mel_gradcam](gradcam_examples/mel_gradcam.jpg) |
| Basal Cell Carcinoma (bcc) | ![bcc_gradcam](gradcam_examples/bcc_gradcam.jpg) |
| Nevus (nv) | ![nv_gradcam](gradcam_examples/nv_gradcam.jpg) |
| Dermatofibroma (df) | ![df_gradcam](gradcam_examples/df_gradcam.jpg) |
| Benign keratosis-like lesions (bkl) | ![bkl_gradcam](gradcam_examples/bkl_gradcam.jpg) | 
| Actinic keratoses (akiec) | ![akiec_gradcam](gradcam_examples/akiec_gradcam.jpg) |
| Vascular lesions (vasc) | ![vasc_gradcam](gradcam_examples/vasc_gradcam.jpg) |
🧩 The red-hot regions in these images correspond to **areas of highest model attention** — confirming the model focuses on lesion regions rather than background skin.

---

### 📈 Benefits of Grad-CAM
- Enhances **transparency** and **trust** in medical AI systems.  
- Enables **qualitative validation** of predictions.  
- Identifies possible **model bias or failure cases**.  
- Makes the project publication-ready for explainable AI (XAI) applications.

---

## 🧩 Tools & Libraries
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **Grad-CAM Visualization Tools**

---

## 📂 Repository Structure
```text
├── HAM10000_skin_classification.ipynb # Full training and evaluation notebook
├── best_model.h5 # Trained hybrid CNN + DenseNet121 model
├── gradcam_examples/ # Grad-CAM output images
│ ├── mel_gradcam.jpg
│ ├── bcc_gradcam.jpg
│ ├── nv_gradcam.jpg
│ └── df_gradcam.jpg
├── requirements.txt
└── README.md
```


---

## 🚀 Results Summary

| Model | Accuracy | Highlight |
|:------|:----------|:-----------|
| DenseNet121 | 93.4% | Strong baseline |
| DenseNet169 | 93.7% | Deeper, stable |
| CNN | 93.7% | Lightweight |
| CNN + DenseNet121 | **93.8%** | 🏆 Best performer |
| DenseNet121 + EfficientNetB3 | 88.2% | Overfit tendency |

---

## 📜 License
Licensed under the **MIT License** — you may reuse or modify this work with attribution.

---

## 🙌 Acknowledgments
- Dataset: [HAM10000 — Human Against Machine with 10000 Training Images](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Inspiration: Building explainable AI models for **trustworthy healthcare applications**.

---

⭐ **If you find this project helpful, consider starring the repository!**
