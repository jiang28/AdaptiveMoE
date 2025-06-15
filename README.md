# 🌧️ Seeing Storms Ahead: Knowledge-Guided Adaptive Mixture of Experts for Precipitation Prediction [Applications]

> **Status**: Under Review  
> **Conference Track**: Applications Track  
> **Paper Title**: *Seeing Storms Ahead: Knowledge-Guided Adaptive Mixture of Experts for Precipitation Prediction*

---

## 🔍 Overview

Accurate precipitation forecasting is essential for agriculture, disaster management, and sustainable planning. However, predicting rainfall remains a significant challenge due to the complex dynamics of the climate system and the heterogeneous nature of observational data sources such as radar, satellite imagery, and surface measurements.

In this project, we propose a **Knowledge-Guided Adaptive Mixture of Experts (MoE)** framework specifically designed for precipitation rate prediction using multimodal climate data. This repository hosts the codebase, curated dataset, pretrained models, and an interactive web-based visualization tool.

---

## 🧠 Key Contributions

- **Adaptive Mixture of Experts (MoE):**  
  A modular deep learning model where each expert specializes in a different data modality or spatio-temporal pattern. A dynamic router learns to assign input samples to appropriate experts.

- **Knowledge-Guided Design:**  
  The data partitioning and expert specialization are informed by climate science principles, enabling the model to better capture domain-specific behavior.

- **Interactive Web-Based Tool:**  
  A lightweight browser application that allows users to explore spatial-temporal climate patterns from Hurricane Ian (2022), supporting practical decision-making.

- **Public Dataset and Models:**  
  A high-resolution gridded climate dataset and pretrained models are provided for reproducibility and benchmarking.

---

## 🧪 Dataset

- **Region:** South Florida  
- **Time Frame:** 2022/09/23 00:00 – 2022/10/02 00:00 (Hurricane Ian)  
- **Resolution:** 100×100 grid, 3km × 3km per cell, 216 hourly timestamps  
- **Modalities Included:**
  - Precipitation rate
  - Cloud cover
  - Wind speed and direction
  - Surface temperature
  - Humidity and dew point
  - Brightness temperature (simulated satellite)
  - Pressure levels (cloud base/top)
  - Total accumulated precipitation

For full details, visit the [`Data/`](./Data) folder or refer to our [dataset README](./Data/README.md).

---

## 🚀 Getting Started

### 🔧 Requirements
- Python 3.8+
- PyTorch >= 1.10
- NumPy, Pandas, Matplotlib
- Flask (for web interface)
- Leaflet.js (frontend map)


