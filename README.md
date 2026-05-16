# 🛡️ HYBRID-AI-POWERED-SEMANTIC-ANOMALY-DETECTOR
### *Uniting Macro-Statistical DNA with Generative Semantic Reasoning*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange?style=for-the-badge&logo=google-gemini)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Hackathon_Prototype-success?style=for-the-badge)

---

## 📖 Table of Contents
1. [Vision & Motivation](#-vision--motivation)
2. [Technical Architecture](#-technical-architecture)
3. [The Hybrid Triad Engine](#-the-hybrid-triad-engine)
4. [Gemini API Integration](#-gemini-api-integration)
5. [Installation & Usage](#-installation--usage)
6. [Challenges Overcome](#-challenges-overcome)
7. [Future Roadmap](#-future-roadmap)

---

## 🎯 Vision & Motivation
Data quality is the silent killer of AI systems. Traditional anomaly detection often focuses purely on mathematical outliers, missing the **"Logical Traps"**—data that looks statistically normal but is semantically impossible (e.g., a "New" car with 500,000 km, or a 25-year-old in 2nd Grade). 

**HYBRID-AI-POWERED-SEMANTIC-ANOMALY-DETECTOR** solves this by creating a "Statistical DNA Profile" of datasets. This allows a Large Language Model (LLM) to audit complex logic without the massive overhead of processing millions of rows individually.

---

## 🏗 Technical Architecture

HYBRID-AI-POWERED-SEMANTIC-ANOMALY-DETECTOR operates on a multi-tiered validation pipeline:
1.  **Macro-Profiler:** Computes distribution metrics (Skewness, Kurtosis, Z-Scores) to create a "DNA Profile."
2.  **ML Engine:** Executes an `Isolation Forest` (Unsupervised ML) to identify row-level density anomalies.
3.  **Semantic Brain:** Gemini 2.5 Flash audits the high-level distribution for logical contradictions.

---

## ⚡ The Hybrid Triad Engine

### 📊 Layer 1: Local Statistical Engine
We use **Interquartile Range (IQR)** and **Z-Score analysis** to find immediate formatting and range violations. This layer is "Zero-Cost" and runs entirely in local memory to save API quota.

### 🤖 Layer 2: Machine Learning Engine
An **Isolation Forest** model is trained on the fly to detect multi-dimensional outliers. This handles the "Normal" anomalies like price spikes or unusual data clusters.

### 🧠 Layer 3: Semantic AI Auditor
The **Gemini 2.5 Flash** engine performs the final logical audit. By analyzing the "DNA Profile," the AI identifies domain-specific impossibilities (e.g., biological constraints, financial contradictions) that math cannot see.

---

## 🔌 Gemini API Integration

### 🧠 Chain-of-Thought (CoT) Reasoning
We implemented a **CoT prompt strategy** that forces the AI to provide a "Logical Analysis" before flagging an anomaly. This ensures high-fidelity results and minimizes hallucinations.

### 💰 Token Optimization
* **Macro-Auditing:** By sending statistical summaries instead of raw rows, we reduced token consumption by **99.9%**.
* **Privacy:** Raw user data never leaves the local environment; only mathematical summaries are audited by the AI.

---

## 📈 Dashboard & Visuals
SentinelAI generates an interactive visual suite:
* **Ensemble Convergence Matrix:** A pseudo-confusion matrix showing the agreement between Math and AI.
* **Feature Correlation Heatmap:** An "X-Ray" of the dataset to find redundant or overlapping features.
* **Health Summary:** A breakdown of "Clean" vs "Corrupted" columns.

---

## 🛠 Installation & Usage

### 1. Install Dependencies
```bash
pip install google-genai pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Sentinel Agent
```python
from SentinelAI import ImprovedUniversalAnomalyAgent

# Initialize with your API Key
agent = ImprovedUniversalAnomalyAgent(api_key="YOUR_GEMINI_API_KEY")

# Analyze via CSV path or raw string
agent.analyze("/kaggle/input/your-dataset.csv")
```

---

## 🚧 Challenges Overcome
* **Handling Big Data:** Solved by moving from row-by-row scanning to macro-statistical profiling.
* **JSON Parsing:** Used Gemini's native JSON mode to ensure 100% reliable integration with Python logic.

---

## 🗺 Future Roadmap
- [ ] **Auto-Cleaning:** AI-generated scripts to fix the detected anomalies automatically.
- [ ] **Streaming Support:** Real-time auditing for live API endpoints.
- [ ] **Multi-Model Support:** Integration with Gemini 2.5 Pro for deep-dive forensic auditing.

---

## 👨‍💻 Author

**Pawan Ranakoti**
**Tamanna Bhatt**

---
*Developed for the Gemini Hackathon 2026.*

