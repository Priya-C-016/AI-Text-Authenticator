# 🧠 AI Text Authenticator (LLM-Based)
A lightweight, production-ready **AI vs Human text classification system** built using **LLM embeddings (SentenceTransformer – MiniLM-L6-v2)** and a **Logistic Regression classifier**.  
It detects whether a piece of text is **AI-generated** or **Human-written** using semantic understanding.

This project includes:
- LLM Embedding Model (MiniLM)
- Machine Learning Classifier
- Training Pipeline (train.py)
- Streamlit Web App (app.py)
- Clean architecture + modular codebase

---

## 🚀 Features
- 🔍 **Detect AI vs Human-generated text**
- ⚡ **Fast inference (MiniLM is only 22MB)**
- 📦 **Lightweight model suitable for laptops**
- 🧠 **Uses state-of-the-art transformer embeddings**
- 📊 **Evaluation metrics: Accuracy, Precision, Recall, F1**
- 🌐 **Streamlit UI for easy interaction**
- 🗂️ **Clean ML pipeline structure**


---

## 🧪 Model Pipeline Overview

1. Load dataset  
2. Convert text → LLM embeddings using **MiniLM-L6-v2**  
3. Train a **Logistic Regression** classifier  
4. Evaluate (Accuracy, Precision, Recall, F1)  
5. Save trained model (`model.pkl`)  
6. Use Streamlit app for inference  

---

## 📊 Model Performance  
*(Example — replace with your actual values from training)*

| Metric      | Score |
|-------------|--------|
| Accuracy    | 0.92   |
| Precision   | 0.91   |
| Recall      | 0.90   |
| F1 Score    | 0.90   |

---

## 🔧 Installation

### **1️⃣ Clone the repo**
```bash
git clone https://github.com/Priya-C-016/AI-Text-Authenticator-LLM-Based.git
cd AI-Text-Authenticator-LLM-Based
```
