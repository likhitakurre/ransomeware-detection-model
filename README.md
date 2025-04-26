# Ransomware Detection and Behavior Analysis System 🚀

This project is a full-stack machine learning application designed to detect ransomware and analyze system behaviors using a hybrid approach of supervised learning, zero-shot classification, and RAG (Retrieval-Augmented Generation) insights.

### ✨ Key Features:
- **Random Forest Classifier** trained to distinguish between benign and ransomware activities.
- **Zero-Shot Learning (ZSL)** using a DeBERTa-v3 model to classify behaviors as ransomware, trojan, spyware, or benign based on system activity descriptions.
- **FAISS Knowledge Base** powered by Sentence Transformers and LangChain to retrieve detailed behavior insights.
- **Lightweight LLM (Phi-2)** generates detailed, multi-line explanations to assist cybersecurity analysis.
- **Flask API** with endpoints for:
  - `/predict` — Make predictions based on system features and activity.
  - `/update_knowledge_base` — Dynamically update the knowledge base with new threat behaviors.
- **Logging** with detailed tracking via `app.log`.

### 🛠️ Tech Stack:
- Python (Flask, Scikit-learn, Transformers, LangChain, FAISS, OpenCV, Pandas)
- Machine Learning (Random Forest, Zero-Shot Classification)
- Natural Language Processing (Phi-2, Hugging Face models)
- FAISS Vector Store for fast similarity search
- CORS-enabled REST API

### ⚡ How it works:
- Feature-based prediction via a trained RandomForest model.
- Context-aware system behavior analysis with zero-shot classification.
- RAG-based multi-line threat insight generation for ransomware activities.

---

Would you also like a **one-line tagline** version for the repo (for the very top of the README)? 🌟

