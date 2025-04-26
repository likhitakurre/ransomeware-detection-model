import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch
import faiss

# Configure logging
logging.basicConfig(
    filename="app.log",  # Log to file
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Validate command-line argument
if len(sys.argv) < 2:
    logging.error("ERROR: Please provide the dataset folder path.")
    sys.exit(1)

dataset_folder = sys.argv[1]
if not os.path.exists(dataset_folder):
    logging.error(f"ERROR: Folder '{dataset_folder}' not found.")
    sys.exit(1)

dataset_path = os.path.join(dataset_folder, "data_file.csv")
if not os.path.exists(dataset_path):
    logging.error(f"ERROR: Dataset file '{dataset_path}' not found.")
    sys.exit(1)

# Load dataset
logging.info("Loading dataset...")
data = pd.read_csv(dataset_path)
X = data.drop(columns=['Benign', 'FileName', 'md5Hash'])
y = data['Benign']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
logging.info("Training RandomForestClassifier...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.10f}")
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
logging.info(f"Model accuracy: {accuracy:.4f}")
logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save model & scaler
joblib.dump(rf_model, 'ransomware_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
logging.info("Model and scaler saved successfully.")

# Load language model
# ✅ Load lightweight LLM: Phi-2
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "microsoft/phi-2"
logging.info(f"Loading LLM: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)  # Use float32 for CPU

llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
logging.info("Phi-2 model loaded successfully.")



# FAISS knowledge base setup
knowledge_base_path = "ransomware_knowledge_base"
knowledge_file = os.path.join(dataset_folder, "ransomware_behavior.txt")

def build_faiss_knowledge_base(data_path, save_path):
    if not os.path.exists(data_path):
        logging.error(f"ERROR: Knowledge base file '{data_path}' not found.")
        return
    
    logging.info("Building FAISS knowledge base...")
    loader = TextLoader(data_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    vectorstore.save_local(save_path)
    logging.info(f"Knowledge base saved at {save_path}")

if not os.path.exists(knowledge_base_path):
    build_faiss_knowledge_base(knowledge_file, knowledge_base_path)

faiss.omp_set_num_threads(16)  

vectorstore = FAISS.load_local(
    knowledge_base_path,
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True  
)

# Retrieve ransomware insights

def retrieve_ransomware_info(query):
    logging.info(f"Retrieving ransomware insights for query: {query}")
    
    # Retrieve more chunks for better context
    docs = vectorstore.similarity_search(query, k=5)
    retrieved_texts = "\n\n".join([doc.page_content for doc in docs])

    # Better prompt for longer, label-specific RAG insights
    prompt = f"""
You are a cybersecurity expert assistant. Analyze the following system behavior description and the retrieved knowledge snippets. Provide a detailed, multi-line explanation to help identify whether this behavior matches ransomware, trojan, spyware, or benign software.

🧪 System Activity:
{query}

📚 Retrieved Knowledge:
{retrieved_texts}

🧠 Provide detailed multi-line insights below:
"""

    # Generate insights using your LLM
    result = llm(prompt)[0]["generated_text"]

    # Clean output to return only the final insight
    return result.split("🧠 Provide detailed multi-line insights below:")[-1].strip()




# Flask API
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.info(f"Received prediction request: {data}")
        
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = rf_model.predict(features_scaled)
        
        system_activity = data.get("system_activity", "")
        labels = ["ransomware", "benign", "trojan", "spyware"]
        zsl_pipeline = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device=-1  # Ensures CPU usage
)

        zsl_result = zsl_pipeline(system_activity, labels,truncation=True,max_length=512)
        zsl_prediction = zsl_result['labels'][0]
        
        
        rag_insights = retrieve_ransomware_info(system_activity) if zsl_prediction == "ransomware" else "No insights needed."

        
        logging.info(f"Prediction result: ML={prediction[0]}, ZSL={zsl_prediction}")
        return jsonify({'ml_prediction': int(prediction[0]), 'zsl_prediction': zsl_prediction, 'rag_insights': rag_insights})
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

@app.route('/update_knowledge_base', methods=['POST'])
def update_knowledge_base():
    try:
        new_data = request.json.get("new_data", [])
        if not new_data:
            return jsonify({"error": "No data provided for update"})
        
        new_docs = [Document(page_content=text) for text in new_data]
        vectorstore.add_documents(new_docs)
        vectorstore.save_local(knowledge_base_path)
        
        logging.info("Knowledge base updated successfully.")
        return jsonify({"message": "Knowledge base updated successfully"})
    except Exception as e:
        logging.error(f"Error updating knowledge base: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    logging.info("Starting Flask API...")
    app.run(debug=False)
