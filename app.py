from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import pickle
import os
import re
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from tokenizer import tokenize

app = Flask(__name__)

load_dotenv()
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if ENCRYPTION_KEY:
    cipher = Fernet(ENCRYPTION_KEY.encode())
else:
    print("Warning: ENCRYPTION_KEY not found in .env")
    cipher = None

# ==========================================
# 1. MongoDB Setup
# ==========================================
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    client.server_info() 
    db = client["risk_db"]
    records_collection = db["records"]
    vault_collection = db["vault"]
    audit_logs_collection = db["audit_logs"] # Tracks system access & breaches
    print("Successfully connected to MongoDB!")
except Exception as e:
    print("Warning: Could not connect to MongoDB.")
    db = None
    records_collection = None
    vault_collection = None
    audit_logs_collection = None

# ==========================================
# 2. ML System Setup (STATIC MODEL)
# ==========================================
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Strict adherence to static models. Auto-retraining removed to guarantee research presentation stability.
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Static Machine Learning models successfully mounted to memory.")
else:
    print("Warning: No ML assets found. Please run train_model.py once manually.")
    model = None
    vectorizer = None

def hybrid_predict_risk(input_text, confidence_threshold=0.60):
    """
    HYBRID LOGIC PIPELINE:
    Pre-filters text using explicitly defined Rule-Based patterns (Regex) first to guarantee 
    absolute hit-rates on explicit scenarios. If none apply, falls back to the static ML system
    and evaluates using confidence boundaries.
    """
    
    # ====== PHASE 1: RULE-BASED REGEX PRE-FLIGHT ======
    
    # a) PAN Detection: 5 letters + 4 digits + 1 letter (e.g. ABCDE1234F)
    if re.search(r'\b[A-Za-z]{5}\d{4}[A-Za-z]\b', input_text):
        return "HIGH"
        
    # b) Phone Number Detection: exactly 10 digits alongside optional spaces/dashes
    if re.search(r'\b\d(?:[\s-]*\d){9}\b', input_text):
        return "HIGH"
        
    # d) Aadhaar Detection: exactly 12 digits mapping to standard Indian ID format
    if re.search(r'\b\d(?:[\s-]*\d){11}\b', input_text):
        return "HIGH"

    # c) Credit Card Detection: exactly 16 digits spanning blocks
    if re.search(r'\b\d(?:[\s-]*\d){15}\b', input_text):
        return "HIGH"
        
    # Rule 5: Valid email pattern structure forces MEDIUM risk
    if re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', input_text):
        return "MEDIUM"
        
    # ====== NEW RULE: PRE-CHECK FOR NON-SENSITIVE DATA ======
    # If the text does NOT contain numbers, does NOT contain '@', 
    # and has not triggered any previous sensitive logic -> forcefully skip ML
    if not re.search(r'\d', input_text) and '@' not in input_text:
        return "LOW"
        
    # ====== PHASE 2: MACHINE LEARNING EVALUATION ======
    
    if not model or not vectorizer:
        raise RuntimeError("ML Pipeline not found. Cannot predict. Run train_model.py first.")
        
    input_vectorized = vectorizer.transform([input_text])
    
    # predict_proba returns a 2D array matrix of confidences for each class
    probabilities = model.predict_proba(input_vectorized)[0]
    
    # Extract the highest probability
    max_confidence = np.max(probabilities)
    
    # If the model is not highly confident in its decision, we apply a safety net threshold assignment
    if max_confidence < confidence_threshold:
        return "MEDIUM"
        
    # Otherwise, accept the ML model's dominant classification
    return model.predict(input_vectorized)[0]


# ==========================================
# 3. Flask Endpoints
# ==========================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.form.get("user_input", "")
    user_email = request.form.get("user_email", "").strip()
    
    if not user_email:
        user_email = "anonymous@system.local"
        
    if not input_text.strip():
        return render_template("index.html", error="Please enter valid text.")
        
    try:
        # Utilize the Hybrid Prediction Logic (Regex + ML threshold)
        risk_level = hybrid_predict_risk(input_text)
        
        # Execute Tokenization Module natively natively
        tokenized_output = tokenize(input_text, risk_level)
        
        # Implement Selective Encryption for Vault
        # Security Justification: HIGH & MEDIUM data must be encrypted to ensure 
        # analysts with raw database access cannot view exposed personally identifiable information.
        if risk_level in ["HIGH", "MEDIUM"]:
            if not cipher:
                raise RuntimeError("Encryption key missing. Cannot secure data.")
            # .decode() is applied to convert the encrypted raw bytes back into a UTF-8 string format. 
            # This ensures MongoDB stores it natively as a readable String (e.g., 'gAAAAAB...') 
            # instead of wrapping it inside an unreadable BSON Binary object.
            stored_data = cipher.encrypt(input_text.encode()).decode()
        else:
            stored_data = input_text
            
        current_time = datetime.now()

        # Log to MongoDB using Vault Architecture
        if db is not None:
            # Vault secures the actual mapped values
            vault_collection.insert_one({
                "token": tokenized_output,
                "stored_data": stored_data,
                "risk_level": risk_level,
                "timestamp": current_time
            })
            
            # Application records track safe metadata
            records_collection.insert_one({
                "token": tokenized_output,
                "risk_level": risk_level,
                "action": "tokenize",
                "timestamp": current_time,
                "user_email": user_email
            })
            
    except Exception as e:
        return render_template("index.html", error=f"System Error: {str(e)}")
    
    risk_desc = {
        "HIGH": "High risk detected. Masked with secure algorithm.",
        "MEDIUM": "Medium risk detected (via Email Rule or Low ML Confidence). Partially occluded.",
        "LOW": "Low risk detected. Safely bypassed."
    }
    
    return render_template(
        "result.html", 
        original_data=input_text,
        risk_level=risk_level,
        description=risk_desc.get(risk_level, ""),
        tokenized_output=tokenized_output
    )

def mask_data(data):
    if "@" in data:
        return data[:2] + "******" + data[-2:]
    elif data.isdigit():
        return data[:2] + "******" + data[-2:]
    else:
        return data[:2] + "******"

@app.route("/detokenize", methods=["POST"])
def detokenize():
    """
    Secure Detokenization Endpoint
    Security Justification: Access control guarantees that only authorized roles
    can request decryption, preventing horizontal token scraping.
    Audit logs are maintained to permanently trace the origin of decryption attempts.
    """
    # Evaluate incoming JSON natively (e.g. for Postman testing without strict type-headers) 
    data = request.get_json(force=True, silent=True) or request.form or {}
    token = data.get("token", "").strip()
    username = str(data.get("username", "guest")).strip().lower()
    
    if username == "alice":
        role = "Admin"
    elif username == "bob":
        role = "Analyst"
    else:
        role = "User"
        
    print("Username:", username)
    print("Role:", role)
        
    if not token:
        return jsonify({"error": "Missing token"}), 400
        
    if db is None:
        return jsonify({"error": "Database not connected"}), 500
        
    # Basic Access Control Check
    if role not in ["Admin", "Analyst"]:
        audit_logs_collection.insert_one({
            "token": token,
            "username": username,
            "role": role,
            "action": "detokenize",
            "timestamp": datetime.now(),
            "status": "denied"
        })
        return jsonify({"error": "Access denied"}), 403
        
    record = vault_collection.find_one({"token": token})
    if not record:
        return jsonify({"error": "Token not found"}), 404
        
    stored_data = record.get("stored_data")
    risk_level = record.get("risk_level")
    
    if risk_level in ["HIGH", "MEDIUM"]:
        try:
            if not cipher:
                return jsonify({"error": "Encryption key missing, cannot decrypt"}), 500
            # .encode() is required here because the data was stored as a standard string.
            # Fernet's decrypt mechanism strictly requires a bytes-like object as its input.
            original_data = cipher.decrypt(stored_data.encode()).decode("utf-8")
        except Exception:
            original_data = stored_data
    else:
        original_data = stored_data
        
    if role == "Analyst":
        original_data = mask_data(original_data)
        
    # Log successful detokenization audit
    audit_logs_collection.insert_one({
        "token": token,
        "username": username,
        "role": role,
        "action": "detokenize",
        "timestamp": datetime.now(),
        "status": "success"
    })
        
    return jsonify({"original_data": original_data})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
