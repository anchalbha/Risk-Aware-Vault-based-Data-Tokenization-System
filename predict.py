import pickle
import os
from tokenizer import tokenize

# Attempt to load the model assets into memory when the module starts.
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    print(f"Warning: ML assets ({MODEL_PATH}, {VECTORIZER_PATH}) not found. Please run train_model.py")

def predict_risk(input_text):
    """Feeds input text into vectorizer and model to extract risk probability classification."""
    if not model or not vectorizer:
        raise RuntimeError("ML Pipeline not found.")
        
    input_vectorized = vectorizer.transform([input_text])
    predicted_risk = model.predict(input_vectorized)[0]
    return predicted_risk

def process_text(input_text):
    """
    Full processing architecture combining the ML Prediction phase and the Vault Tokenizer.
    """
    # 1. Classify text sensitivity
    risk_level = predict_risk(input_text)
    
    # 2. Tokenize text appropriately
    tokenized_output = tokenize(input_text, risk_level)
    
    return risk_level, tokenized_output

if __name__ == "__main__":
    if model and vectorizer:
        print("Model and text vectorizer loaded securely.")
        print("Testing Advanced ML Prediction Pipeline:\n")
        
        # Test edge cases simulating varying PII risk levels
        test_samples = [
            "123-456-7890",                      # Expected: HIGH
            "4111-2222-3333-4444",               # Expected: HIGH
            "researcher@university.edu",         # Expected: MEDIUM
            "john_doe_99",                       # Expected: MEDIUM
            "This is a normal research note.",   # Expected: LOW
            "My Aadhaar is 5555 1234 9876",      # Expected: HIGH
        ]
        
        width = 85
        print("-" * width)
        print(f"{'INPUT TEXT':<30} | {'RISK':<8} | {'TOKENIZED RESULT'}")
        print("-" * width)
        
        for text in test_samples:
            risk, token = process_text(text)
            print(f"{text:<30} | {risk:<8} | {token}")
        print("-" * width)
