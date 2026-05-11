import os
import pickle
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = 'saved_model.pkl'

def train_and_save_model():
    print("Training new model...")
    # 1. Create a small manually crafted dataset
    data = [
        # HIGH Risk (Financial, phone numbers)
        ("123-456-7890", "HIGH"),
        ("9876543210", "HIGH"),
        ("My phone number is 555-0192", "HIGH"),
        ("Account balance is $50,000", "HIGH"),
        ("Credit card 4111-2222-3333-4444", "HIGH"),
        ("Bank routing number 123456789", "HIGH"),
        
        # MEDIUM Risk (Emails, personal identifiers)
        ("test@example.com", "MEDIUM"),
        ("Contact me at user@domain.com", "MEDIUM"),
        ("admin@company.org", "MEDIUM"),
        ("Here is my email: private@mail.com", "MEDIUM"),
        
        # LOW Risk (Normal text)
        ("Hello, how are you?", "LOW"),
        ("The weather is nice today.", "LOW"),
        ("I love learning Python.", "LOW"),
        ("This is a random test sentence.", "LOW"),
        ("Apples are red.", "LOW")
    ]

    # Separate features (X) and labels (y)
    X_train = [item[0] for item in data]
    y_train = [item[1] for item in data]

    # 2. Vectorize the text using TF-IDF (character n-grams work better for formats like phone/emails with small datasets)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # 3. Train a Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    # 4. Save the model and vectorizer using pickle
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

    print(f"Model trained and saved successfully as '{MODEL_PATH}'.")

def load_system():
    """Loads the trained model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
        
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['vectorizer']

def predict_risk(input_text, model, vectorizer):
    """Predicts the risk level of the input text."""
    # Convert input using the loaded vectorizer
    input_vectorized = vectorizer.transform([input_text])
    
    # Predict using the loaded model
    prediction = model.predict(input_vectorized)[0]
    return prediction

def tokenize_data(input_text, risk_level):
    """Tokenizes data based on predicted risk level."""
    if risk_level == "HIGH":
        # Generate a secure random token (completely masks original data)
        token = "TOK_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        return token
    elif risk_level == "MEDIUM":
        # Partially masked token (shows first/last characters)
        if len(input_text) > 4:
            return input_text[:2] + "****" + input_text[-2:]
        return "****"
    else: # LOW
        # Simple token, leaves text mostly as-is with a tag
        return f"[SAFE] {input_text}"

if __name__ == "__main__":
    # If this file is run directly, train the model and run some tests
    train_and_save_model()
    test_model, test_vectorizer = load_system()
    
    # Example test cases
    test_inputs = ["555-123-4567", "hello world", "person@email.com"]
    print("\n--- Running test cases ---")
    for test in test_inputs:
        risk = predict_risk(test, test_model, test_vectorizer)
        tokenized = tokenize_data(test, risk)
        print(f"Input: {test} | Risk: {risk} | Tokenized: {tokenized}")
