import os
import pickle
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

def get_mongodb_data():
    """
    Connects to MongoDB and extracts 'input' and 'risk' columns from records.
    """
    try:
        # serverSelectionTimeoutMS is set to fail quickly if DB is offline
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()
        db = client["risk_db"]
        collection = db["records"]
        
        # Fetch only required fields
        records = list(collection.find({}, {"_id": 0, "input": 1, "risk": 1}))
        return records
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return []

def run_retraining_pipeline():
    """
    Full Retraining Logic:
    1. Fetches MongoDB data.
    2. Cleans text, removes duplicates, validates categories.
    3. Aborts if fewer than 100 valid records.
    4. Evaluates old model vs. new model.
    5. Saves new model ONLY if accuracy improves or remains identical.
    """
    records = get_mongodb_data()
    
    if not records:
        return False, "Database Error: No data could be retrieved from MongoDB."
        
    df = pd.DataFrame(records)
    
    # 1. Pipeline Safety: Data Validation
    if 'input' not in df.columns or 'risk' not in df.columns:
        return False, "Data Error: Database collection is missing required fields."
        
    # Drop completely empty texts
    df = df[df['input'].astype(str).str.strip() != '']
    df.dropna(subset=['input', 'risk'], inplace=True)
    
    # Only keep standardized risks, dropping random anomalies
    valid_risks = ['HIGH', 'MEDIUM', 'LOW']
    df = df[df['risk'].isin(valid_risks)]
    
    # Drop duplicates to prevent model bias on repeated identical sentences
    df.drop_duplicates(subset=['input'], inplace=True)
    
    num_records = len(df)
    
    # Hard stop to prevent model degradation from extremely low data counts
    if num_records < 100:
        return False, f"Safety Abort: Not enough valid records. Found {num_records}, minimum required is 100."
        
    print(f"Starting retraining pipeline. Using {num_records} cleaned database records...")
    
    X_raw = df['input']
    y = df['risk']
    
    # 80/20 train/test split specifically on RAW data so we can compare models fairly
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    
    # 2. Retraining Module
    new_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    X_train_vec = new_vectorizer.fit_transform(X_train_raw)
    X_test_vec = new_vectorizer.transform(X_test_raw)
    
    # Logistic Regression bound to max_iter=200 natively
    new_model = LogisticRegression(max_iter=200)
    new_model.fit(X_train_vec, y_train)
    
    new_y_pred = new_model.predict(X_test_vec)
    new_accuracy = accuracy_score(y_test, new_y_pred)
    
    # 3. Performance Safety Checks
    old_accuracy = 0.0
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                old_model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                old_vectorizer = pickle.load(f)
                
            # Compute old model's accuracy securely against the NEW test dataset batch
            X_test_old_vec = old_vectorizer.transform(X_test_raw)
            old_y_pred = old_model.predict(X_test_old_vec)
            old_accuracy = accuracy_score(y_test, old_y_pred)
        except Exception as e:
            print(f"Old model comparison failure: {e}")
            
    print(f"== ACCURACY REPORT ==")
    print(f"Old Model Accuracy: {old_accuracy * 100:.2f}%")
    print(f"New Model Accuracy: {new_accuracy * 100:.2f}%")
    
    if new_accuracy >= old_accuracy:
        print("Performance metric valid. Overwriting models...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(new_model, f)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(new_vectorizer, f)
            
        success_msg = f"Retraining successful! Used {num_records} records. Model updated securely. Accuracy: {old_accuracy*100:.2f}% -> {new_accuracy*100:.2f}%"
        return True, success_msg
    else:
        print("Performance metric invalid. Discarding models...")
        fail_msg = f"Retraining aborted! {num_records} records processed but new model accuracy ({new_accuracy*100:.2f}%) underperformed the existing baseline ({old_accuracy*100:.2f}%)."
        return False, fail_msg

if __name__ == "__main__":
    # If run through terminal directly instead of Flask
    success, message = run_retraining_pipeline()
    print("\n" + message)
