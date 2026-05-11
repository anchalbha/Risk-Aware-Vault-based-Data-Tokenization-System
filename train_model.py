import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def train_and_evaluate():
    data_file = "synthetic_dataset.csv"
    print(f"Loading dataset from {data_file}...")
    
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Please run dataset_generation.py first.")
        return
    
    X = df['text']
    y = df['risk_level']
    
    print("Vectorizing Text Using TF-IDF...")
    # For classification of PII formats (phones, emails, SSN-like),
    # Character-level n-grams consistently perform better than word n-grams
    # Per system specifications: analyzer='char' and ngram_range=(2,5)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
    X_vectorized = vectorizer.fit_transform(X)
    
    # 80-20 Train-Test split for model validation
    print("Splitting dataset into 80% training / 20% testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )
    
    # Train Logistic Regression Model
    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating Model Performance on Test Set...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"RESEARCH METRICS REPORT")
    print("="*50)
    print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
    print(f"NOTE: Target accuracy is >90% for publication.")
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))
    
    print("\nCONFUSION MATRIX:")
    print("Rows: Actual | Columns: Predicted (Order varies but labels are mapped below)")
    cm = confusion_matrix(y_test, y_pred, labels=["HIGH", "MEDIUM", "LOW"])
    print(pd.DataFrame(cm, index=["Actual HIGH", "Actual MEDIUM", "Actual LOW"], 
                       columns=["Pred HIGH", "Pred MEDIUM", "Pred LOW"]))
    print("="*50 + "\n")
    
    # Final step: Save Pipeline Components
    print("Saving ML Pipeline artifacts...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Successfully saved 'model.pkl' and 'vectorizer.pkl'. Training completed.")

if __name__ == "__main__":
    train_and_evaluate()
