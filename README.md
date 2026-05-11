# Risk-Aware Vault-Based Data Tokenization System

This repository contains a secure, hybrid risk-aware tokenization system built with Python, Flask, and MongoDB. The system dynamically evaluates the sensitivity of input data using a combination of rule-based logic and machine learning, and tokenizes it securely using a vault architecture.

## Features

- **Hybrid Risk Evaluation Engine:**
  - **Rule-Based Pre-Flight:** Instantly detects explicit sensitive patterns using Regex (e.g., PAN cards, Indian Aadhaar numbers, Credit Cards, Emails, and Phone Numbers).
  - **Machine Learning Fallback:** Evaluates non-explicit text using a trained machine learning model (`model.pkl` and `vectorizer.pkl`), applying a confidence threshold.
- **Secure Vault Architecture:**
  - Separates non-sensitive application data (`records`) from highly sensitive, encrypted source data (`vault`).
- **Selective Encryption:**
  - Uses `cryptography.fernet` to securely encrypt data classified as `HIGH` or `MEDIUM` risk before it hits the database.
- **Role-Based Access Control (RBAC) & Detokenization:**
  - Supports role-based detokenization (e.g., `Admin` gets full decryption, `Analyst` gets masked data, `User` is denied).
- **Audit Logging:**
  - Maintains strict `audit_logs` tracking detokenization attempts, usernames, roles, and timestamps.

## Prerequisites

- **Python 3.8+**
- **MongoDB** (running locally on default port `27017`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anchalbha/Risk-Aware-Vault-based-Data-Tokenization-System.git
   cd Risk-Aware-Vault-based-Data-Tokenization-System
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add a Fernet encryption key:
   ```env
   ENCRYPTION_KEY=your_generated_fernet_key_here
   ```
   *(You can generate a key using: `from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())`)*

5. **Train the ML Model:**
   If `model.pkl` and `vectorizer.pkl` are not present or need updating, generate the dataset and run the training script:
   ```bash
   python dataset_generation.py
   python train_model.py
   ```

## Usage

1. **Start MongoDB:**
   Ensure your local MongoDB instance is running.
   
2. **Run the Flask Application:**
   ```bash
   python app.py
   ```
   The web server will start at `http://127.0.0.1:5000/`.

3. **Interacting with the UI:**
   - Navigate to `http://127.0.0.1:5000/` in your browser.
   - Enter text into the form to evaluate its risk and safely tokenize it.

4. **Detokenization API:**
   You can securely retrieve data via the `/detokenize` endpoint using tools like `curl` or Postman.
   ```bash
   curl -X POST http://127.0.0.1:5000/detokenize \
        -H "Content-Type: application/json" \
        -d '{"token": "YOUR_TOKEN_HERE", "username": "alice"}'
   ```
   *(`alice` maps to Admin, `bob` maps to Analyst)*

## Architecture

1. **Rule-Based Engine (`hybrid_predict_risk`)**: Regex patterns catch highly sensitive structured data first.
2. **Machine Learning Model**: NLP-based fallback classification.
3. **Vault Isolation**: Actual encrypted payloads sit safely in the `vault` collection, disconnected from application usage identifiers.
4. **Endpoint (`/detokenize`)**: Ensures requests are rigorously audited before decrypting text based on the mapped `token`.

## License
MIT License
