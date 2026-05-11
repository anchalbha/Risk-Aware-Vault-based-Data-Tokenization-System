import random
import string
import csv

def generate_phone():
    """Generates a random US-style phone number."""
    return f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def generate_aadhaar():
    """Generates a random Aadhaar-like 12 digit number format."""
    return f"{random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"

def generate_credit_card():
    """Generates a random 16 digit credit card-like format."""
    return f"{random.randint(4000,4999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

def generate_bank_routing():
    """Generates a random 9 digit bank routing number."""
    return "".join([str(random.randint(0,9)) for _ in range(9)])

def generate_email():
    """Generates a random email address."""
    domains = ["gmail.com", "yahoo.com", "outlook.com", "university.edu", "company.org"]
    username = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
    if random.random() > 0.5:
        username += str(random.randint(1, 99))
    return f"{username}@{random.choice(domains)}"

def generate_username():
    """Generates a realistic username."""
    chars = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 8)))
    nums = "".join(random.choices(string.digits, k=random.randint(2, 4)))
    return f"{chars}_{nums}"

def generate_normal_text():
    """Generates a realistic normal text sentence."""
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world",
        "apple", "banana", "this", "is", "a", "test", "sentence", "for", "machine", "learning",
        "research", "paper", "data", "science", "python", "programming", "code", "vault", "system",
        "tokenization", "security", "privacy", "evaluation", "metrics", "pipeline", "today"
    ]
    sentence_length = random.randint(4, 10)
    return " ".join(random.choices(words, k=sentence_length)).capitalize() + "."

def generate_dataset(num_samples=10000):
    dataset = []
    
    # Weight distribution slightly towards LOW text to reflect natural communication
    for _ in range(num_samples):
        category = random.choices(["HIGH", "MEDIUM", "LOW"], weights=[0.3, 0.3, 0.4])[0]
        
        if category == "HIGH":
            # high risk sub-categories
            func = random.choice([generate_phone, generate_aadhaar, generate_credit_card, generate_bank_routing])
            text = func()
        elif category == "MEDIUM":
            # medium risk sub-categories
            func = random.choice([generate_email, generate_username])
            text = func()
        else:
            # low risk
            text = generate_normal_text()
            
        dataset.append((text, category))
        
    return dataset

if __name__ == "__main__":
    n_samples = 10000
    print(f"Generating synthetic research dataset with {n_samples} samples...")
    data = generate_dataset(n_samples)
    
    # Save the synthetic data to CSV format
    filename = "synthetic_dataset.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "risk_level"])  # CSV Header
        writer.writerows(data)
        
    print(f"Success! Dataset saved to {filename}")
    print(f"Total samples: {len(data)}")
