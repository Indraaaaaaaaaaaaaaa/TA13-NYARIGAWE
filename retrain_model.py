"""
Script untuk retrain dan save model SVM sentiment analysis.
Memastikan model disimpan dengan benar dan sudah fitted.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

CLEAN_PATH = "nyarigawe_reviews_clean.csv"
MODEL_PATH = "svm_sentiment_model.joblib"

print("=== RETRAIN MODEL ===")
print(f"Data path: {CLEAN_PATH}")
print(f"Model path: {MODEL_PATH}")

# Load data
if not os.path.exists(CLEAN_PATH):
    raise FileNotFoundError(f"{CLEAN_PATH} tidak ditemukan!")

df_clean = pd.read_csv(CLEAN_PATH)
print(f"\nData shape: {df_clean.shape}")

if "text" not in df_clean.columns or "label" not in df_clean.columns:
    raise ValueError("File harus punya kolom: text dan label")

df_clean["text"] = df_clean["text"].astype(str)
df_clean["label"] = df_clean["label"].astype(int)

print(f"Distribusi label:\n{df_clean['label'].value_counts()}")

# Prepare data
X = df_clean["text"]
y = df_clean["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Create and train model
print("\nTraining model...")
model = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel="linear", C=1, random_state=42)
)

model.fit(X_train, y_train)
print("Model training selesai!")

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# Test prediction before saving
print("\nTesting prediction sebelum save...")
test_texts = [
    "aplikasinya sangat membantu dan mudah digunakan",
    "sering error dan susah dipakai mengecewakan"
]
test_preds = model.predict(test_texts)
print(f"Test predictions: {test_preds}")

# Verify the model is fitted
print("\nVerifying model is fitted...")
try:
    # Check if TF-IDF is fitted by trying to transform
    tfidf = model.named_steps['tfidfvectorizer']
    _ = tfidf.transform(["test text"])
    print("✓ TF-IDF vectorizer is properly fitted!")
except Exception as e:
    print(f"✗ Error: TF-IDF not fitted properly: {e}")
    raise

# Save model
print(f"\nSaving model to {MODEL_PATH}...")
joblib.dump(model, MODEL_PATH)
print("✓ Model saved successfully!")

# Test loading
print("\nTesting model load...")
loaded_model = joblib.load(MODEL_PATH)
test_preds_loaded = loaded_model.predict(test_texts)
print(f"Loaded model predictions: {test_preds_loaded}")

if (test_preds == test_preds_loaded).all():
    print("✓ Model loaded and working correctly!")
else:
    print("✗ Warning: Predictions don't match!")
    
print("\n=== DONE ===")

