import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle
import re

# paths to datasets
TRUE_CSV = "data/True.csv"
FAKE_CSV = "data/Fake.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "misinfo_model.pkl")

#clean text
def clean_text(text):
    # remove URLs
    text = re.sub(r"http\S+", "", text)           
    # remove punctuation
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)  
    # lowercase
    text = text.lower()                           
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()     
    return text

# load datasets
true_df = pd.read_csv(TRUE_CSV)
fake_df = pd.read_csv(FAKE_CSV)

# add labels
true_df["label"] = "trustworthy"
fake_df["label"] = "misinfo"

# combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# combine title and text
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["content"] = df["content"].apply(clean_text)

# features and labels
X = df["content"]
y = df["label"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=2000))
])

# train model
print("Training model...")
model.fit(X_train, y_train)

# evaluate model
print("\nEvaluating on test set...")
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# save model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to {MODEL_PATH}")
