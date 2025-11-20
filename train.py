import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("ai_vs_human_text.csv")

# FIX: Convert string labels to numeric
df["label_num"] = df["label"].map({
    "AI-generated": 1,
    "Human-written": 0
})

texts = df["text"].astype(str).tolist()
labels = df["label_num"].tolist()

# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading SentenceTransformer (MiniLM)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Encode Text
# -----------------------------
print("Encoding text into embeddings...")
embeddings = embedder.encode(texts, show_progress_bar=True)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)

# -----------------------------
# Train Classifier
# -----------------------------
print("Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n------ Model Performance ------")
print("Accuracy:", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:", round(rec, 4))
print("F1 Score:", round(f1, 4))
print("--------------------------------\n")

# -----------------------------
# Save Model
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump((embedder, clf), f)

print("Model training complete! Saved as model.pkl")
