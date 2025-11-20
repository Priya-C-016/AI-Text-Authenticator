import streamlit as st
import pickle

st.set_page_config(page_title="AI vs Human Text Classifier", layout="wide")

# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        embedder, clf = pickle.load(f)
    return embedder, clf

embedder, clf = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 AI vs Human Text Classifier")
st.write("This model uses **LLM embeddings (MiniLM)** + Logistic Regression to predict whether text is written by a Human or AI.")

user_input = st.text_area("Enter text:", height=200, placeholder="Type or paste text here...")

if st.button("Analyze", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Encode input
        emb = embedder.encode([user_input])
        
        # Predict
        pred = clf.predict(emb)[0]
        prob = clf.predict_proba(emb)[0][1]

        st.subheader("🔍 Prediction")
        if pred == 1:
            st.success("This text is LIKELY **AI-generated** 🤖")
        else:
            st.info("This text is LIKELY **Human-written** 🧑‍💻")

        # Show confidence
        st.write(f"**Confidence:** {round(prob * 100, 2)}%")

st.markdown("""
---
### ℹ️ About the Model
- Embedding Model: **all-MiniLM-L6-v2**
- Classifier: **Logistic Regression**
- Dataset: AI vs Human Text Corpus  
""")
