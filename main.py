import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import os

# Set up paths
model_path = "model" 
label_encoder_path = "label_encoder/label_encoder.pkl" 

tokenizer = None
model = None
label_encoder = None

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")

try:
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
        st.success("LabelEncoder loaded successfully!")
    else:
        st.error(f"LabelEncoder file not found at: {label_encoder_path}")
except Exception as e:
    st.error(f"Failed to load LabelEncoder: {str(e)}")

def predict(text):
    if model is None or label_encoder is None:
        st.error("Model or LabelEncoder not loaded properly!")
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model(**inputs)
        prediction = torch.argmax(output.logits, dim=-1).item()
        return label_encoder.inverse_transform([prediction])[0]

st.title("Documents Classification")
st.write("Enter text to classify:")

text = st.text_area("Your text here:")

if st.button("Predict"):
    if text:
        if model and label_encoder: 
            label = predict(text)
            st.write("Prediction:", label)
    else:
        st.warning("Please enter some text.")