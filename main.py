import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import os
import PyPDF2

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

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

st.title("Documents Classification")
st.write("Upload a PDF to classify:")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if st.button("Predict"):
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        if text.strip():
            label = predict(text)
            st.write("Prediction:", label)
        else:
            st.warning("No text could be extracted from the PDF.")
    else:
        st.warning("Please upload a PDF file.")
