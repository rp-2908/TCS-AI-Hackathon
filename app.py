import streamlit as st
from PIL import Image
import pytesseract
import ollama
import pandas as pd
import json
import io

# Tesseract binary config
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="TCS Offline AI Doc Processor", layout="wide")
st.title("📄 Offline Intelligent Document Processing")

# Sidebar settings
model_choice = st.sidebar.selectbox(
    "Select Local SLM",
    ["llama3.2:3b", "gemma3:4b", "qwen2.5-coder:3b", "deepseek-r1:1.5b"]
)
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Extract Key-Value Pairs (JSON)", "Summarize Document", "Audit / Check Inconsistencies", "Custom Prompt"]
)

uploaded_file = st.file_uploader("Upload Document / Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        
        # 1. OCR Extraction
        raw_text = pytesseract.image_to_string(image)
        with st.expander("Show Raw OCR Text"):
            st.text(raw_text)

    # 2. Build Prompt based on task
    if task_type == "Extract Key-Value Pairs (JSON)":
        prompt = f"Extract all important fields and key-value pairs from this text as pure JSON:\n\n{raw_text}"
    elif task_type == "Summarize Document":
        prompt = f"Provide a clear, concise bulleted summary of this document:\n\n{raw_text}"
    elif task_type == "Audit / Check Inconsistencies":
        prompt = f"Analyze this text for any math errors, missing fields, or logical inconsistencies:\n\n{raw_text}"
    else:
        custom_p = st.text_area("Enter your custom prompt:")
        prompt = f"{custom_p}\n\nDocument Text:\n{raw_text}"

    with col2:
        st.subheader("SLM Analysis")
        if st.button("Process Document"):
            with st.spinner(f"Running inference on {model_choice}..."):
                response = ollama.chat(
                    model=model_choice,
                    messages=[{"role": "user", "content": prompt}]
                )
                output_text = response['message']['content']
                st.write(output_text)

                # 3. Export Options (CSV / Spreadsheets for OpenOffice)
                st.subheader("Export")
                st.download_button(
                    label="Download Result as Text",
                    data=output_text,
                    file_name="processed_output.txt",
                    mime="text/plain"
                )