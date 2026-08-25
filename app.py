import streamlit as st
from PIL import Image
import pytesseract
import pypdf
import docx
import ollama
import pandas as pd
import json
import io

# Optional: Set Tesseract path if required on Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Offline Document Intelligence Pipeline", layout="wide")

st.title("📄 Offline Intelligent Document Processing (IDP)")
st.caption("Powered by Local SLMs (Ollama) & Tesseract OCR | 100% On-Device & Air-Gapped")

# Universal Text Extraction Function
def extract_text_from_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    
    # 1. Plain Text
    if ext == 'txt':
        return uploaded_file.read().decode('utf-8', errors='ignore')
    
    # 2. Word Document
    elif ext == 'docx':
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    
    # 3. PDF Document
    elif ext == 'pdf':
        reader = pypdf.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or '' for page in reader.pages])
    
    # 4. Image Formats (OCR)
    elif ext in ['png', 'jpg', 'jpeg']:
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
        
    return ""

# Sidebar Configuration
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Select Local SLM:",
    ["llama3.2:3b", "gemma3:4b", "qwen2.5-coder:3b", "deepseek-r1:1.5b"],
    index=0
)

task_type = st.sidebar.selectbox(
    "Select Extraction / NLP Task:",
    [
        "Extract Key-Value Pairs (JSON)",
        "Summarize Document",
        "Audit / Check Inconsistencies",
        "Custom Prompt"
    ]
)

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Document (Image, PDF, DOCX, TXT):",
    type=["png", "jpg", "jpeg", "pdf", "docx", "txt"]
)

if uploaded_file:
    col1, col2 = st.columns(2)
    ext = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner("Extracting content from document..."):
        raw_text = extract_text_from_file(uploaded_file)

    with col1:
        st.subheader("Input Document")
        if ext in ['png', 'jpg', 'jpeg']:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        else:
            st.info(f"📄 Uploaded File: **{uploaded_file.name}**")
            st.caption(f"Format: **{ext.upper()}** document processed directly.")

        with st.expander("Show Extracted Raw Text", expanded=False):
            st.text_area("Raw Text Content", raw_text, height=250)

    # Build prompt based on task
    if task_type == "Extract Key-Value Pairs (JSON)":
        prompt = (
            "Extract all important fields, entities, and key-value pairs from this document. "
            "Return valid JSON only, without conversational markdown or commentary:\n\n"
            f"{raw_text}"
        )
    elif task_type == "Summarize Document":
        prompt = f"Provide a clear, concise bulleted summary of this document:\n\n{raw_text}"
    elif task_type == "Audit / Check Inconsistencies":
        prompt = (
            "Analyze this text for any math errors, missing mandatory fields, or logical inconsistencies. "
            "Explain step-by-step:\n\n"
            f"{raw_text}"
        )
    else:
        custom_p = st.text_area("Enter your custom prompt:", value="Extract table records as CSV:")
        prompt = f"{custom_p}\n\nDocument Content:\n{raw_text}"

    with col2:
        st.subheader("SLM Analysis")
        if st.button("Process Document", type="primary"):
            if not raw_text.strip():
                st.error("No readable text could be extracted from this document.")
            else:
                with st.spinner(f"Running inference locally with {model_choice}..."):
                    response = ollama.chat(
                        model=model_choice,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    output_text = response['message']['content']
                    st.markdown(output_text)

                    st.markdown("---")
                    st.subheader("Export Options")

                    # Primary Text / JSON Download
                    st.download_button(
                        label="📥 Download Output (.txt / .json)",
                        data=output_text,
                        file_name="processed_output.txt",
                        mime="text/plain"
                    )

                    # Auto CSV generation for OpenOffice Calc if JSON output is detected
                    try:
                        clean_json_str = output_text.strip()
                        if clean_json_str.startswith("```json"):
                            clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
                        elif clean_json_str.startswith("```"):
                            clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()
                            
                        parsed_json = json.loads(clean_json_str)
                        if isinstance(parsed_json, list):
                            df = pd.DataFrame(parsed_json)
                        elif isinstance(parsed_json, dict):
                            df = pd.json_normalize(parsed_json)
                        else:
                            df = None

                        if df is not None and not df.empty:
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📊 Download Table for OpenOffice Calc (.csv)",
                                data=csv_data,
                                file_name="extracted_table.csv",
                                mime="text/csv"
                            )
                    except Exception:
                        pass