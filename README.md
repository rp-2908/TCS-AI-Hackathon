# 📖 Smart Practice Paper Generator & RAG Study Tutor

An offline-first, local AI-powered practice assessment generator for students. Built with **Streamlit**, **Ollama**, and an in-memory **TF-IDF Vector Retrieval (RAG)** pipeline.

---

## ✨ Features

- **Document Grounding (RAG):** Upload multiple PDFs, DOCX, or TXT lecture notes. Content is chunked and vectorized locally for high-precision retrieval without context loss.
- **Structured Practice Generation:** Generates multi-tiered practice papers (MCQs, Short Conceptual, In-Depth Problem Solving) with Bloom's Taxonomy and Course Outcome (CO) alignment.
- **On-Demand Detailed Solutions:** Generates comprehensive step-by-step solutions and marking schemes at the bottom of the paper.
- **Quality & Cognitive Analytics:** Live Plotly dashboard visualizing Bloom's cognitive distribution, topic coverage, and vector index status.
- **Clean PDF Export:** Exports print-ready practice papers with administrative tags stripped.
- **RAG-Powered AI Study Tutor:** Interactive chatbot grounded in your uploaded lecture notes.

---

## 🛠️ Tech Stack

- **UI:** Streamlit
- **Local Inference:** Ollama (`llama3.2:3b`, `qwen2.5-coder:3b`, `deepseek-r1:1.5b`)
- **Vector Retrieval:** scikit-learn (TF-IDF Vectorizer + Cosine Similarity)
- **Analytics:** Plotly Express & Pandas
- **Document Processing & Export:** PyPDF, python-docx, FPDF2

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Install [Ollama](https://ollama.com) and pull your preferred local SLM:
```bash
ollama pull llama3.2:3b
```

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/rp-2908/TCS-AI-Hackathon.git
cd TCS-AI-Hackathon
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```


💡 How to Use
1. Setup Practice Parameters: Use the sidebar to configure the subject, target time, marks, difficulty tier, and question distribution.

2. Provide Knowledge Base: Type custom study topics or drag and drop multiple lecture files (PDF, DOCX, TXT) to index into the vector store.

3. Generate Questions: Click Generate Practice Questions to produce the question set.

4. Generate Solutions: Click Generate Full Step-by-Step Solutions & Answer Key to append the complete answer guide.

5. Download & Analyze: View cognitive depth in the Topic & Bloom Analytics tab or download a clean, formatted PDF export.

6. Ask the Tutor: Switch to the RAG AI Study Tutor tab for follow-up explanations grounded in your uploaded documents.
