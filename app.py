import streamlit as st
import pandas as pd
import ollama
import pypdf
import docx
import json
import io
import re
import numpy as np
import plotly.express as px
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Smart Practice Paper Generator (RAG-Enabled)", 
    page_icon="📖", 
    layout="wide"
)

# --- In-Memory Vector Store State ---
if 'rag_corpus' not in st.session_state:
    st.session_state['rag_corpus'] = []
if 'rag_vectorizer' not in st.session_state:
    st.session_state['rag_vectorizer'] = None
if 'rag_vectors' not in st.session_state:
    st.session_state['rag_vectors'] = None

# --- Helper: Document Extraction ---
def extract_text(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'txt':
        return uploaded_file.read().decode('utf-8', errors='ignore')
    elif ext == 'docx':
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    elif ext == 'pdf':
        reader = pypdf.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or '' for page in reader.pages])
    return ""

def chunk_text(text, chunk_size=400, overlap=80):
    """Splits text into overlapping word chunks for vector retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
    return chunks

def index_documents(chunks):
    """Indexes text chunks into an in-memory vector space using TF-IDF."""
    if not chunks:
        return False
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(chunks)
    
    st.session_state['rag_corpus'] = chunks
    st.session_state['rag_vectorizer'] = vectorizer
    st.session_state['rag_vectors'] = vectors
    return True

def retrieve_context(query, top_k=4):
    """Computes cosine similarity and retrieves the most relevant chunks."""
    corpus = st.session_state.get('rag_corpus', [])
    vectorizer = st.session_state.get('rag_vectorizer')
    vectors = st.session_state.get('rag_vectors')
    
    if not corpus or vectorizer is None or vectors is None:
        return ""
    
    try:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, vectors).flatten()
        
        # Get top matching indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        matched_chunks = [corpus[i] for i in top_indices if similarities[i] > 0.05]
        
        if not matched_chunks:
            matched_chunks = corpus[:top_k]
            
        return "\n\n---\n\n".join(matched_chunks)
    except Exception:
        return "\n\n---\n\n".join(corpus[:top_k])

def clean_markdown_for_pdf(text):
    """Strips markdown syntax, headers, Bloom's tags, and CO tags for a clean exam export."""
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    bloom_words = r'Remember|Understand|Apply|Analyze|Evaluate|Create'
    text = re.sub(rf'[\[\(]\s*({bloom_words})\s*[\]\)]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\[\(]\s*CO\s*\d+\s*[\]\)]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = text.replace('•', '-').replace('–', '-').replace('—', '-')
    text = text.replace('“', '"').replace('”', '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def create_pdf(text_content, subject_title, target_time, total_marks_val):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # 1. Student Practice Header
    pdf.set_font("Helvetica", style="B", size=15)
    pdf.cell(w=0, h=8, text="STUDENT SELF-ASSESSMENT & PRACTICE PAPER", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", style="B", size=11)
    clean_subj = clean_markdown_for_pdf(subject_title).upper()
    pdf.cell(w=0, h=6, text=f"SUBJECT: {clean_subj}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    # Horizontal Rule
    pdf.set_draw_color(180, 180, 180)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)
    
    # Metadata Row
    pdf.set_font("Helvetica", style="B", size=9.5)
    pdf.cell(w=110, h=5, text=f"Target Marks: {total_marks_val} pts")
    pdf.cell(w=70, h=5, text=f"Recommended Time: {target_time}", align="R", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(1)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    
    # 2. Body Text Rendering
    epw = pdf.epw 
    cleaned_body = clean_markdown_for_pdf(text_content)
    encoded_text = cleaned_body.encode('latin-1', 'replace').decode('latin-1')
    
    for line in encoded_text.split('\n'):
        pdf.set_x(pdf.l_margin)
        line_strip = line.strip()
        
        if not line_strip:
            pdf.ln(2)
            continue
            
        if line_strip.lower().startswith("section") or line_strip.lower().startswith("part") or (line_strip.isupper() and len(line_strip) < 45):
            pdf.ln(2)
            pdf.set_font("Helvetica", style="B", size=10.5)
            pdf.multi_cell(w=epw, h=6, text=line_strip, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=9)
        else:
            pdf.set_font("Helvetica", size=9)
            pdf.multi_cell(w=epw, h=5, text=line, new_x="LMARGIN", new_y="NEXT")
            
    return bytes(pdf.output())

# --- Sidebar: Student Practice Configuration ---
st.sidebar.title("🎯 Practice Setup")

model_choice = st.sidebar.selectbox(
    "Select AI Model:",
    ["llama3.2:3b", "qwen2.5-coder:3b", "gemma3:4b", "deepseek-r1:1.5b"],
    index=0
)

subject_name = st.sidebar.text_input("Subject / Topic Area:", value="Machine Learning")
target_time = st.sidebar.text_input("Estimated Practice Time:", value="60 Minutes")
total_marks = st.sidebar.slider("Total Practice Marks:", min_value=10, max_value=100, value=40, step=5)

difficulty = st.sidebar.select_slider(
    "Target Difficulty Level:",
    options=["Beginner / Fundamentals", "Balanced / Exam Level", "Advanced / Competitive"],
    value="Balanced / Exam Level"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔢 Practice Question Mix")
num_mcqs = st.sidebar.number_input("MCQs (Quick Concept Check):", min_value=0, max_value=20, value=5)
num_short = st.sidebar.number_input("Short Practice Questions (3-5 Marks):", min_value=0, max_value=10, value=3)
num_long = st.sidebar.number_input("Problem Solving Questions (10 Marks):", min_value=0, max_value=5, value=2)

# --- Main App Tabs ---
tab1, tab2, tab3 = st.tabs(["📝 Practice Paper Generator", "📊 Topic & Bloom Analytics", "🤖 RAG AI Study Tutor"])

# ==========================================
# TAB 1: PRACTICE PAPER GENERATOR
# ==========================================
with tab1:
    st.header("📖 AI Smart Practice Assessment Generator")
    st.caption("Enhanced with In-Memory Semantic Vector Retrieval (RAG) for deep document grounding.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Study Material / Topics")
        input_mode = st.radio("Input Source:", ["Type Study Topics", "Upload Notes / Syllabus / PDF"], horizontal=True)
        
        raw_text_data = ""

        if input_mode == "Type Study Topics":
            raw_text_data = st.text_area(
                "Enter Topics to Practice:",
                value="Unit 1: Linear & Logistic Regression, Regularization\nUnit 2: Decision Trees, Random Forests, Ensembles\nUnit 3: Support Vector Machines, Kernels\nUnit 4: Neural Networks, Backpropagation",
                height=180
            )
        else:
            uploaded_docs = st.file_uploader(
                "Upload Study Material (PDF, DOCX, TXT):", 
                type=["pdf", "docx", "txt"], 
                accept_multiple_files=True
            )
            if uploaded_docs:
                extracted_texts = []
                for doc_file in uploaded_docs:
                    extracted_texts.append(extract_text(doc_file))
                raw_text_data = "\n\n".join(extracted_texts)
                
                chunks = chunk_text(raw_text_data)
                if chunks:
                    success = index_documents(chunks)
                    if success:
                        st.success(f"✅ RAG Active: Indexed {len(uploaded_docs)} file(s) ({len(chunks)} vector chunks created).")

        st.subheader("2. Focus Areas")
        special_reqs = st.text_input("Topics to emphasize:", value="Focus heavily on Bias-Variance Tradeoff and Regularization.")

        generate_btn = st.button("🚀 Generate Practice Questions", type="primary", use_container_width=True)

    with col2:
        st.subheader("Generated Practice Paper")
        
        if generate_btn:
            if not raw_text_data.strip():
                st.error("Please enter topics or upload study material first.")
            else:
                # Retrieve relevant vector context if documents are indexed, otherwise fallback to raw input
                if st.session_state.get('rag_corpus'):
                    search_query = f"{subject_name} {special_reqs}"
                    retrieved_context = retrieve_context(search_query, top_k=5)
                else:
                    retrieved_context = raw_text_data[:3000]

                prompt = f"""You are an expert tutor creating a self-assessment practice paper for a student studying '{subject_name}'.

CRITICAL INSTRUCTIONS:
1. SPECIFIC FOCUS: {special_reqs}
2. BASE CONTENT ON RETRIEVED CONTEXT: Use the reference material below to generate accurate questions.
3. Output ONLY the Practice Questions right now. Do not include answers in this section.

### PART 1: PRACTICE QUESTIONS

SECTION A: MULTIPLE CHOICE QUESTIONS ({num_mcqs} Questions, 1 Mark each)
Q1. [Bloom Tag] (CO Tag) [Question Text]
a) [Option A]  b) [Option B]  c) [Option C]  d) [Option D]
... (Generate exactly {num_mcqs} MCQs)

SECTION B: SHORT CONCEPTUAL QUESTIONS ({num_short} Questions, 3-5 Marks each)
Q[Number]. [Bloom Tag] (CO Tag) [Question Text] ([Marks] Marks)
... (Generate exactly {num_short} Short Questions)

SECTION C: IN-DEPTH PROBLEM SOLVING ({num_long} Questions, 10 Marks each)
Q[Number]. [Bloom Tag] (CO Tag) [Problem / Implementation Scenario] (10 Marks)
... (Generate exactly {num_long} Long Questions)

PRACTICE DETAILS:
- Subject: {subject_name}
- Total Marks: {total_marks}
- Difficulty Level: {difficulty}

RETRIEVED KNOWLEDGE CONTEXT:
{retrieved_context}
"""
                with st.spinner(f"Generating questions using {model_choice} with RAG context..."):
                    try:
                        response = ollama.chat(
                            model=model_choice,
                            messages=[{"role": "user", "content": prompt}],
                            options={"num_predict": 4096, "temperature": 0.2}
                        )
                        paper_text = response['message']['content']
                        st.session_state['generated_paper'] = paper_text
                    except Exception as e:
                        st.error(f"Error calling Ollama: {str(e)}")

        if 'generated_paper' in st.session_state:
            st.markdown(st.session_state['generated_paper'])
            
            # --- Dedicated Solution Generator Button ---
            if st.button("💡 Generate Full Step-by-Step Solutions & Answer Key", use_container_width=True):
                with st.spinner(f"Generating detailed solutions using {model_choice}..."):
                    sol_prompt = f"""You are an expert professor. Provide the complete, accurate STEP-BY-STEP ANSWER KEY and SOLUTIONS for all questions in this practice paper.

EXAM PAPER:
{st.session_state['generated_paper']}

FORMAT REQUIREMENT:
### PART 2: STEP-BY-STEP SOLUTIONS & ANSWER KEY
- SECTION A (MCQs): List question number, correct option (a/b/c/d), and detailed concept explanation.
- SECTION B (Short Answers): Provide exact model answers with key formulas/points.
- SECTION C (Problem Solving): Provide complete step-by-step mathematical/conceptual worked solutions and algorithms.
"""
                    try:
                        sol_res = ollama.chat(
                            model=model_choice,
                            messages=[{"role": "user", "content": sol_prompt}],
                            options={"num_predict": 4096, "temperature": 0.2}
                        )
                        solutions_text = sol_res['message']['content']
                        st.session_state['generated_paper'] += "\n\n---\n\n" + solutions_text
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating solutions: {str(e)}")

            st.markdown("---")
            
            d_col1, d_col2 = st.columns(2)
            
            with d_col1:
                pdf_data = create_pdf(
                    text_content=st.session_state['generated_paper'],
                    subject_title=subject_name,
                    target_time=target_time,
                    total_marks_val=total_marks
                )
                st.download_button(
                    label="📄 Download Clean PDF",
                    data=pdf_data,
                    file_name=f"{subject_name.replace(' ', '_')}_Practice_Paper.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
                
            with d_col2:
                st.download_button(
                    label="📝 Download Text (.txt)",
                    data=st.session_state['generated_paper'],
                    file_name=f"{subject_name.replace(' ', '_')}_Practice_Paper.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ==========================================
# TAB 2: TOPIC & BLOOM ANALYTICS
# ==========================================
with tab2:
    st.header("📊 Practice Quality & Cognitive Analytics")
    st.caption("Live insights computed dynamically from your generated practice test.")

    paper_raw = st.session_state.get('generated_paper', '')

    if not paper_raw:
        st.info("ℹ️ Generate a practice paper in Tab 1 to see live cognitive and topic breakdown.")
    else:
        co_matches = re.findall(r'CO\s*(\d+)', paper_raw, re.IGNORECASE)
        co_counts = {}
        for co_num in sorted(set(co_matches)):
            co_counts[f"CO{co_num}"] = co_matches.count(co_num)
        
        bloom_categories = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        bloom_found = {}
        for cat in bloom_categories:
            count = len(re.findall(rf'\b{cat}\b', paper_raw, re.IGNORECASE))
            if count > 0:
                bloom_found[cat] = count

        if not bloom_found:
            bloom_found = {"Remember": num_mcqs, "Apply": num_short, "Analyze": num_long}

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Practice Target", f"{total_marks} pts")
        m2.metric("Difficulty", difficulty.split('/')[0].strip())
        m3.metric("Vector DB Status", f"Active ({len(st.session_state.get('rag_corpus', []))} Chunks)" if st.session_state.get('rag_corpus') else "Direct Prompt")
        m4.metric("Cognitive Depth", f"{len(bloom_found)} Bloom Tiers")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            if co_counts:
                df_co = pd.DataFrame(list(co_counts.items()), columns=["Study Unit / CO", "Questions"])
                fig_co = px.pie(
                    df_co, 
                    values="Questions", 
                    names="Study Unit / CO", 
                    title="Topic & Course Outcome Coverage",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_co, use_container_width=True)
            else:
                df_sec = pd.DataFrame({
                    "Section": ["MCQs (Concepts)", "Short Questions", "Problem Solving (Long)"],
                    "Weightage (Marks)": [num_mcqs * 1, num_short * 4, num_long * 10]
                })
                fig_sec = px.pie(df_sec, values="Weightage (Marks)", names="Section", title="Marks Distribution by Question Type")
                st.plotly_chart(fig_sec, use_container_width=True)

        with col_b:
            df_bloom = pd.DataFrame(list(bloom_found.items()), columns=["Cognitive Skill", "Question Count"])
            fig_bloom = px.bar(
                df_bloom, 
                x="Cognitive Skill", 
                y="Question Count", 
                color="Cognitive Skill", 
                title="Bloom's Cognitive Skill Distribution",
                text_auto=True
            )
            st.plotly_chart(fig_bloom, use_container_width=True)

        st.subheader("📋 Section-Wise Practice Breakdown")
        summary_table = pd.DataFrame([
            {"Section": "Section A", "Question Type": "Multiple Choice (MCQ)", "Count": f"{num_mcqs} Questions", "Unit Value": "1 Mark"},
            {"Section": "Section B", "Question Type": "Short Conceptual", "Count": f"{num_short} Questions", "Unit Value": "3-5 Marks"},
            {"Section": "Section C", "Question Type": "In-Depth Problem Solving", "Count": f"{num_long} Questions", "Unit Value": "10 Marks"},
        ])
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: RAG AI STUDY TUTOR
# ==========================================
with tab3:
    st.header("🤖 RAG-Powered AI Study Tutor")
    st.caption("Ask questions about your uploaded documents or get explanations grounded in the vector database.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your RAG Study Tutor. Ask me any question, and I will look up your indexed notes to explain the answer."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask a study question (e.g., 'Explain the concept behind Question 1 in detail')"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve relevant passages from vector store for the query
        rag_context = retrieve_context(user_query, top_k=3)

        assistant_prompt = f"""You are a helpful and patient personal academic tutor for '{subject_name}'.

RETRIEVED STUDY CONTEXT FROM VECTOR STORE:
{rag_context}

PRACTICE PAPER CONTEXT:
{st.session_state.get('generated_paper', '')[:1500]}

Student Query: {user_query}
Provide a clear, encouraging, and step-by-step academic explanation based on the retrieved context.
"""
        with st.chat_message("assistant"):
            with st.spinner("Searching vector store and generating explanation..."):
                res = ollama.chat(model=model_choice, messages=[{"role": "user", "content": assistant_prompt}])
                bot_reply = res['message']['content']
                st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})