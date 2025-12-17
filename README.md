# GenAI Project- PDF Chat App with Optional General Knowledge Toggle

#  Chat with PDFs using Gemini (LangChain + Streamlit)

An interactive web application that allows users to **upload PDF documents and chat with them intelligently** using Google’s Gemini models.  
The app supports **document-grounded answers with source citations** as well as an optional **mixed mode** where the model can combine document context with its own general knowledge.


##  Live Demo

 **Deployed App**:  
 *(https://genai1-chat-pdf.streamlit.app/)*



##  Features

-  Upload multiple PDF files
-  Semantic search over document content using **FAISS**
-  Question answering powered by **Google Gemini**
-  **Source citations** showing which document chunks were used
-  Two answer modes:
   **Document-only mode** (strictly answers from PDFs)
   **Hybrid mode** (PDF + general knowledge)
-  Optimized performance using Streamlit caching
-  Secure API key handling via environment variables



##  Tech Stack

- **Frontend / UI**: Streamlit
- **LLM**: Google Gemini (via `langchain-google-genai`)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS
- **PDF Parsing**: PyPDF2
- **Framework**: LangChain (modular v0.1+ architecture)



##  How It Works

1. User uploads one or more PDFs
2. PDFs are:
   - Extracted page-by-page
   - Split into overlapping semantic chunks
3. Chunks are embedded and stored locally using FAISS
4. On each question:
   - Relevant chunks are retrieved via similarity search
   - A structured prompt is created
   - Gemini generates an answer
   - Source chunks are returned as citations

##  Answer Modes Explained

###  Document-Only Mode
- The model is **restricted** to the retrieved PDF context
- If the answer is not found, it explicitly says so
- Ideal for:
  - Academic material
  - Legal / policy documents
  - Notes and reports

###  Hybrid Mode
- The model can:
  - Use document context **first**
  - Supplement with general knowledge if needed
- Ideal for:
  - Learning
  - Research
  - Concept clarification


##  Running Locally

### 1️ Clone the repository
```bash
git clone https://github.com/carcked1t/GenAI1.git
cd GenAI1
````

### 2️ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️ Add environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

### 5️ Run the app

```bash
streamlit run app.py
```



##  Project Structure

```
GenAI1/
│
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies
├── .gitignore
├── README.md
│
├── faiss_index/            # Vector store (generated locally)
├── venv/                   # Virtual environment (ignored)
```

---

## ⚠️ Challenges Faced & Solutions

### 1️ LangChain Import Errors

**Issue:**
Frequent `ModuleNotFoundError` due to breaking changes in LangChain.

**Solution:**
Migrated to the new modular structure:

* `langchain_core`
* `langchain_community`
* `langchain_text_splitters`

---

### 2️ Gemini Model Availability Errors

**Issue:**
Some Gemini model names returned `404 NOT_FOUND`.

**Solution:**
Used `models/gemini-1.5-flash` / `models/gemini-flash-latest`
Verified model availability via Google’s API documentation.

---

### 3️ Slow Streamlit Reloads

**Issue:**
Every interaction reloaded embeddings and models.

**Solution:**
Implemented:

```python
@st.cache_resource
```

to cache:

* Embeddings
* FAISS index
* LLM chain

---

### 4️ “Answer Not Found” for All Queries

**Issue:**
Prompt was too restrictive and context retrieval was insufficient.

**Solution:**

* Improved chunk size & overlap
* Retrieved top-K relevant chunks
* Added hybrid answer mode

---

### 5️ API Quota Exhaustion

**Issue:**
Free Gemini quota exhausted during testing.

**Solution:**

* Reduced unnecessary calls
* Cached chains
* Planned fallback handling for deployment

---

## How This Is Different from Typical “Chat with PDF” Apps

*  **Clear separation between document-only and hybrid reasoning**
*  **Explicit source citations**, not just answers
*  Built with **modern LangChain architecture**
*  Optimized for real deployment, not just demos
*  Transparent failure handling (e.g., “answer not found”)

This project focuses on **reliability, clarity, and real-world usability**.


##  Future Improvements
* Conversation memory
* Streaming responses
* Support for DOCX / TXT files
* Model fallback when Gemini quota is exceeded
