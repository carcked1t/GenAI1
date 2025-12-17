import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# ---------------- CONFIG ----------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="Chat with PDF (Gemini RAG)",
    layout="wide"
)

# ---------------- CACHES ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=0.3
    )

@st.cache_resource
def load_faiss():
    embeddings = load_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# ---------------- HELPERS ----------------
def get_pdf_documents(pdf_docs):
    """
    Extract text + source metadata (pdf name + page)
    """
    documents = []

    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                documents.append({
                    "text": text,
                    "source": f"{pdf.name} — page {i + 1}"
                })

    return documents


def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = []
    metadatas = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": doc["source"]})

    return texts, metadatas


def build_vector_store(texts, metadatas):
    embeddings = load_embeddings()
    db = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    db.save_local("faiss_index")


def build_chain(system_instruction):
    prompt = PromptTemplate(
        template=f"""
{system_instruction}

Context:
{{context}}

Question:
{{question}}

Answer:
""",
        input_variables=["context", "question"]
    )

    llm = load_llm()

    return (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# ---------------- USER QUERY ----------------
def user_input(question, mode):
    db = load_faiss()
    docs = db.similarity_search(question, k=4)

    context = "\n\n".join(doc.page_content for doc in docs)

    if mode == "📄 Document only":
        system_instruction = """
        You must answer ONLY using the provided context.

        You ARE allowed to:
        - Summarize information
        - Paraphrase definitions
        - Combine multiple sentences

        You are NOT allowed to:
        - Add facts not present in the context

        If the context truly contains no relevant information, say:
        "Answer not found in the provided document."
        """
    else:
        system_instruction = """
First use the provided context.
If the context is insufficient, you MAY use general knowledge.
Clearly mention when outside knowledge is used.
"""

    chain = build_chain(system_instruction)

    response = chain.invoke({
        "context": context,
        "question": question
    })

    st.markdown("### 📌 Answer")
    st.write(response)

    st.markdown("### 📚 Sources")
    sources = sorted(set(doc.metadata["source"] for doc in docs))
    for src in sources:
        st.write("•", src)

# ---------------- STREAMLIT UI ----------------
def main():
    st.header("📄 Chat with PDF using Gemini (RAG)")

    mode = st.radio(
        "Answer mode",
        ["📄 Document only", "🌍 Document + General Knowledge"]
    )

    question = st.text_input("Ask a question from your PDFs")

    if question:
        with st.spinner("Thinking..."):
            user_input(question, mode)

    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdfs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if pdfs and st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                documents = get_pdf_documents(pdfs)
                texts, metadatas = get_text_chunks(documents)
                build_vector_store(texts, metadatas)
                st.success("PDFs indexed successfully!")

if __name__ == "__main__":
    main()
