import os
import json
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from dotenv import load_dotenv
import fitz  # PyMuPDF

load_dotenv()  # Take environment variables from .env (especially Hugging Face API key)

st.title("Intern Bot")
st.sidebar.title("Input Sources")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

pdf_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
process_input_clicked = st.sidebar.button("Process Inputs")
file_path = "embeddings.json"

main_placeholder = st.empty()

# Load Hugging Face model and tokenizer for question answering
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

class Document:
    def _init_(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

if process_input_clicked:
    texts = []

    if urls:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading from URLs...Started...✅✅✅")
        data = loader.load()

        # Split data from URLs
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter for URLs...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        texts.extend([doc.page_content for doc in docs])

    if pdf_file:
        # Load data from PDF
        main_placeholder.text("Data Loading from PDF...Started...✅✅✅")
        pdf_text = load_pdf(pdf_file)

        # Wrap the PDF text in a Document object
        pdf_docs = [Document(pdf_text)]

        # Split data from PDF
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter for PDF...Started...✅✅✅")
        docs = text_splitter.split_documents(pdf_docs)
        texts.extend([doc.page_content for doc in docs])

    # Initialize HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Convert documents to embeddings
    text_embeddings = embeddings.embed_documents(texts)

    # Save embeddings and texts to JSON
    data_to_save = [{"content": text, "embedding": embedding} for text, embedding in zip(texts, text_embeddings)]
    
    with open(file_path, "w") as f:
        json.dump(data_to_save, f)

    main_placeholder.text("Data saved to JSON file successfully...✅✅✅")

query = st.text_input("Question:", "")

if query:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Extract embeddings and texts
        texts = [item["content"] for item in data]
        embeddings_array = np.array([item["embedding"] for item in data])
        
        # Initialize QA pipeline
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        
        # Get query embedding
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = np.array(embeddings.embed_query(query)).reshape(1, -1)
        
        # Perform similarity search
        similarities = cosine_similarity(query_embedding, embeddings_array).flatten()
        similar_indices = similarities.argsort()[-5:][::-1]
        
        # Process each similar document and get the answer using the question-answering pipeline
        answers = []
        for idx in similar_indices:
            doc_content = texts[idx]
            result = qa_pipeline({"question": query, "context": doc_content})
            answers.append(result["answer"])
        
        # Display the answers
        st.header("Answer")
        for answer in answers:
            st.write(answer)