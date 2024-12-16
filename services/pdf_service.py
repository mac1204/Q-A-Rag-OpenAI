import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document

from typing import List
#from langchain.rerankers import Reranker
#from langchain.rerankers.openai import OpenAIReranker
from sentence_transformers import SentenceTransformer, util


def custom_rerank(query, documents):
    """Rerank documents based on relevance to the query."""
    # Load a SentenceTransformer model for embeddings
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Generate embeddings for the query and documents
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode([doc.page_content for doc in documents], convert_to_tensor=True)

    # Compute cosine similarity scores
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    # Pair documents with scores and sort by relevance
    ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked_results


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def build_index_with_recursive_splitter(text):
    """Build a vector store index using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in docs]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def build_index_with_sentence_transformer(text):
    """Build a vector store index using SentenceTransformersTokenTextSplitter."""
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=50,
        model_name='sentence-transformers/all-mpnet-base-v2',
        tokens_per_chunk=384
    )
    docs = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in docs]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def query_multiple_retrievers(retrievers, questions):
    """Query multiple retrievers and aggregate responses."""
    #llm = OpenAI(temperature=0)  # Instantiate OpenAI LLM
    llm = OpenAI(temperature=0, model="gpt-4o-mini")
    responses = {}

    for question in questions:
        print(f"Querying: {question}")
        all_results = []
        for retriever in retrievers:
            try:
                # Perform similarity search for each retriever
                results = retriever.similarity_search(question, k=3)
                all_results.extend(results)
            except Exception as e:
                print(f"Retriever Error: {str(e)}")

        # Deduplicate results based on content
        unique_results = list({result.page_content: result for result in all_results}.values())

        # Check if there is any context retrieved
        if not unique_results:
            responses[question] = "Data Not Available"
            continue

        # Combine retrieved documents for context
        context = "\n".join([res.page_content for res in unique_results[:3]])
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        try:
            response = llm(prompt)  # Generate response using LLM
            responses[question] = response.strip()
        except Exception as e:
            responses[question] = f"Error: {str(e)}"

    return responses


def query_multiple_retrievers_with_custom_reranker(retrievers, questions):
    """Query multiple retrievers, rerank results, and aggregate responses."""
    llm = OpenAI(temperature=0)  # Instantiate OpenAI LLM
    responses = {}

    for question in questions:
        print(f"Querying: {question}")
        all_results = []

        for retriever in retrievers:
            try:
                # Perform similarity search for each retriever
                results = retriever.similarity_search(question, k=10)
                all_results.extend(results)
            except Exception as e:
                print(f"Retriever Error: {str(e)}")

        # Deduplicate results based on content
        unique_results = list({result.page_content: result for result in all_results}.values())

        # Check if there is any context retrieved
        if not unique_results:
            responses[question] = "Data Not Available"
            continue

        # Rerank results using the custom reranker
        ranked_results = custom_rerank(question, unique_results)

        # Take the top-ranked results for context
        top_results = [res[0] for res in ranked_results[:5]]
        context = "\n".join([res.page_content for res in top_results])

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        try:
            response = llm(prompt)  # Generate response using LLM
            responses[question] = response.strip()
        except Exception as e:
            responses[question] = f"Error: {str(e)}"

    return responses

def process_pdf(pdf_file, questions):
    """Complete processing of PDF and questions with multiple retrievers."""
    # Save the PDF to a temporary location
    pdf_path = f"/tmp/{pdf_file.filename}"
    pdf_file.save(pdf_path)

    try:
        # Extract content
        content = extract_text_from_pdf(pdf_path)

        # Build retrievers
        retriever1 = build_index_with_recursive_splitter(content)
        retriever2 = build_index_with_sentence_transformer(content)

        # Query retrievers
        answers = query_multiple_retrievers_with_custom_reranker([retriever1, retriever2], questions)
    finally:
        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    return answers


'''
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def build_index_from_text(text):
    """Build a vector store index from text."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in docs]

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def build_index_from_text_new(text):
    """Build a vector store index from text using SentenceTransformersTokenTextSplitter."""
    # Initialize the text splitter with desired parameters
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=50,  # Number of tokens to overlap between chunks
        model_name='sentence-transformers/all-mpnet-base-v2',  # Model for tokenization
        tokens_per_chunk=384  # Desired number of tokens per chunk
    )

    # Split the text into chunks
    docs = text_splitter.split_text(text)

    # Create Document objects for each chunk
    documents = [Document(page_content=chunk) for chunk in docs]

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore

def query_index(vectorstore, questions):
    """Query the vector store with questions."""
    llm = OpenAI(temperature=0)  # Instantiate OpenAI LLM
    responses = {}

    for question in questions:
        print(question)
        try:
            # Perform similarity search
            result = vectorstore.similarity_search(question, k=3)
            if result:
                # Combine retrieved document with the question into a proper prompt
                prompt = f"Context: {result[0].page_content}\nQuestion: {question}\nAnswer:"
                print(prompt)
                response = llm(prompt)  # Pass the prompt as a string
                responses[question] = response.strip()  # Extract and clean the response
            else:
                responses[question] = "Data Not Available"
        except Exception as e:
            responses[question] = f"Error: {str(e)}"

    return responses

def process_pdf(pdf_file, questions):
    """Complete processing of PDF and questions."""
    # Save the PDF to a temporary location
    pdf_path = f"/tmp/{pdf_file.filename}"
    pdf_file.save(pdf_path)

    try:
        # Extract content
        content = extract_text_from_pdf(pdf_path)
        # Build index
        vectorstore = build_index_from_text_new(content)
        # Query index
        answers = query_index(vectorstore, questions)
    finally:
        # Clean up temporary file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    return answers
'''