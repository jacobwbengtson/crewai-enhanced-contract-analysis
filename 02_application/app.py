# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
import streamlit as st
import uuid
import pandas as pd
import tempfile
import hashlib
import time
import numpy as np
import subprocess
import sys
import threading
import matplotlib.pyplot as plt
import re
import json
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import logging
logging.basicConfig(level=logging.INFO)
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
sys.path.append(('/home/cdsw/02_application/main.py'))
from main import ChromaDBStorage, process_documents, get_document_processor

# Set page config first - must be the first Streamlit command
st.set_page_config(
    page_title="Legal Document Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import from main.py


# Define the path to store uploaded contracts
UPLOAD_FOLDER = "/home/cdsw/02_application/contracts"
RESULTS_FOLDER = "/home/cdsw/02_application/results"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Add global variable to track if API key is configured
api_key_configured = False

# Initialize session state
if 'processed_doc' not in st.session_state:
    st.session_state.processed_doc = None
if 'all_processed_docs' not in st.session_state:
    st.session_state.all_processed_docs = []
if 'similar_docs' not in st.session_state:
    st.session_state.similar_docs = []
if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs = []
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None
if 'similarity_threshold' not in st.session_state:
    st.session_state.similarity_threshold = 0.5
if 'max_similar_docs' not in st.session_state:
    st.session_state.max_similar_docs = 5
if 'openai_endpoint' not in st.session_state:
    st.session_state.openai_endpoint = "https://api.openai.com/v1"
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'openai_model' not in st.session_state:
    st.session_state.openai_model = "gpt-4o"

# Initialize session state for status messages
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []


def add_status_message(message, message_type="info"):
    """Add a status message to be displayed."""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.status_messages.append({
        "message": message,
        "type": message_type,
        "timestamp": timestamp
    })

def ensure_string(obj):
    """Convert an object to string if it's not a string already."""
    if isinstance(obj, str):
        return obj
    elif hasattr(obj, '__str__'):
        return str(obj)
    else:
        return f"[Object of type {type(obj).__name__}]"

def ensure_api_key():
    """Ensure API key is set before processing documents."""
    global api_key_configured

    # Set environment variable directly from session state
    if st.session_state.openai_api_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
        api_key_configured = True
        return True
    elif os.environ.get("OPENAI_API_KEY"):
        api_key_configured = True
        return True
    else:
        api_key_configured = False
        return False


def handle_openai_error(func):
    """Decorator to handle OpenAI errors gracefully."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if "OpenAI API key" in str(e):
                st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
                return None
            else:
                st.error(f"Error: {str(e)}")
                return None
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

    return wrapper


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to the contracts folder and return the file path."""
    # Create a unique filename to avoid conflicts
    file_extension = os.path.splitext(uploaded_file.name)[1]
    secure_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename)

    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path, uploaded_file.name


def clean_document_text(text, file_path):
    """Clean document text to remove artifacts and metadata."""
    # Get file extension
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # PDF-specific cleaning
    if file_extension == '.pdf':
        # Remove PDF object references (e.g., "150 0 obj", "endobj")
        text = re.sub(r'\d+ \d+ obj.*?endobj', ' ', text, flags=re.DOTALL)

        # Remove PDF streams
        text = re.sub(r'stream\s.*?endstream', ' ', text, flags=re.DOTALL)

        # Remove PDF dictionary objects (e.g., << ... >>)
        text = re.sub(r'<<.*?>>', ' ', text, flags=re.DOTALL)

        # Remove PDF operators and commands
        text = re.sub(r'/[A-Za-z0-9_]+\s+', ' ', text)

        # Remove PDF metadata like CID, CMap entries
        text = re.sub(r'/(CID|CMap|Registry|Ordering|Supplement|CIDToGIDMap).*?def', ' ', text)

        # Remove "R" references
        text = re.sub(r'\d+ \d+ R', ' ', text)

        # Remove EvoPdf artifacts
        text = re.sub(r'EvoPdf_[a-zA-Z0-9_]+', '', text)

    # General cleaning for all document types

    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove any lines that are just numbers (page numbers, etc.)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    return text


@handle_openai_error
def process_document(file_path, original_filename):
    """Process a document using OpenAI."""
    global api_key_configured

    # Double check that API key is set
    if not api_key_configured and not ensure_api_key():
        st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
        return None

    # Initialize ChromaDB storage
    chroma_storage = ChromaDBStorage(
        db_path="/home/cdsw/02_application/chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    # Generate a unique document ID
    doc_id = f"doc_{hashlib.md5(original_filename.encode()).hexdigest()[:8]}"

    # Process the document
    with st.spinner(f"Processing {original_filename}..."):
        # Get the document processor
        document_processor = get_document_processor()
        processor_type = document_processor.__class__.__name__
        print(f"Using document processor: {processor_type}")

        processed_doc = document_processor.process_document(file_path, doc_id)

        # Store chunks in ChromaDB
        chroma_storage.add_texts(
            processed_doc["chunks"],
            processed_doc["metadatas"],
            processed_doc["ids"]
        )

        # Generate summary and analysis
        summary = document_processor.summarize(processed_doc["text"])
        analysis = document_processor.analyze(processed_doc["text"])

        # Store summary in ChromaDB
        chroma_storage.add_summary(
            doc_id=doc_id,
            source=file_path,
            filename=original_filename,
            summary=summary,
            analysis=analysis
        )

        # Save results to file
        summary_file = os.path.join(RESULTS_FOLDER, f"{os.path.splitext(original_filename)[0]}_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Document: {original_filename}\n")
            f.write(f"Processed as: {doc_id}\n\n")
            f.write(f"SUMMARY:\n{summary}\n\n")
            f.write(f"ANALYSIS:\n{analysis}")

    return {
        "doc_id": doc_id,
        "filename": original_filename,
        "file_path": file_path,
        "summary": summary,
        "analysis": analysis
    }


def find_similar_documents(processed_doc, max_results=5, similarity_threshold=0.5):
    """Find similar legal documents to the processed document using vector similarity."""
    chroma_storage = ChromaDBStorage(
        db_path="/home/cdsw/02_application/chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    # Get the summary text of the processed document to use as query
    # Convert CrewOutput to string if needed
    summary = ensure_string(processed_doc["summary"])
    query_text = summary

    # Search for similar summaries using vector similarity
    try:
        results = chroma_storage.summaries_collection.query(
            query_texts=[query_text],
            n_results=max_results + 1  # +1 because we'll filter out the current document
        )
    except Exception as e:
        st.error(f"Error searching for similar documents: {e}")
        return []

    # Filter out the current document and format results
    similar_docs = []

    # Check if results contains the expected keys
    if results and "ids" in results and "metadatas" in results:
        for i, doc_id in enumerate(results["ids"][0]):
            # Skip the current document
            metadata = results["metadatas"][0][i]
            if metadata.get("doc_id") == processed_doc["doc_id"]:
                continue

            # Get document details
            doc_id = metadata.get("doc_id", "unknown")
            filename = metadata.get("filename", "unknown")

            # Add distance/similarity score if available
            similarity_score = None
            if "distances" in results and len(results["distances"]) > 0:
                # Convert distance to similarity score (1 - distance)
                similarity_score = 1 - results["distances"][0][i]

                # Only include documents above the similarity threshold
                if similarity_score < similarity_threshold:
                    continue

            similar_docs.append({
                "doc_id": doc_id,
                "filename": filename,
                "similarity_score": similarity_score
            })

    return similar_docs

def get_document_summary(doc_id):
    """Get the summary for a legal document."""
    chroma_storage = ChromaDBStorage(
        db_path="/home/cdsw/02_application/chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    summary = chroma_storage.get_summary(doc_id)
    return summary


@handle_openai_error
@handle_openai_error
def compare_documents(new_doc, selected_docs):
    """Compare the new document with selected documents."""
    # Double check that API key is set
    if not api_key_configured and not ensure_api_key():
        st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
        return None

    # Get document processor
    document_processor = get_document_processor()

    # Get text from the new document
    try:
        with open(new_doc["file_path"], 'r', encoding='utf-8', errors='ignore') as f:
            new_text = f.read()

        # Debug info
        print(f"Loaded text from {new_doc['file_path']}, length: {len(new_text)} chars")
        print(f"Text preview: {new_text[:100]}...")

        if len(new_text) < 10:
            # Try with binary mode if text mode didn't work
            with open(new_doc["file_path"], 'rb') as f:
                binary_data = f.read()
                new_text = binary_data.decode('utf-8', errors='ignore')
                print(f"Loaded binary text, length: {len(new_text)} chars")
    except Exception as e:
        error_msg = f"Error reading file {new_doc['file_path']}: {e}"
        print(error_msg)
        return error_msg

    # Check if we have text to compare
    if not new_text or len(new_text) < 10:
        return "Error: Could not extract meaningful text from the uploaded document. The file may be corrupted or in an unsupported format."

    # Get summaries for selected documents
    summaries = []
    for doc_id, summary in selected_docs.items():
        # Parse the summary text
        if isinstance(summary, dict) and "text" in summary:
            summaries.append({
                "doc_id": doc_id,
                "text": summary["text"]
            })
        else:
            # Try to extract from the raw text
            try:
                if isinstance(summary, str):
                    parts = summary.split("SUMMARY:", 1)
                    if len(parts) > 1:
                        summary_text = parts[1].split("ANALYSIS:", 1)[0].strip()
                        analysis_text = parts[1].split("ANALYSIS:", 1)[1].strip() if "ANALYSIS:" in parts[1] else ""

                        summaries.append({
                            "doc_id": doc_id,
                            "text": f"SUMMARY:\n{summary_text}\n\nANALYSIS:\n{analysis_text}"
                        })
                else:
                    # Convert non-string summary to string
                    summaries.append({
                        "doc_id": doc_id,
                        "text": str(summary)
                    })
            except Exception as e:
                print(f"Error parsing summary for {doc_id}: {e}")
                # Add with minimal text to avoid errors
                summaries.append({
                    "doc_id": doc_id,
                    "text": f"Document ID: {doc_id}"
                })

    # Check if we have summaries to compare with
    if not summaries:
        return "Error: No document summaries found to compare with. Please select at least one document for comparison."

    # Compare documents
    comparison_result = document_processor.compare_with_summaries(
        new_text,
        summaries,
        focus_areas=["legal provisions", "obligations", "rights", "liability", "termination", "jurisdiction"]
    )

    # Convert CrewOutput to string if needed
    if hasattr(comparison_result, '__str__'):
        comparison_result = str(comparison_result)

    # Check if the result looks like JSON and format it if needed
    if isinstance(comparison_result, str) and (comparison_result.startswith('{') or comparison_result.startswith('[')):
        try:
            import json
            # Try to parse it as JSON
            data = json.loads(comparison_result)
            # Format it into a more readable format
            readable_result = format_comparison_json(data)
            return readable_result
        except json.JSONDecodeError:
            # Not valid JSON, return as is
            return comparison_result

    return comparison_result

def format_comparison_json(data):
    """Format comparison JSON data into a readable string format."""
    if isinstance(data, dict):
        if "comparison" in data:
            # Extract the main comparison
            result = "# Document Comparison Results\n\n"

            if "documents" in data:
                result += "## Documents Compared\n\n"
                for doc in data["documents"]:
                    result += f"- {doc.get('name', 'Unnamed document')} (ID: {doc.get('id', 'Unknown')})\n"
                result += "\n"

            if "similarities" in data:
                result += "## Similarities\n\n"
                for item in data["similarities"]:
                    result += f"- {item}\n"
                result += "\n"

            if "differences" in data:
                result += "## Differences\n\n"
                for item in data["differences"]:
                    result += f"- {item}\n"
                result += "\n"

            if "analysis" in data:
                result += "## Analysis\n\n"
                result += data["analysis"]
                result += "\n\n"

            return result

        # Generic dictionary formatting
        result = ""
        for key, value in data.items():
            key_str = key.replace("_", " ").title()
            result += f"## {key_str}\n\n"

            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            result += f"### {k.replace('_', ' ').title()}\n{v}\n\n"
                    else:
                        result += f"- {item}\n"
            elif isinstance(value, dict):
                for k, v in value.items():
                    result += f"### {k.replace('_', ' ').title()}\n{v}\n\n"
            else:
                result += f"{value}\n\n"

        return result

    elif isinstance(data, list):
        # Format list data
        result = ""
        for item in data:
            if isinstance(item, dict):
                for key, value in item.items():
                    result += f"### {key.replace('_', ' ').title()}\n{value}\n\n"
            else:
                result += f"- {item}\n"

        return result

    # Default fallback
    return str(data)

@handle_openai_error
def extract_legal_definitions(doc_id):
    """Extract legal definitions from a document with improved accuracy."""
    # Double check that API key is set
    if not api_key_configured and not ensure_api_key():
        st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
        return None

    # Get document processor
    document_processor = get_document_processor()

    chroma_storage = ChromaDBStorage(
        db_path="/home/cdsw/02_application/chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    # Get the document text
    if doc_id.startswith("doc_"):
        # Get the document from the database
        summary = chroma_storage.get_summary(doc_id)
        if not summary:
            return "Document not found in the database."

        if "metadata" in summary and "source" in summary["metadata"]:
            file_path = summary["metadata"]["source"]
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()

                # Clean the text of PDF artifacts
                cleaned_text = clean_document_text(raw_text, file_path)

            except Exception as e:
                return f"Error reading document: {e}"
        else:
            return "Document source file not found."
    else:
        return "Invalid document ID."

    # Extract legal definitions
    try:
        definitions = document_processor.extract_legal_definitions_improved(cleaned_text)
    except AttributeError:
        # Fall back to the regular method if the improved one doesn't exist
        # Apply our own post-processing to filter PDF artifacts
        definitions = document_processor.extract_legal_definitions(cleaned_text)
        # Check if the result contains PDF metadata artifacts
        pdf_artifacts = [
            "CIDSystemInfo", "CMapName", "CMapType", "CIDToGIDMap",
            "obj", "R", "def", "endobj", "stream", "endstream"
        ]

        # Check for PDF artifacts in the definitions
        contains_artifacts = any(artifact in definitions for artifact in pdf_artifacts)

        if contains_artifacts:
            # If PDF artifacts are detected, replace the entire output with a clearer message
            definitions = "No formal legal definitions were found in this document. The extracted text appears to contain technical metadata rather than legal content."

    return definitions


@handle_openai_error
def assess_legal_risks(doc_id):
    """Perform a legal risk assessment on a document."""
    # Double check that API key is set
    if not api_key_configured and not ensure_api_key():
        st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
        return None

    # Get document processor
    document_processor = get_document_processor()

    chroma_storage = ChromaDBStorage(
        db_path="/home/cdsw/02_application/chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    # Get the document text
    if doc_id.startswith("doc_"):
        # Get the document from the database
        summary = chroma_storage.get_summary(doc_id)
        if not summary:
            return "Document not found in the database."

        if "metadata" in summary and "source" in summary["metadata"]:
            file_path = summary["metadata"]["source"]
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                # Clean the text of PDF artifacts
                text = clean_document_text(text, file_path)
            except Exception as e:
                return f"Error reading document: {e}"
        else:
            return "Document source file not found."
    else:
        return "Invalid document ID."

    # Perform risk assessment
    risk_assessment = document_processor.assess_legal_risks(
        text,
        risk_categories=["contractual", "regulatory", "litigation", "intellectual property"]
    )
    return risk_assessment


def advanced_search_documents(query_text, max_results=5, search_chunks=True):
    """Search for legal documents or document chunks based on a text query."""
    chroma_storage = ChromaDBStorage(
        db_path="/home/cdsw/02_application/chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    if search_chunks:
        # Search in document chunks
        results = chroma_storage.search_chunks(query_text, n_results=max_results)
    else:
        # Search in document summaries
        results = chroma_storage.search_summaries(query_text, n_results=max_results)

    # Format and return results
    formatted_results = []

    # Check if results contains the expected keys
    if results and "ids" in results and "metadatas" in results:
        for i, doc_id in enumerate(results["ids"][0]):
            # Get document details
            metadata = results["metadatas"][0][i]
            result_text = results["documents"][0][i] if "documents" in results else ""

            # Get similarity score if available
            similarity_score = None
            if "distances" in results and len(results["distances"]) > 0:
                similarity_score = 1 - results["distances"][0][i]  # Convert distance to similarity

            # Format result
            formatted_results.append({
                "id": doc_id,
                "doc_id": metadata.get("doc_id", "unknown"),
                "filename": metadata.get("filename", metadata.get("source", "unknown")),
                "chunk": metadata.get("chunk", "N/A") if search_chunks else "summary",
                "text": result_text,
                "similarity_score": similarity_score
            })

    return formatted_results


def get_all_documents():
    """Get all legal documents in the database."""
    chroma_storage = ChromaDBStorage(
        db_path="/home/cdsw/02_application/chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    # Get all summaries
    all_summaries = chroma_storage.get_all_summaries()

    # Format and return document list
    documents = []

    if all_summaries and "ids" in all_summaries and "metadatas" in all_summaries:
        for i, doc_id in enumerate(all_summaries["ids"]):
            metadata = all_summaries["metadatas"][i]

            documents.append({
                "doc_id": metadata.get("doc_id", "unknown"),
                "filename": metadata.get("filename", "unknown"),
                "source": metadata.get("source", "unknown")
            })

    return documents


def check_database_has_documents():
    """Check if there are any documents in the database."""
    try:
        chroma_storage = ChromaDBStorage(
            db_path="/home/cdsw/02_application/chromadb",
            chunks_collection="document_chunks",
            summaries_collection="document_summaries"
        )

        # Try to get all summaries
        all_summaries = chroma_storage.get_all_summaries()

        # Check if there are any documents
        if all_summaries and "ids" in all_summaries and len(all_summaries["ids"]) > 0:
            return True
        return False
    except Exception as e:
        st.error(f"Error checking database: {e}")
        return False


def run_initial_processing():
    """Run the main.py script to process initial documents."""
    try:
        add_status_message("No documents found in the database. Running initial document processing...", "info")

        # Call the process_documents function directly
        with st.spinner("Processing legal documents in 'contracts' folder. This may take a while..."):
            process_documents()

        add_status_message("Initial document processing completed!", "success")
        return True
    except Exception as e:
        error_msg = f"Error during initial document processing: {e}"
        add_status_message(error_msg, "error")
        import traceback
        add_status_message(traceback.format_exc(), "error")
        return False

def create_risk_matrix(risks):
    """Create a risk matrix visualization for legal risks."""
    # Convert severity and likelihood to numeric values
    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    likelihood_map = {"Low": 1, "Medium": 2, "High": 3}

    # Create dataframe
    risk_df = pd.DataFrame(risks)
    risk_df["severity_val"] = risk_df["severity"].map(severity_map)
    risk_df["likelihood_val"] = risk_df["likelihood"].map(likelihood_map)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create risk matrix
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.5, 3.5)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add risk zones
    # Low risk (green)
    ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, color='green', alpha=0.3))
    # Medium risk (yellow)
    ax.add_patch(plt.Rectangle((0.5, 1.5), 1, 1, color='yellow', alpha=0.3))
    ax.add_patch(plt.Rectangle((1.5, 0.5), 1, 1, color='yellow', alpha=0.3))
    ax.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, color='yellow', alpha=0.3))
    # High risk (red)
    ax.add_patch(plt.Rectangle((0.5, 2.5), 1, 1, color='red', alpha=0.3))
    ax.add_patch(plt.Rectangle((1.5, 2.5), 2, 1, color='red', alpha=0.3))
    ax.add_patch(plt.Rectangle((2.5, 0.5), 1, 3, color='red', alpha=0.3))

    # Plot risks
    for i, risk in risk_df.iterrows():
        ax.scatter(risk["likelihood_val"], risk["severity_val"], s=100, color='blue')
        ax.annotate(risk["name"], (risk["likelihood_val"], risk["severity_val"]),
                    xytext=(5, 5), textcoords='offset points')

    # Set labels
    ax.set_xlabel('Likelihood')
    ax.set_ylabel('Severity')
    ax.set_title('Legal Risk Matrix')

    # Set ticks
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Medium', 'High'])

    return fig


def parse_risk_assessment(risk_text):
    """Parse risk assessment text into structured data for visualization."""
    risks = []

    # Convert CrewOutput to string if needed
    if hasattr(risk_text, '__str__'):
        risk_text = str(risk_text)

    # Handle different formats
    if not isinstance(risk_text, str):
        print(f"Warning: risk_text is not a string but a {type(risk_text)}")
        return []

    # Now we can safely split the string
    lines = risk_text.split('\n')

    current_risk = None
    for line in lines:
        line = line.strip()
        if line.startswith('Risk:') or line.startswith('- Risk:'):
            # New risk found
            if current_risk and 'name' in current_risk and 'severity' in current_risk and 'likelihood' in current_risk:
                risks.append(current_risk)
            current_risk = {'name': line.split(':', 1)[1].strip()}
        elif line.startswith('Severity:') and current_risk:
            severity = line.split(':', 1)[1].strip()
            if any(level in severity for level in ['High', 'Medium', 'Low']):
                for level in ['High', 'Medium', 'Low']:
                    if level in severity:
                        current_risk['severity'] = level
                        break
        elif line.startswith('Likelihood:') and current_risk:
            likelihood = line.split(':', 1)[1].strip()
            if any(level in likelihood for level in ['High', 'Medium', 'Low']):
                for level in ['High', 'Medium', 'Low']:
                    if level in likelihood:
                        current_risk['likelihood'] = level
                        break

    # Add the last risk if it exists
    if current_risk and 'name' in current_risk and 'severity' in current_risk and 'likelihood' in current_risk:
        risks.append(current_risk)

    return risks


def display_legal_definitions(definitions_text):
    """Parse and display legal definitions in a structured format."""
    st.subheader("Legal Definitions")

    # Convert CrewOutput to string if needed
    if hasattr(definitions_text, '__str__'):
        definitions_text = str(definitions_text)

    # Handle non-string types
    if not isinstance(definitions_text, str):
        st.error(f"Unexpected type for definitions: {type(definitions_text)}")
        return

    # Check if definitions contains the "no definitions found" message
    if "No formal legal definitions" in definitions_text:
        st.info(definitions_text)
        return

    # Simple parsing logic - this could be enhanced with regex or NLP
    sections = definitions_text.split('\n\n')

    # Keep track of displayed terms to avoid duplicates
    displayed_terms = set()

    for section in sections:
        if not section.strip():
            continue

        lines = section.split('\n')
        if not lines:
            continue

        # Assume first line is the term
        term = lines[0].strip()
        if ':' in term:
            term = term.split(':', 1)[0].strip()

        # Skip if empty, too long to be a term, or PDF artifact-like term
        if (not term or len(term) > 100 or
                term.startswith('/') or
                term.startswith('**') or
                any(artifact in term for artifact in ["CID", "obj", "Map", "stream", "EvoPdf", "endobj"])):
            continue

        # Skip if we've already displayed this term
        if term in displayed_terms:
            continue

        displayed_terms.add(term)

        # Create expander for each term
        with st.expander(term):
            st.write(section)

def save_settings_to_file():
    """Save current settings to a JSON file."""
    settings = {
        "openai_endpoint": st.session_state.openai_endpoint,
        "openai_api_key": st.session_state.openai_api_key,
        "openai_model": st.session_state.openai_model
    }

    try:
        os.makedirs("config", exist_ok=True)
        with open("config/settings.json", "w") as f:
            # Actually save the API key this time
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def load_settings_from_file():
    """Load settings from JSON file."""
    try:
        if os.path.exists("config/settings.json"):
            with open("config/settings.json", "r") as f:
                settings = json.load(f)

                # Update session state with loaded settings
                if "openai_endpoint" in settings:
                    st.session_state.openai_endpoint = settings["openai_endpoint"]
                if "openai_model" in settings:
                    st.session_state.openai_model = settings["openai_model"]
                # Actually load the API key
                if "openai_api_key" in settings:
                    st.session_state.openai_api_key = settings["openai_api_key"]

            return True
        return False
    except Exception as e:
        print(f"Error loading settings: {e}")
        return False


def test_crewai_connection():
    """Test connection to OpenAI API via CrewAI."""
    try:
        # Ensure API key is set
        if not ensure_api_key():
            return {"success": False, "message": "API key is required"}

        # Log for debugging
        logging.info(f"Testing CrewAI connection with API key: {'*' * 5}{st.session_state.openai_api_key[-4:] if st.session_state.openai_api_key else 'None'}")
        logging.info(f"Using model: {st.session_state.openai_model}")
        logging.info(f"Using endpoint: {st.session_state.openai_endpoint}")

        # Initialize LLM with more error checking
        try:
            llm = ChatOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=st.session_state.openai_endpoint,
                model=st.session_state.openai_model,
                temperature=0
            )
            logging.info("LLM initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            return {"success": False, "message": f"Failed to initialize LLM: {str(e)}"}

        # Create a simple agent and task
        test_agent = Agent(
            role="Legal Document Tester",
            goal="Test connection to the API",
            backstory="I'm testing if the API connection works properly.",
            llm=llm,
            verbose=True
        )
        logging.info("Agent created successfully")

        test_task = Task(
            description="Confirm if you can connect to the API by saying 'Connection successful!'",
            agent=test_agent,
            expected_output="A confirmation message"
        )
        logging.info("Task created successfully")

        # Create a minimal crew
        test_crew = Crew(
            agents=[test_agent],
            tasks=[test_task],
            verbose=True,
            process=Process.sequential
        )
        logging.info("Crew created successfully")

        # Execute the crew
        logging.info("Starting crew kickoff...")
        result = test_crew.kickoff()
        logging.info(f"Crew kickoff completed. Result: {result}")

        return {"success": True, "message": f"Connection successful! Response: {result}"}

    except Exception as e:
        logging.error(f"Error testing CrewAI connection: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"success": False, "message": str(e)}


def check_environment():
    """Check environment variables and settings."""
    logging.info("Checking environment...")

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    logging.info(f"API key in environment: {'Yes' if api_key else 'No'}")

    # Check other critical settings
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Agents YAML exists: {os.path.exists('agents.yaml')}")
    logging.info(f"Tasks YAML exists: {os.path.exists('tasks.yaml')}")

    # Check for ChromaDB
    chroma_path = "/home/cdsw/02_application/chromadb"
    logging.info(f"ChromaDB path exists: {os.path.exists(chroma_path)}")

    # Print important paths
    logging.info(f"UPLOAD_FOLDER path: {os.path.abspath(UPLOAD_FOLDER)}")
    logging.info(f"RESULTS_FOLDER path: {os.path.abspath(RESULTS_FOLDER)}")


# Call this at the start of the main function
def main():
    st.title("Legal Document Analysis System")

    # Add debugging information
    check_environment()

def load_yaml_file(file_path: str):
    """Load YAML file."""
    print(f"Loading YAML file: {file_path}")
    try:
        import yaml
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        print(f"Successfully loaded YAML file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return {}


def display_settings_tab():
    """Display settings tab for OpenAI API configuration."""
    st.header("System Settings")

    with st.expander("OpenAI API Settings", expanded=True):
        st.info(
            "Configure OpenAI API settings for legal document analysis. An API key is required to process documents.")

        # OpenAI endpoint
        openai_endpoint = st.text_input(
            "OpenAI API Endpoint",
            value=st.session_state.openai_endpoint,
            help="The endpoint URL for OpenAI API (or compatible service)"
        )

        # OpenAI API key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Your OpenAI API key (required for document processing)"
        )

        # OpenAI model selection
        openai_models = [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

        openai_model = st.selectbox(
            "OpenAI/Compatible Model",
            options=openai_models,
            index=openai_models.index(
                st.session_state.openai_model) if st.session_state.openai_model in openai_models else 0,
            help="Select the model to use for legal document analysis"
        )

        # Add a warning if no API key is set
        if not st.session_state.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            st.warning("⚠️ No OpenAI API key detected. You must provide an API key to process documents.")

        # Save settings button
        if st.button("Save API Settings"):
            st.session_state.openai_endpoint = openai_endpoint
            st.session_state.openai_api_key = openai_api_key
            st.session_state.openai_model = openai_model

            # Save settings to a config file
            save_settings_to_file()

            # Set the API key in environment variable immediately
            if ensure_api_key():
                st.success("Settings saved successfully! OpenAI API key configured.")
            else:
                st.warning("Settings saved, but no API key provided. Document processing will not work.")

    with st.expander("CrewAI Configuration", expanded=True):
        st.info("View your CrewAI agents and tasks configurations for legal document analysis")

        # Display the current agents and tasks
        st.subheader("Current Agent Configurations")
        agents_yaml = load_yaml_file("agents.yaml")
        if agents_yaml:
            st.json(json.dumps(agents_yaml))
        else:
            st.warning("Could not load agents.yaml")

        st.subheader("Current Task Configurations")
        tasks_yaml = load_yaml_file("tasks.yaml")
        if tasks_yaml:
            st.json(json.dumps(tasks_yaml))
        else:
            st.warning("Could not load tasks.yaml")

        # Test connection button
        if st.button("Test OpenAI Connection"):
            if not st.session_state.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                st.error("OpenAI API key is required. Please enter your API key above.")
            else:
                test_result = test_crewai_connection()
                if test_result["success"]:
                    st.success(test_result["message"])
                else:
                    st.error(f"Connection failed: {test_result['message']}")


###########################################
# STREAMLIT UI
###########################################

def main():
    st.title("Legal Document Analysis System")

    # Load settings at startup
    load_settings_from_file()

    # Ensure API key is set
    ensure_api_key()

    # Check if there are documents in the database
    if not check_database_has_documents():
        # Check if there are documents in the contracts folder
        contracts_folder = "contracts"
        has_contracts = False

        if os.path.exists(contracts_folder):
            for root, dirs, files in os.walk(contracts_folder):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.pdf', '.docx', '.txt']:
                        has_contracts = True
                        break
                if has_contracts:
                    break

        if has_contracts:
            # Run initial processing if documents exist but aren't in the database
            run_initial_processing()
        else:
            add_status_message(
                "No legal documents found in the 'contracts' folder. Please upload documents to get started.",
                "warning")

    # Display status messages if any
    if st.session_state.status_messages:
        with st.expander("System Messages", expanded=True):
            for msg in st.session_state.status_messages:
                if msg["type"] == "success":
                    st.success(f"[{msg['timestamp']}] {msg['message']}")
                elif msg["type"] == "error":
                    st.error(f"[{msg['timestamp']}] {msg['message']}")
                elif msg["type"] == "warning":
                    st.warning(f"[{msg['timestamp']}] {msg['message']}")
                else:
                    st.info(f"[{msg['timestamp']}] {msg['message']}")

            if st.button("Clear Messages"):
                st.session_state.status_messages = []
                st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("Legal Document Processing")
        st.write("Upload legal documents to process and compare with existing documents.")

        # Display API key status
        if not api_key_configured:
            st.warning(
                "⚠️ No API key configured. Please add your OpenAI API key in the Settings tab before processing documents.")

        # File uploader - now accepts multiple files
        uploaded_files = st.file_uploader("Upload Legal Documents", type=["pdf", "docx", "txt"],
                                          accept_multiple_files=True)

        if uploaded_files and len(uploaded_files) > 0:
            # Process button
            if st.button("Process Legal Documents"):
                # Check for API key first
                if not ensure_api_key():
                    st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
                else:
                    processed_docs = []

                    for uploaded_file in uploaded_files:
                        try:
                            # Save the file
                            file_path, original_filename = save_uploaded_file(uploaded_file)
                            add_status_message(f"File saved: {original_filename}", "success")

                            # Process the document
                            processed_doc = process_document(file_path, original_filename)
                            if processed_doc:  # Check if processing was successful
                                processed_docs.append(processed_doc)
                                add_status_message(f"Legal document processed: {original_filename}", "success")
                            else:
                                add_status_message(f"Failed to process {original_filename}", "error")
                        except Exception as e:
                            error_msg = f"Error processing {uploaded_file.name}: {e}"
                            add_status_message(error_msg, "error")
                            import traceback
                            add_status_message(traceback.format_exc(), "error")

                    # After processing all documents, select the first one to display
                    if processed_docs:
                        st.session_state.all_processed_docs = processed_docs
                        st.session_state.processed_doc = processed_docs[0]

                        # Find similar documents for the currently selected document
                        st.session_state.similar_docs = find_similar_documents(
                            st.session_state.processed_doc,
                            max_results=st.session_state.max_similar_docs,
                            similarity_threshold=st.session_state.similarity_threshold
                        )

                        add_status_message(f"Processed {len(processed_docs)} legal documents", "success")
                        add_status_message(f"Found {len(st.session_state.similar_docs)} similar legal documents",
                                           "info")

                        # Reset selected docs and comparison
                        st.session_state.selected_docs = []
                        st.session_state.summaries = {}
                        st.session_state.comparison_result = None

                        # Force a rerun to update the UI
                        st.rerun()

        # Search options
        st.header("Search Options")

        # Similarity threshold slider
        st.session_state.similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="Minimum similarity score for documents to be considered similar (0 = show all, 1 = perfect match)"
        )

        # Max results slider
        st.session_state.max_similar_docs = st.slider(
            "Max Similar Documents",
            min_value=1,
            max_value=20,
            value=st.session_state.max_similar_docs,
            help="Maximum number of similar documents to display"
        )

        # Advanced search section
        st.header("Legal Document Search")
        search_query = st.text_input("Search Query", placeholder="Enter legal terms or keywords to search...")
        search_chunks = st.checkbox("Search in Document Chunks", value=True,
                                    help="When checked, searches in document chunks. Otherwise, searches in summaries.")

        if st.button("Search Legal Database"):
            if search_query:
                with st.spinner("Searching legal documents..."):
                    search_results = advanced_search_documents(
                        search_query,
                        max_results=st.session_state.max_similar_docs,
                        search_chunks=search_chunks
                    )

                    # Create a new tab for search results
                    st.session_state.search_results = search_results
            else:
                st.warning("Please enter a search query")

    # Main content - create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Document Processing & Legal Analysis",
        "Search Results",
        "Document Database",
        "Settings"
    ])

    # Tab 1: Document Processing & Legal Analysis
    with tab1:
        # If we have multiple documents, show a selector
        if hasattr(st.session_state, 'all_processed_docs') and len(st.session_state.all_processed_docs) > 1:
            st.subheader("Select Document to Analyze")
            doc_options = {doc["filename"]: i for i, doc in enumerate(st.session_state.all_processed_docs)}

            # Find which document is currently selected
            current_doc_index = next((i for i, doc in enumerate(st.session_state.all_processed_docs)
                                      if doc["doc_id"] == st.session_state.processed_doc["doc_id"]), 0)

            # Get the current document name
            current_doc_name = st.session_state.all_processed_docs[current_doc_index]["filename"]

            # Create a selectbox with the current document pre-selected
            selected_doc_name = st.selectbox(
                "Choose a document",
                options=list(doc_options.keys()),
                index=list(doc_options.keys()).index(current_doc_name)
            )

            # Update the current document when selection changes
            selected_index = doc_options[selected_doc_name]
            if st.session_state.processed_doc["doc_id"] != st.session_state.all_processed_docs[selected_index][
                "doc_id"]:
                st.session_state.processed_doc = st.session_state.all_processed_docs[selected_index]

                # Update similar documents for the newly selected document
                st.session_state.similar_docs = find_similar_documents(
                    st.session_state.processed_doc,
                    max_results=st.session_state.max_similar_docs,
                    similarity_threshold=st.session_state.similarity_threshold
                )
                st.rerun()  # Rerun to refresh the display with the new document

        if st.session_state.processed_doc:
            # Display document information
            st.header("Processed Legal Document")
            st.write(f"**Filename:** {st.session_state.processed_doc['filename']}")
            st.write(f"**Document ID:** {st.session_state.processed_doc['doc_id']}")

            # Create tabs for different types of analysis within the same tab
            analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
                "Summary & Analysis",
                "Legal Provisions",
                "Legal Definitions",
                "Risk Assessment",
                "Similar Documents"
            ])

            # Tab 1: Summary and Analysis
            with analysis_tab1:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Legal Summary")
                    st.write(st.session_state.processed_doc["summary"])

                with col2:
                    st.subheader("Legal Analysis")
                    st.write(st.session_state.processed_doc["analysis"])

            # Tab 2: Legal Provisions, Obligations & Rights
            with analysis_tab2:
                st.subheader("Legal Provisions, Obligations & Rights")

                # Extract from the main analysis
                if "analysis" in st.session_state.processed_doc:
                    analysis_text = st.session_state.processed_doc["analysis"]

                    # Extract provisions section
                    provisions_section = ""
                    if "Provisions" in analysis_text or "Key legal provisions" in analysis_text:
                        if "Provisions" in analysis_text:
                            provisions_section = analysis_text.split("Provisions", 1)[1]
                        else:
                            provisions_section = analysis_text.split("Key legal provisions", 1)[1]

                        # Split at the next major section
                        for next_section in ["Obligations", "Rights", "Definitions"]:
                            if next_section in provisions_section:
                                provisions_section = provisions_section.split(next_section, 1)[0]
                                break

                    # Extract obligations section
                    obligations_section = ""
                    if "Obligations" in analysis_text:
                        obligations_section = analysis_text.split("Obligations", 1)[1]
                        if "Rights" in obligations_section:
                            obligations_section = obligations_section.split("Rights", 1)[0]

                    # Extract rights section
                    rights_section = ""
                    if "Rights" in analysis_text:
                        rights_section = analysis_text.split("Rights", 1)[1]
                        # Split at the next major section
                        for next_section in ["Definitions", "Important clauses", "Liability", "Termination"]:
                            if next_section in rights_section:
                                rights_section = rights_section.split(next_section, 1)[0]
                                break

                    # Display
                    if provisions_section:
                        st.write("### Key Legal Provisions")
                        st.write(provisions_section)

                    st.write("### Obligations")
                    st.write(
                        obligations_section if obligations_section else "No specific obligations analysis available.")

                    st.write("### Rights")
                    st.write(rights_section if rights_section else "No specific rights analysis available.")
                else:
                    st.info("Process the document with legal analysis to see provisions, obligations and rights.")

            # Tab 3: Legal Definitions
            with analysis_tab3:
                st.subheader("Legal Definitions")

                # Add a button to extract definitions
                if st.button("Extract Legal Definitions"):
                    # Check for API key first
                    if not ensure_api_key():
                        st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
                    else:
                        with st.spinner("Extracting legal definitions..."):
                            if st.session_state.processed_doc and "doc_id" in st.session_state.processed_doc:
                                definitions = extract_legal_definitions(st.session_state.processed_doc["doc_id"])
                                if definitions:  # Check if extraction was successful
                                    st.session_state.definitions = definitions
                                else:
                                    st.error("Failed to extract legal definitions. Please check your API key.")
                            else:
                                st.error("No document available for definition extraction.")

                # Display definitions if available
                if hasattr(st.session_state, 'definitions'):
                    display_legal_definitions(st.session_state.definitions)
                else:
                    st.info("Click 'Extract Legal Definitions' to analyze defined terms in the document.")

            # Tab 4: Risk Assessment
            with analysis_tab4:
                st.subheader("Legal Risk Assessment")

                # Risk assessment button
                if st.button("Perform Legal Risk Assessment"):
                    # Check for API key first
                    if not ensure_api_key():
                        st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
                    else:
                        with st.spinner("Performing legal risk assessment..."):
                            risk_assessment = assess_legal_risks(st.session_state.processed_doc["doc_id"])
                            if risk_assessment:  # Check if assessment was successful
                                st.session_state.risk_assessment = risk_assessment
                            else:
                                st.error("Failed to perform risk assessment. Please check your API key.")

                # Display risk assessment if available
                if hasattr(st.session_state, 'risk_assessment'):
                    # Text assessment
                    st.write(st.session_state.risk_assessment)

                    # Risk matrix visualization
                    st.subheader("Risk Matrix Visualization")
                    risks = parse_risk_assessment(st.session_state.risk_assessment)
                    if risks:
                        fig = create_risk_matrix(risks)
                        st.pyplot(fig)
                    else:
                        st.info("No structured risk data available for visualization.")
                else:
                    st.info("Click 'Perform Legal Risk Assessment' to analyze risks in the document.")

            # Tab 5: Similar Documents and Comparison
            with analysis_tab5:
                st.subheader("Similar Legal Documents")

                if not st.session_state.similar_docs:
                    st.info("No similar legal documents found in the database.")
                else:
                    # Create a table of similar documents
                    st.write(f"Found {len(st.session_state.similar_docs)} similar legal documents:")

                    # Create a dataframe for display
                    similar_docs_df = pd.DataFrame(st.session_state.similar_docs)

                    # Format similarity score
                    if "similarity_score" in similar_docs_df.columns:
                        similar_docs_df["similarity_score"] = similar_docs_df["similarity_score"].apply(
                            lambda x: f"{x:.2f}" if x is not None else "N/A"
                        )

                        # Sort by similarity score
                        similar_docs_df = similar_docs_df.sort_values(by="similarity_score", ascending=False)

                    # Display table
                    st.dataframe(similar_docs_df)

                    # Create checkboxes for each document
                    st.write("Select documents to view summaries or compare:")

                    selected_indices = []
                    for i, doc in enumerate(st.session_state.similar_docs):
                        label = f"{doc['filename']} (ID: {doc['doc_id']})"
                        if "similarity_score" in doc and doc["similarity_score"] is not None:
                            score_display = f"{doc['similarity_score']:.2f}"
                            label += f" - Similarity: {score_display}"

                        if st.checkbox(label, key=f"doc_{doc['doc_id']}"):
                            selected_indices.append(i)

                    st.session_state.selected_docs = [st.session_state.similar_docs[i]["doc_id"] for i in
                                                      selected_indices]

                    # Buttons for actions
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("View Legal Summaries") and st.session_state.selected_docs:
                            st.session_state.summaries = {}
                            for doc_id in st.session_state.selected_docs:
                                summary = get_document_summary(doc_id)
                                if summary:
                                    st.session_state.summaries[doc_id] = summary

                            st.session_state.comparison_result = None

                    with col2:
                        if st.button("Compare Legal Documents") and st.session_state.selected_docs:
                            # Check for API key first
                            if not ensure_api_key():
                                st.error(
                                    "OpenAI API key is required. Please configure your API key in the Settings tab.")
                            else:
                                # Fetch summaries if not already fetched
                                if not st.session_state.summaries:
                                    for doc_id in st.session_state.selected_docs:
                                        summary = get_document_summary(doc_id)
                                        if summary:
                                            st.session_state.summaries[doc_id] = summary

                                # Compare documents
                                comparison_result = compare_documents(
                                    st.session_state.processed_doc,
                                    st.session_state.summaries
                                )

                                if comparison_result:  # Check if comparison was successful
                                    st.session_state.comparison_result = comparison_result
                                else:
                                    st.error("Failed to compare documents. Please check your API key.")

                # Display summaries
                if st.session_state.summaries:
                    st.header("Legal Document Summaries")

                    for doc_id, summary in st.session_state.summaries.items():
                        st.subheader(
                            f"Document: {next((doc['filename'] for doc in st.session_state.similar_docs if doc['doc_id'] == doc_id), doc_id)}")

                        if isinstance(summary, dict) and "text" in summary:
                            # Display the summary text
                            st.write(summary["text"])
                        else:
                            # Try to extract from the raw text
                            parts = summary.split("SUMMARY:", 1)
                            if len(parts) > 1:
                                summary_text = parts[1].split("ANALYSIS:", 1)[0].strip()
                                analysis_text = parts[1].split("ANALYSIS:", 1)[1].strip() if "ANALYSIS:" in parts[
                                    1] else ""

                                st.write("**Legal Summary:**")
                                st.write(summary_text)

                                st.write("**Legal Analysis:**")
                                st.write(analysis_text)
                            else:
                                st.write(summary)

                        st.markdown("---")

                # Display comparison result
                if st.session_state.comparison_result:
                    st.header("Legal Document Comparison")
                    st.write(st.session_state.comparison_result)

        else:
            # Display instructions
            st.info("Upload legal document(s) and click 'Process Legal Documents' to start.")
            st.write("""
            **This system allows you to:**
            1. Upload multiple legal documents (PDF, DOCX, TXT) simultaneously
            2. Process them to extract text, generate legal summaries, and store in a vector database
            3. Analyze key legal provisions, obligations, and rights
            4. Extract and analyze legal definitions with improved accuracy
            5. Perform risk assessment of legal documents
            6. Find similar legal documents in the database
            7. Compare documents to identify legal similarities and differences
            """)

            if not api_key_configured:
                st.warning("⚠️ Before processing documents, please configure your OpenAI API key in the Settings tab.")

    # Tab 2: Search Results
    with tab2:
        if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
            st.header("Legal Search Results")

            # Create a dataframe for display
            results_df = pd.DataFrame(st.session_state.search_results)

            # Format similarity score
            if "similarity_score" in results_df.columns:
                results_df["similarity_score"] = results_df["similarity_score"].apply(
                    lambda x: f"{x:.2f}" if x is not None else "N/A"
                )

            # Display table of results
            st.dataframe(results_df[["doc_id", "filename", "chunk", "similarity_score"]])

            # Display individual results
            for i, result in enumerate(st.session_state.search_results):
                # Format the similarity score
                score_display = f"{result['similarity_score']:.2f}" if result['similarity_score'] is not None else "N/A"
                with st.expander(f"Result {i + 1}: {result['filename']} - {result['doc_id']} (Score: {score_display})"):
                    st.write("**Document ID:** " + result["doc_id"])
                    st.write("**Filename:** " + result["filename"])

                    if result["chunk"] != "summary":
                        st.write(f"**Chunk:** {result['chunk']}")

                    st.write("**Content:**")
                    st.write(result["text"])

                    # Button to view full document summary
                    if st.button(f"View Full Legal Document", key=f"view_summary_{i}"):
                        summary = get_document_summary(result["doc_id"])
                        if summary:
                            st.session_state.view_summary = {
                                "doc_id": result["doc_id"],
                                "filename": result["filename"],
                                "summary": summary
                            }

            # Display selected summary if any
            if hasattr(st.session_state, 'view_summary') and st.session_state.view_summary:
                st.header(f"Legal Document Summary: {st.session_state.view_summary['filename']}")

                summary = st.session_state.view_summary["summary"]

                if isinstance(summary, dict) and "text" in summary:
                    # Display the summary text
                    st.write(summary["text"])
                else:
                    # Try to extract from the raw text
                    parts = summary.split("SUMMARY:", 1)
                    if len(parts) > 1:
                        summary_text = parts[1].split("ANALYSIS:", 1)[0].strip()
                        analysis_text = parts[1].split("ANALYSIS:", 1)[1].strip() if "ANALYSIS:" in parts[1] else ""

                        st.write("**Legal Summary:**")
                        st.write(summary_text)

                        st.write("**Legal Analysis:**")
                        st.write(analysis_text)
                    else:
                        st.write(summary)
        else:
            st.info("Use the legal document search in the sidebar to search for documents.")

    # Tab 3: Document Database
    with tab3:
        st.header("Legal Document Database")

        # Add a refresh button
        if st.button("Refresh Document List"):
            st.rerun()

        # Get all documents in the database
        documents = get_all_documents()

        if documents:
            st.write(f"Found {len(documents)} legal documents in the database:")

            # Create a dataframe for display
            docs_df = pd.DataFrame(documents)

            # Display table
            st.dataframe(docs_df)

            # Select documents to view
            selected_doc_id = st.selectbox(
                "Select a document to view its legal summary:",
                options=[doc["doc_id"] for doc in documents],
                format_func=lambda x: next((f"{doc['filename']} (ID: {x})" for doc in documents if doc["doc_id"] == x),
                                           x)
            )

            if st.button("View Legal Document Summary"):
                summary = get_document_summary(selected_doc_id)
                if summary:
                    st.subheader(
                        f"Legal summary for document: {next((doc['filename'] for doc in documents if doc['doc_id'] == selected_doc_id), selected_doc_id)}")

                    if isinstance(summary, dict) and "text" in summary:
                        # Display the summary text
                        st.write(summary["text"])
                    else:
                        # Try to extract from the raw text
                        parts = summary.split("SUMMARY:", 1)
                        if len(parts) > 1:
                            summary_text = parts[1].split("ANALYSIS:", 1)[0].strip()
                            analysis_text = parts[1].split("ANALYSIS:", 1)[1].strip() if "ANALYSIS:" in parts[1] else ""

                            st.write("**Legal Summary:**")
                            st.write(summary_text)

                            st.write("**Legal Analysis:**")
                            st.write(analysis_text)
                        else:
                            st.write(summary)

            # Add option to run process_documents again
            st.markdown("---")
            st.subheader("Process Additional Legal Documents")
            if st.button("Process Legal Documents in 'contracts' Folder"):
                # Check for API key first
                if not ensure_api_key():
                    st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
                else:
                    try:
                        with st.spinner("Processing legal documents in 'contracts' folder. This may take a while..."):
                            result = process_documents()
                        add_status_message(
                            f"Legal document processing completed! Processed {len(result) if result else 0} documents.",
                            "success")
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error processing legal documents: {e}"
                        add_status_message(error_msg, "error")
                        import traceback
                        add_status_message(traceback.format_exc(), "error")
        else:
            st.info("No legal documents found in the database.")

            # Add option to run process_documents
            st.subheader("Process Legal Documents")
            if st.button("Process Legal Documents in 'contracts' Folder"):
                # Check for API key first
                if not ensure_api_key():
                    st.error("OpenAI API key is required. Please configure your API key in the Settings tab.")
                else:
                    try:
                        with st.spinner("Processing legal documents in 'contracts' folder. This may take a while..."):
                            result = process_documents()
                        add_status_message(
                            f"Legal document processing completed! Processed {len(result) if result else 0} documents.",
                            "success")
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Error processing legal documents: {e}"
                        add_status_message(error_msg, "error")
                        import traceback
                        add_status_message(traceback.format_exc(), "error")

    # Tab 4: Settings
    with tab4:
        display_settings_tab()


if __name__ == "__main__":
    main()