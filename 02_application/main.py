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
import yaml
import requests
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import docx
import PyPDF2
import openai
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class ChromaDBStorage:
    """Storage for document chunks and summaries in ChromaDB."""

    def __init__(self, db_path: str = "./chromadb", chunks_collection: str = "document_chunks",
                 summaries_collection: str = "document_summaries"):
        """Initialize ChromaDB storage."""
        print(f"Initializing ChromaDB storage at {os.path.abspath(db_path)}")
        print(f"Chunks collection: {chunks_collection}")
        print(f"Summaries collection: {summaries_collection}")

        self.db_path = db_path
        self.chunks_collection_name = chunks_collection
        self.summaries_collection_name = summaries_collection

        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        print(f"ChromaDB directory exists: {os.path.exists(db_path)}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        print("ChromaDB client initialized")

        # Create embedding function
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("Embedding function created")

        # Get or create chunks collection
        try:
            self.chunks_collection = self.client.get_collection(
                name=chunks_collection,
                embedding_function=self.ef
            )
            print(f"Retrieved existing chunks collection: {chunks_collection}")
        except:
            print(f"Chunks collection not found, creating: {chunks_collection}")
            self.chunks_collection = self.client.create_collection(
                name=chunks_collection,
                embedding_function=self.ef
            )
            print(f"Created new chunks collection: {chunks_collection}")

        # Get or create summaries collection
        try:
            self.summaries_collection = self.client.get_collection(
                name=summaries_collection,
                embedding_function=self.ef
            )
            print(f"Retrieved existing summaries collection: {summaries_collection}")
        except:
            print(f"Summaries collection not found, creating: {summaries_collection}")
            self.summaries_collection = self.client.create_collection(
                name=summaries_collection,
                embedding_function=self.ef
            )
            print(f"Created new summaries collection: {summaries_collection}")

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Add texts to the chunks collection."""
        print(f"Adding {len(texts)} chunks to ChromaDB collection {self.chunks_collection_name}")
        print(f"First chunk preview: {texts[0][:100]}..." if texts else "No chunks to add")
        print(f"First metadata: {metadatas[0]}" if metadatas else "No metadata")
        print(f"First ID: {ids[0]}" if ids else "No IDs")

        try:
            self.chunks_collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(texts)} chunks to ChromaDB")

            # Verify addition
            result = self.chunks_collection.get()
            print(f"Chunks collection now contains {len(result['ids'])} items")

            return f"Added {len(texts)} chunks to ChromaDB collection {self.chunks_collection_name}"
        except Exception as e:
            print(f"Error adding texts to ChromaDB: {e}")
            raise

    def add_summary(self, doc_id: str, source: str, filename: str,
                    summary: str, analysis: str):
        """Add document summary to the summaries collection."""
        print(f"Adding summary for legal document {doc_id} to ChromaDB")

        # Create a combined document with summary and analysis
        combined_text = f"SUMMARY:\n{summary}\n\nANALYSIS:\n{analysis}"

        # Create metadata
        metadata = {
            "doc_id": doc_id,
            "source": source,
            "filename": filename,
            "type": "legal_summary_analysis"
        }

        # Create ID for the summary
        summary_id = f"{doc_id}_summary"

        try:
            self.summaries_collection.add(
                documents=[combined_text],
                metadatas=[metadata],
                ids=[summary_id]
            )
            print(f"Successfully added legal summary for document {doc_id} to ChromaDB")

            # Verify addition
            result = self.summaries_collection.get()
            print(f"Summaries collection now contains {len(result['ids'])} items")

            return f"Added legal summary for document {doc_id} to ChromaDB collection {self.summaries_collection_name}"
        except Exception as e:
            print(f"Error adding legal summary to ChromaDB: {e}")
            raise

    def search_chunks(self, query: str, n_results: int = 5):
        """Search for similar texts in chunks collection."""
        print(f"Searching for chunks with query: {query}")
        results = self.chunks_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        print(f"Found {len(results['ids'][0])} matching chunks")
        return results

    def search_summaries(self, query: str, n_results: int = 5):
        """Search for similar summaries."""
        print(f"Searching for legal summaries with query: {query}")
        results = self.summaries_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        print(f"Found {len(results['ids'][0])} matching legal summaries")
        return results

    def get_all_summaries(self):
        """Get all document summaries."""
        print(f"Retrieving all legal document summaries")
        results = self.summaries_collection.get()
        print(f"Retrieved {len(results['ids'])} legal summaries")
        return results

    def get_summary(self, doc_id: str):
        """Get summary for a specific legal document."""
        print(f"Retrieving legal summary for document {doc_id}")
        summary_id = f"{doc_id}_summary"
        try:
            result = self.summaries_collection.get(ids=[summary_id])
            if result["ids"]:
                print(f"Found legal summary for document {doc_id}")
                return {
                    "doc_id": doc_id,
                    "summary_id": summary_id,
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0]
                }
            else:
                print(f"No legal summary found for document {doc_id}")
                return None
        except Exception as e:
            print(f"Error retrieving legal summary for document {doc_id}: {e}")
            return None


class OpenAIDocumentProcessor:
    """Processor for legal documents using OpenAI API."""

    def __init__(self, model_name: str = "gpt-4o-latest", max_length: int = 5000, api_key: str = None,
                 api_base: str = None):
        """Initialize document processor."""
        print(f"Initializing OpenAI legal document processor with model: {model_name}")
        self.model_name = model_name
        self.max_length = max_length

        # Configure API settings
        if api_key:
            openai.api_key = api_key

        # Set custom API base URL if provided
        if api_base:
            openai.base_url = api_base
            print(f"Using custom API base URL: {api_base}")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    def extract_text(self, file_path: str) -> str:
        """Extract text from various document formats."""
        print(f"Extracting text from legal document: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_from_docx(file_path)
        elif file_extension == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        print(f"Extracting text from PDF: {file_path}")
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            print(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            raise

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        print(f"Extracting text from DOCX: {file_path}")
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            text = '\n'.join(full_text)
            print(f"Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            raise

    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT files."""
        print(f"Extracting text from TXT: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            print(f"Extracted {len(text)} characters from TXT")
            return text
        except Exception as e:
            print(f"Error extracting text from TXT: {e}")
            raise

    def clean_document_text(self, text: str, file_path: str) -> str:
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

    def process_document(self, file_path: str, doc_id: str) -> Dict[str, Any]:
        """Process a legal document and prepare it for storage and analysis."""
        print(f"Processing legal document: {file_path}, ID: {doc_id}")

        # Extract text from document
        text = self.extract_text(file_path)
        print(f"Extracted text length: {len(text)} characters")

        # Split text into chunks
        print("Splitting text into chunks for legal analysis...")
        chunks = self.text_splitter.split_text(text)
        print(f"Created {len(chunks)} chunks")

        # Prepare metadata
        print("Preparing legal document metadata...")
        metadatas = [{"source": file_path, "doc_id": doc_id, "chunk": i} for i in range(len(chunks))]

        # Prepare IDs
        print("Preparing IDs...")
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        return {
            "text": text,
            "chunks": chunks,
            "metadatas": metadatas,
            "ids": ids,
            "doc_id": doc_id
        }

    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt."""
        try:
            print(f"Calling OpenAI API with model: {self.model_name}")
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a legal expert assistant specialized in analyzing legal documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"Error calling OpenAI API: {str(e)}"
            print(error_msg)
            # Add more detailed error information for debugging custom endpoints
            if "api_base" in str(e) or "endpoint" in str(e) or "URL" in str(e) or "host" in str(e):
                error_msg += "\nPossible API endpoint configuration issue. Please check your API settings."
            elif "model" in str(e) or "engine" in str(e):
                error_msg += "\nPossible model configuration issue. The specified model may not exist or you may not have access to it."
            elif "key" in str(e) or "authorization" in str(e) or "auth" in str(e):
                error_msg += "\nPossible API key issue. Please check that your API key is valid and has appropriate permissions."
            return error_msg

    def summarize(self, text: str, max_length: int = None) -> str:
        """Summarize legal text using OpenAI."""
        print(f"Summarizing legal text, length: {len(text)} characters")
        if max_length is None:
            max_length = self.max_length

        # Limit text to max_length
        if len(text) > max_length:
            print(f"Text too long, truncating to {max_length} characters")
            text = text[:max_length]

        # Prepare prompt for legal summarization
        prompt = f"""
        Please provide a concise legal summary of the following document:

        {text}

        Focus specifically on:
        1. The type of legal document (contract, agreement, policy, etc.)
        2. Parties involved and their legal responsibilities
        3. Key legal provisions, obligations, and rights
        4. Important deadlines or dates
        5. Legal remedies and dispute resolution mechanisms
        6. Governing law and jurisdiction
        7. Any unusual or potentially problematic legal terms

        Format the summary with appropriate legal terminology and structure.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for legal summarization, model: {self.model_name}")
        return self._call_openai_api(prompt)

    def analyze(self, text: str, analysis_depth: str = "detailed") -> str:
        """Analyze legal text using OpenAI."""
        print(f"Analyzing legal text, length: {len(text)} characters, depth: {analysis_depth}")

        # Limit text to max_length
        if len(text) > self.max_length:
            print(f"Text too long, truncating to {self.max_length} characters")
            text = text[:self.max_length]

        # Prepare prompt for legal analysis
        prompt = f"""
        Please perform a {analysis_depth} legal analysis of the following document:

        {text}

        Identify and extract the following elements:
        1. Document type and purpose
        2. Parties and their legal roles
        3. Key legal provisions
        4. Legal obligations for each party
        5. Legal rights for each party
        6. Defined terms and their legal implications
        7. Liability clauses and risk allocation
        8. Termination provisions
        9. Governing law and jurisdiction
        10. Dispute resolution mechanisms
        11. Potential legal risks or issues
        12. Unusual or non-standard legal provisions
        13. Legal enforceability considerations

        Format your response in a structured manner using appropriate legal terminology.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for legal analysis, model: {self.model_name}")
        return self._call_openai_api(prompt)

    def compare_documents(self, texts: List[str], focus_areas: List[str]) -> str:
        """Compare multiple legal documents using OpenAI."""
        print(f"Comparing {len(texts)} legal documents, focus areas: {focus_areas}")

        # Prepare prompt for legal comparison
        prompt = f"""
        Please compare the following legal documents, focusing on {', '.join(focus_areas)}:

        """

        for i, text in enumerate(texts):
            # Limit each text to max_length / number of documents
            max_text_length = self.max_length // len(texts)
            if len(text) > max_text_length:
                print(f"Document {i + 1} too long, truncating to {max_text_length} characters")
                text = text[:max_text_length]

            prompt += f"\nDocument {i + 1}:\n{text}\n"

        prompt += f"""
        Provide a detailed legal comparison highlighting key similarities and differences
        in the following areas: {', '.join(focus_areas)}.

        Specifically analyze:
        1. Which document provides more favorable terms for each party
        2. Differences in legal obligations and rights
        3. Differences in risk allocation and liability
        4. Variations in legal remedies and dispute resolution
        5. Potential legal conflicts between the documents

        Format your analysis with clear legal reasoning and appropriate legal terminology.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for legal comparison, model: {self.model_name}")
        return self._call_openai_api(prompt)

    def compare_with_summaries(self, new_text: str, summaries: List[Dict[str, Any]],
                               focus_areas: List[str] = None) -> str:
        """Compare a new legal document with existing legal summaries."""
        print(f"Comparing new legal document with {len(summaries)} existing summaries")

        if not focus_areas:
            focus_areas = ["legal provisions", "obligations", "rights", "liability", "termination", "jurisdiction"]

        # Summarize the new document first
        new_summary = self.summarize(new_text)

        # Prepare prompt for comparison
        prompt = f"""
        Please compare this new legal document:

        NEW DOCUMENT SUMMARY:
        {new_summary}

        With the following existing legal document summaries:

        """

        for i, summary in enumerate(summaries):
            doc_id = summary.get("doc_id", f"doc_{i + 1}")
            summary_text = summary.get("text", "")

            # Limit each summary if needed
            max_text_length = (self.max_length // (len(summaries) + 1))
            if len(summary_text) > max_text_length:
                print(f"Summary {i + 1} too long, truncating")
                summary_text = summary_text[:max_text_length]

            prompt += f"\nEXISTING DOCUMENT {i + 1} (ID: {doc_id}):\n{summary_text}\n"

        prompt += f"""
        Please provide a detailed legal comparison focusing on:
        1. Legal similarities between the new document and each existing document
        2. Legal conflicts or contradictions between provisions in the new document and existing documents
        3. Unique legal elements in the new document not found in existing documents
        4. Specific {', '.join(focus_areas)} that overlap or conflict
        5. Legal risk assessment of inconsistencies between documents
        6. Which document provides more favorable terms for each party

        Be specific about which document ID contains conflicting or similar legal provisions.
        Use appropriate legal terminology and structure your analysis in a legally meaningful way.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for legal summary comparison, model: {self.model_name}")
        return self._call_openai_api(prompt)

    def assess_legal_risks(self, text: str, risk_categories: List[str] = None) -> str:
        """Assess legal risks in a document."""
        print(f"Assessing legal risks, text length: {len(text)} characters")

        if not risk_categories:
            risk_categories = ["contractual", "regulatory", "litigation", "intellectual property"]

        # Limit text to max_length
        if len(text) > self.max_length:
            print(f"Text too long, truncating to {self.max_length} characters")
            text = text[:self.max_length]

        # Prepare prompt for risk assessment
        prompt = f"""
        Please assess the legal risks in the following document:

        {text}

        Conduct a thorough legal risk assessment focusing on these categories:
        {', '.join(risk_categories)}

        For each identified risk:
        1. Describe the specific risk and the relevant document section
        2. Assess the severity (High/Medium/Low)
        3. Assess the likelihood (High/Medium/Low)
        4. Provide potential mitigation strategies
        5. Identify potential legal consequences if not addressed

        Format your response as a structured legal risk assessment report.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for legal risk assessment, model: {self.model_name}")
        return self._call_openai_api(prompt)

    def extract_legal_definitions(self, text: str) -> str:
        """Extract and analyze legal definitions from a document."""
        print(f"Extracting legal definitions, text length: {len(text)} characters")

        # Limit text to max_length
        if len(text) > self.max_length:
            print(f"Text too long, truncating to {self.max_length} characters")
            text = text[:self.max_length]

        # Prepare prompt for definition extraction
        prompt = f"""
        Please extract and analyze all defined legal terms from the following document:

        {text}

        For each defined term:
        1. Extract the exact definition as stated in the document
        2. Provide the section or clause where it appears
        3. Analyze whether the definition is clear or potentially ambiguous
        4. Identify any inconsistencies if the term is defined or used differently elsewhere
        5. Note any unusual or non-standard definitions that differ from common legal usage

        Format your response as a legal terminology glossary with analysis.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for legal definition extraction, model: {self.model_name}")
        return self._call_openai_api(prompt)

    def extract_legal_definitions_improved(self, text: str) -> str:
        """Extract and analyze legal definitions from a document with improved accuracy."""
        print(f"Extracting legal definitions with improved method, text length: {len(text)} characters")

        # Limit text to max_length
        if len(text) > self.max_length:
            print(f"Text too long, truncating to {self.max_length} characters")
            text = text[:self.max_length]

        # Improved prompt for definition extraction
        prompt = f"""
        Analyze the following legal document and extract ONLY actual legal terms that are formally defined within the document.

        {text}

        IMPORTANT INSTRUCTIONS:

        1. ONLY extract terms that are explicitly defined in the document using patterns like:
           - "X means/refers to..."
           - "X is defined as..."
           - "X: [definition]"
           - Terms appearing in a definitions section

        2. IGNORE all of the following (these are NOT legal terms):
           - PDF metadata and formatting artifacts (like CIDSystemInfo, CMapName, obj, endobj)
           - Technical markers or identifiers (like EvoPdf_ekfbdmochonfkgfppddhjdikgnlnnfej)
           - Random alphanumeric strings
           - Page numbers or section markers

        3. For each genuine legal term you identify:
           - Extract the term itself
           - Provide its exact definition from the document
           - Note which section it appears in
           - Analyze whether the definition is clear or ambiguous
           - Note any inconsistencies in how the term is used elsewhere in the document

        4. If you cannot find any formally defined legal terms, state: "No formal legal definitions were found in this document."

        Format your response as a clean, professional legal terminology analysis that would be useful for a lawyer reviewing this document.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for improved legal definition extraction, model: {self.model_name}")
        result = self._call_openai_api(prompt)

        # Post-process the result to catch any remaining PDF artifacts
        result = self._post_process_legal_definitions(result)

        return result

    def _post_process_legal_definitions(self, text: str) -> str:
        """Post-process the legal definitions to remove any remaining PDF artifacts."""

        # Check if the response contains PDF metadata artifacts
        pdf_artifacts = [
            "CIDSystemInfo", "CMapName", "CMapType", "CIDToGIDMap",
            "obj", "R", "def", "endobj", "stream", "endstream"
        ]

        # Check for PDF artifacts in the definitions
        contains_artifacts = any(artifact in text for artifact in pdf_artifacts)

        if contains_artifacts:
            # If PDF artifacts are detected, replace the entire output with a clearer message
            return "No formal legal definitions were found in this document. The extracted text appears to contain technical metadata rather than legal content."

        return text

    def analyze_governing_law(self, text: str) -> str:
        """Analyze governing law and jurisdiction clauses."""
        print(f"Analyzing governing law clauses, text length: {len(text)} characters")

        # Limit text to max_length
        if len(text) > self.max_length:
            print(f"Text too long, truncating to {self.max_length} characters")
            text = text[:self.max_length]

        # Prepare prompt for governing law analysis
        prompt = f"""
        Please analyze the governing law and jurisdiction clauses in the following document:

        {text}

        Your analysis should include:
        1. Identify the specific governing law provision(s) and exact text
        2. Identify any jurisdiction or venue provisions and exact text
        3. Analyze potential implications of the chosen law/jurisdiction
        4. Identify any potential enforceability issues
        5. Note any unusual aspects of these provisions compared to standard practice
        6. Identify any potential conflicts with other provisions in the document

        Format your response as a detailed legal analysis of these provisions.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for governing law analysis, model: {self.model_name}")
        return self._call_openai_api(prompt)

    def check_legal_compliance(self, text: str, regulatory_areas: List[str] = None) -> str:
        """Check document for compliance with regulations in specified areas."""
        print(f"Checking legal compliance, text length: {len(text)} characters")

        if not regulatory_areas:
            regulatory_areas = ["data privacy", "consumer protection", "employment", "intellectual property"]

        # Limit text to max_length
        if len(text) > self.max_length:
            print(f"Text too long, truncating to {self.max_length} characters")
            text = text[:self.max_length]

        # Prepare prompt for compliance check
        prompt = f"""
        Please check the following document for legal compliance with regulations in these areas:
        {', '.join(regulatory_areas)}

        Document text:
        {text}

        For each regulatory area:
        1. Identify relevant provisions in the document related to this area
        2. Analyze whether these provisions appear to be compliant with current regulations
        3. Identify potential compliance gaps or issues
        4. Suggest improvements to enhance compliance
        5. Note any potential legal risks related to compliance in this area

        Format your response as a structured legal compliance assessment.
        """

        # Call OpenAI API
        print(f"Calling OpenAI API for legal compliance check, model: {self.model_name}")
        return self._call_openai_api(prompt)


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """Load YAML file."""
    print(f"Loading YAML file: {file_path}")
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        print(f"Successfully loaded YAML file: {file_path}")
        return data
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return {}


def process_documents():
    """Process all legal documents in the contracts folder."""
    print("\n========== STARTING LEGAL DOCUMENT PROCESSING ==========\n")

    # Set up tools
    print("Setting up ChromaDB storage...")
    chroma_storage = ChromaDBStorage(
        db_path="./chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    print("Setting up OpenAI document processor...")

    # Check for OpenAI API key in environment variables
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print(
            "WARNING: OPENAI_API_KEY environment variable not found. You'll need to set this for the OpenAI API to work.")

    # Get custom settings if available (in Streamlit context)
    api_model = os.environ.get('OPENAI_API_MODEL', "gpt-4o-latest")  # Default
    api_endpoint = os.environ.get('OPENAI_API_BASE')  # Default

    # Check if running in Streamlit context and get settings
    try:
        import streamlit as st
        if 'api_model' in st.session_state:
            api_model = st.session_state.api_model
            print(f"Using custom model from session state: {api_model}")

        if 'api_endpoint' in st.session_state:
            api_endpoint = st.session_state.api_endpoint
            print(f"Using custom API endpoint from session state: {api_endpoint}")
    except:
        # Not running in Streamlit context
        print("Not running in Streamlit context, using environment variables or defaults")

    document_processor = OpenAIDocumentProcessor(
        model_name=api_model,
        max_length=5000,
        api_key=api_key,
        api_base=api_endpoint
    )

    # Process all documents in the contracts folder
    contracts_folder = "contracts"
    print(f"Processing legal documents from: {os.path.abspath(contracts_folder)}")

    # Create contracts folder if it doesn't exist
    if not os.path.exists(contracts_folder):
        print(f"Contracts folder not found, creating: {contracts_folder}")
        os.makedirs(contracts_folder)
        print(f"Created {contracts_folder} directory. Please place your legal documents there and run again.")
        return

    print(f"Contracts folder exists: {os.path.exists(contracts_folder)}")

    # Get all files in the contracts folder (including subfolders)
    contract_files = []

    print("Scanning for legal document files (including subfolders)...")
    for root, dirs, files in os.walk(contracts_folder):
        print(f"Scanning directory: {root}")
        for filename in files:
            file_path = os.path.join(root, filename)
            # Check if file is of supported type
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.pdf', '.docx', '.txt']:
                print(f"Found supported legal file: {file_path}")
                contract_files.append(file_path)
            else:
                print(f"Skipping unsupported file: {file_path}")

    if not contract_files:
        print(f"No supported legal documents found in {contracts_folder} folder.")
        print("Supported formats: PDF, DOCX, TXT")
        return

    print(f"Found {len(contract_files)} legal document(s) to process:")
    for i, file_path in enumerate(contract_files):
        print(f"  {i + 1}. {file_path}")

    # Process each document
    all_results = {}
    for i, file_path in enumerate(contract_files):
        doc_id = f"doc_{i + 1:03d}"
        filename = os.path.basename(file_path)

        print(f"\nProcessing legal document {i + 1}/{len(contract_files)}: {filename}")

        try:
            # Process the document
            print(f"Extracting and processing legal document...")
            processed_doc = document_processor.process_document(file_path, doc_id)

            # Store chunks in ChromaDB
            print(f"Storing legal document chunks in ChromaDB...")
            storage_result = chroma_storage.add_texts(
                processed_doc["chunks"],
                processed_doc["metadatas"],
                processed_doc["ids"]
            )

            print(f"Legal document processing result: {storage_result}")

            # Generate summary and analysis
            print("Generating legal document summary...")
            summary = document_processor.summarize(processed_doc["text"])

            print("Generating legal document analysis...")
            analysis = document_processor.analyze(processed_doc["text"])

            # Store summary in ChromaDB
            print("Storing legal document summary in ChromaDB...")
            summary_result = chroma_storage.add_summary(
                doc_id=doc_id,
                source=file_path,
                filename=filename,
                summary=summary,
                analysis=analysis
            )

            print(f"Legal summary storage result: {summary_result}")

            # Store results
            results = {
                "filename": filename,
                "processing": storage_result,
                "summary": summary,
                "analysis": analysis,
                "summary_storage": summary_result
            }

            all_results[doc_id] = results

            # Save results to file
            results_folder = "results"
            if not os.path.exists(results_folder):
                print(f"Creating results folder: {results_folder}")
                os.makedirs(results_folder)

            summary_file = os.path.join(results_folder, f"{os.path.splitext(filename)[0]}_summary.txt")
            print(f"Saving legal summary and analysis to: {summary_file}")

            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Legal Document: {filename}\n")
                f.write(f"Processed as: {doc_id}\n\n")
                f.write(f"SUMMARY:\n{results['summary']}\n\n")
                f.write(f"ANALYSIS:\n{results['analysis']}")

            print(f"Successfully saved legal summary and analysis to {summary_file}")

        except Exception as e:
            print(f"Error processing legal document {filename}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue

    # Print summary of processing
    print(f"\n==== Legal Document Processing Summary ====")
    print(f"Successfully processed {len(all_results)} out of {len(contract_files)} legal documents.")
    print(f"Results saved to the 'results' folder.")
    print(f"Document chunks and summaries stored in ChromaDB.")

    # Removed the code that creates the comparison script
    if len(all_results) > 1:
        print("\nMultiple legal documents were processed. You can compare them using the web interface.")

    print("\n========== LEGAL DOCUMENT PROCESSING COMPLETED ==========\n")
    return all_results


def main():
    """Main function."""
    print("Legal Document Processing System")
    print("---------------------------------------------")
    print("This system will process all legal documents in the 'contracts' folder (including subfolders).")
    print(f"Current working directory: {os.getcwd()}")

    try:
        print("About to start legal document processing")
        process_documents()
        print("Legal document processing completed successfully.")
    except Exception as e:
        print(f"Error during legal document processing: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("Please check the error message and try again.")


if __name__ == "__main__":
    print("Legal document processing script is starting...")
    main()
    print("Legal document processing script has finished.")