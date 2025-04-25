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
import json
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import docx
import PyPDF2
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI


class ChromaDBStorage:
    """Storage for document chunks and summaries in ChromaDB."""

    def __init__(self, db_path: str = "/home/cdsw/02_applicaton/chromadb", chunks_collection: str = "document_chunks",
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

        # Create embedding function - USING ONLY LOCAL MODELS
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("Embedding function created (using SentenceTransformers)")

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


class CrewAIDocumentProcessor:
    """Document processor using CrewAI for legal document analysis."""

    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o",
                 agents_yaml_path: str = "agents.yaml",
                 tasks_yaml_path: str = "tasks.yaml"):
        """Initialize CrewAI document processor."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.agents_yaml_path = agents_yaml_path
        self.tasks_yaml_path = tasks_yaml_path

        # Load agents and tasks configurations
        self.agents_config = self._load_yaml_file(agents_yaml_path)
        self.tasks_config = self._load_yaml_file(tasks_yaml_path)

        # Initialize LLM
        self._initialize_llm()

        # Initialize CrewAI agents (lazy loading - will be created when needed)
        self.agents = {}
        self.tasks = {}

        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    def _load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        print(f"Loading YAML file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
            print(f"Successfully loaded YAML file: {file_path}")
            return data
        except Exception as e:
            print(f"Error loading YAML file {file_path}: {e}")
            return {}

    def _initialize_llm(self):
        """Initialize language model for CrewAI."""
        try:
            # Make sure we have an API key
            if not self.api_key:
                raise ValueError("OpenAI API key is required")

            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=0.2
            )
            print(f"Initialized LLM: {self.model} at {self.base_url}")
        except Exception as e:
            error_msg = f"Error initializing LLM: {e}"
            print(error_msg)
            self.llm = None
            raise ValueError(f"Failed to initialize OpenAI LLM: {error_msg}")

    def _create_agent(self, agent_type: str) -> Agent:
        """Create a CrewAI agent from configuration."""
        if agent_type not in self.agents_config:
            # Create a default agent if not found in config
            print(f"Agent type '{agent_type}' not found in configuration, using default")
            role = "Legal Document Processor"
            goal = "Process legal documents efficiently and extract key information"
            backstory = "An expert in legal document analysis with experience in processing legal documents"
        else:
            # Get agent config
            agent_config = self.agents_config[agent_type]
            role = agent_config.get("role", "Legal Document Assistant")
            goal = agent_config.get("goal", "Process legal documents efficiently")
            backstory = agent_config.get("backstory", "An expert in legal document analysis")

        # Create agent
        try:
            agent = Agent(
                role=role,
                goal=goal,
                backstory=backstory,
                llm=self.llm,
                verbose=True,
                allow_delegation=True
            )
            return agent
        except Exception as e:
            print(f"Error creating agent: {e}")
            raise ValueError(f"Failed to create agent {agent_type}: {e}")

    def _create_task(self, task_type: str, context: Dict[str, Any] = None, **kwargs) -> Task:
        """Create a CrewAI task from configuration with context."""
        if task_type not in self.tasks_config:
            raise ValueError(f"Task type '{task_type}' not found in configuration")

        task_config = self.tasks_config[task_type]

        # Get agent type
        agent_type = task_config.get("agent", "document_processor")

        # Get or create agent
        if agent_type not in self.agents:
            self.agents[agent_type] = self._create_agent(agent_type)

        # Format the description with kwargs values
        description = task_config.get("description", "")
        try:
            # Format with kwargs, handling missing keys gracefully
            for key, value in kwargs.items():
                if f"{{{key}}}" in description:
                    description = description.replace(f"{{{key}}}", str(value))
        except Exception as e:
            print(f"Warning: Error formatting task description: {e}")

        # Create task - IMPORTANT: only add context if it's compatible with the task
        # Some newer versions of CrewAI may expect context to be a list, not a dict
        try:
            # First try without context
            task = Task(
                description=description,
                expected_output=task_config.get("expected_output", ""),
                agent=self.agents[agent_type]
            )
            return task
        except Exception as e:
            print(f"Warning: Could not create task without context: {e}")
            try:
                # If that fails, try with an empty list as context (if context expected)
                task = Task(
                    description=description,
                    expected_output=task_config.get("expected_output", ""),
                    agent=self.agents[agent_type],
                    context=[]
                )
                return task
            except Exception as e2:
                print(f"Error creating task with empty list context: {e2}")
                raise ValueError(f"Failed to create task {task_type}: {e2}")

    def extract_text(self, file_path: str) -> str:
        """Extract text from various document formats."""
        print(f"Extracting text from document: {file_path}")
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
        """Extract text from PDF files with improved reliability."""
        print(f"Extracting text from PDF: {file_path}")

        # Try multiple extraction methods until we get usable text
        text = ""

        # Method 1: PyPDF2
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                print(f"PyPDF2 extracted {len(text)} characters from PDF")

                # If we got a reasonable amount of text, return it
                if len(text) > 100:
                    return text
        except Exception as e:
            print(f"Error extracting text with PyPDF2: {e}")

        # If PyPDF2 didn't work well, try another method or external tool
        # For example, you could call an external tool like pdftotext (from poppler-utils)
        # or a Python library like pdf2text if available

        try:
            # Check if pdftotext is available (from poppler-utils)
            import subprocess
            import tempfile

            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
                temp_output = temp_file.name

            # Call pdftotext
            try:
                subprocess.run(['pdftotext', '-layout', file_path, temp_output],
                               check=True, capture_output=True)

                # Read the output
                with open(temp_output, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                print(f"pdftotext extracted {len(text)} characters from PDF")

                # Clean up
                os.unlink(temp_output)

                # If we got a reasonable amount of text, return it
                if len(text) > 100:
                    return text
            except subprocess.CalledProcessError:
                print("pdftotext command failed or is not installed")
            except FileNotFoundError:
                print("pdftotext command not found")
        except Exception as e:
            print(f"Error using alternative PDF extraction: {e}")

        # As a last resort, try raw binary reading with different encodings
        try:
            with open(file_path, 'rb') as file:
                binary_data = file.read()

                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        text = binary_data.decode(encoding, errors='ignore')
                        print(f"Raw binary with {encoding} extracted {len(text)} characters")

                        # If we got text that looks reasonable, return it
                        if len(text) > 100:
                            return text
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error with raw binary reading: {e}")

        # If we got here, we couldn't extract good text
        # Return whatever we have, even if it's not great
        if not text:
            text = f"Failed to extract text from PDF: {file_path}"
            print(text)

        return text

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

    def process_document(self, file_path: str, doc_id: str) -> Dict[str, Any]:
        """Process a document using CrewAI with improved PDF handling."""
        print(f"Processing document: {file_path}, ID: {doc_id}")

        # Check if file exists and has content
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")

        # Extract text with improved PDF handling
        text = self.extract_text(file_path)

        # Verify we have text
        if not text or len(text) < 10:
            print(f"Warning: Very little text extracted from {file_path}. File may be corrupted or unreadable.")
            text = f"[Unable to extract meaningful text from: {file_path}]"

        # Log information about the extracted text
        print(f"Extracted {len(text)} characters from document")
        print(f"Text preview: {text[:200]}...")

        # Clean the text of potential PDF artifacts
        text = self._clean_pdf_artifacts(text)
        print(f"After cleaning, text length: {len(text)} characters")

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks")

        # Prepare metadata
        metadatas = [{"source": file_path, "doc_id": doc_id, "chunk": i} for i in range(len(chunks))]

        # Prepare IDs
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        # Create task for document processing using the tasks.yaml
        try:
            # Create a simpler task description directly
            description = f"Process the document with ID {doc_id} and prepare it for analysis."

            if "document_processor" not in self.agents:
                self.agents["document_processor"] = self._create_agent("document_processor")

            process_task = Task(
                description=description,
                expected_output="Processed document metadata",
                agent=self.agents["document_processor"]
            )

            # Create crew for processing
            crew = Crew(
                agents=[self.agents["document_processor"]],
                tasks=[process_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            crew_result = crew.kickoff()
            print(f"CrewAI processing result: {crew_result}")
        except Exception as e:
            print(f"Warning: Error in CrewAI processing: {e}")
            import traceback
            traceback.print_exc()
            # Continue with basic processing even if CrewAI fails
            crew_result = f"Document processed without CrewAI: {e}"

        return {
            "text": text,
            "chunks": chunks,
            "metadatas": metadatas,
            "ids": ids,
            "doc_id": doc_id,
            "crew_result": crew_result
        }

    def _clean_pdf_artifacts(self, text: str) -> str:
        """Clean PDF artifacts from extracted text."""
        # Remove PDF object references
        text = re.sub(r'\d+ \d+ obj.*?endobj', ' ', text, flags=re.DOTALL)

        # Remove PDF streams
        text = re.sub(r'stream\s.*?endstream', ' ', text, flags=re.DOTALL)

        # Remove PDF dictionary objects
        text = re.sub(r'<<.*?>>', ' ', text, flags=re.DOTALL)

        # Remove PDF operators and commands
        text = re.sub(r'/[A-Za-z0-9_]+\s+', ' ', text)

        # Remove PDF metadata like CID, CMap entries
        text = re.sub(r'/(CID|CMap|Registry|Ordering|Supplement|CIDToGIDMap).*?def', ' ', text)

        # Remove "R" references
        text = re.sub(r'\d+ \d+ R', ' ', text)

        # Remove common PDF artifacts
        text = re.sub(r'EvoPdf_[a-zA-Z0-9_]+', '', text)

        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)

        return text

    def summarize(self, text: str, max_length: int = 2500) -> str:
        """Summarize text using CrewAI."""
        print(f"Summarizing text, length: {len(text)} characters")

        # Limit text to avoid token limits
        truncated_text = text[:max_length * 2] if len(text) > max_length * 2 else text

        try:
            # Load task configuration
            if "summarize_document" not in self.tasks_config:
                raise ValueError("Task 'summarize_document' not found in configuration")

            task_config = self.tasks_config["summarize_document"]

            # Get agent type
            agent_type = task_config.get("agent", "document_summarizer")

            # Get or create agent
            if agent_type not in self.agents:
                self.agents[agent_type] = self._create_agent(agent_type)

            # Format the description - directly insert doc_text to avoid context issues
            description = task_config.get("description", "")
            description = description.replace("{doc_text}", truncated_text)
            if "{max_length}" in description:
                description = description.replace("{max_length}", str(max_length))

            # Create task without context (which appears to be the issue)
            summarize_task = Task(
                description=description,
                expected_output=task_config.get("expected_output", ""),
                agent=self.agents[agent_type]
                # No context parameter!
            )

            # Create crew for summarization
            crew = Crew(
                agents=[self.agents[agent_type]],
                tasks=[summarize_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()

            # Convert CrewOutput to string
            summary_text = str(result) if hasattr(result, '__str__') else "Error: Unable to convert result to string"

            return summary_text
        except Exception as e:
            print(f"Error in summarize method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to a simple summary if CrewAI fails
            return f"Error generating summary: {str(e)}"

    def analyze(self, text: str, analysis_depth: str = "detailed") -> str:
        """Analyze text using CrewAI."""
        print(f"Analyzing text, length: {len(text)} characters, depth: {analysis_depth}")

        # Limit text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text

        try:
            # Load task configuration
            if "analyze_document" not in self.tasks_config:
                raise ValueError("Task 'analyze_document' not found in configuration")

            task_config = self.tasks_config["analyze_document"]

            # Get agent type
            agent_type = task_config.get("agent", "legal_analyzer")

            # Get or create agent
            if agent_type not in self.agents:
                self.agents[agent_type] = self._create_agent(agent_type)

            # Format the description directly - avoid using context
            description = task_config.get("description", "")
            description = description.replace("{doc_text}", truncated_text)
            description = description.replace("{analysis_depth}", analysis_depth)
            if "{doc_id}" in description:
                description = description.replace("{doc_id}", "current")

            # Create task without context parameter
            analyze_task = Task(
                description=description,
                expected_output=task_config.get("expected_output", ""),
                agent=self.agents[agent_type]
                # No context parameter!
            )

            # Create crew for analysis
            crew = Crew(
                agents=[self.agents[agent_type]],
                tasks=[analyze_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()

            # Convert CrewOutput to string
            analysis_text = str(result) if hasattr(result, '__str__') else "Error: Unable to convert result to string"

            return analysis_text
        except Exception as e:
            print(f"Error in analyze method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to a simple analysis if CrewAI fails
            return f"Error generating analysis: {str(e)}"

    def compare_with_summaries(self, new_text: str, summaries: List[Dict[str, Any]],
                               focus_areas: List[str] = None) -> str:
        """Compare a new document with existing summaries using CrewAI."""
        print(f"Comparing new document with {len(summaries)} existing summaries")

        if not focus_areas:
            focus_areas = ["legal provisions", "obligations", "rights", "liability", "termination", "jurisdiction"]

        # Limit text to avoid token limits
        truncated_new_text = new_text[:3000] if len(new_text) > 3000 else new_text

        try:
            # Prepare summaries text
            summaries_text = "\n\n".join(
                [f"DOCUMENT {i + 1} (ID: {summary.get('doc_id', f'doc_{i + 1}')}): {summary.get('text', '')}"
                 for i, summary in enumerate(summaries)])

            # Limit summaries text length
            truncated_summaries_text = summaries_text[:5000] if len(summaries_text) > 5000 else summaries_text

            # Create description directly without using task configuration
            description = "Compare the following legal documents, focusing on: " + ", ".join(focus_areas) + ". "
            description += "Identify key similarities and differences in legal terms, obligations, and rights. "
            description += "Provide a well-structured analysis that highlights important legal distinctions. "
            description += f"Document IDs: {','.join([summary.get('doc_id', f'doc_{i}') for i, summary in enumerate(summaries)])}\n\n"
            description += f"NEW DOCUMENT TEXT:\n{truncated_new_text}\n\n"
            description += f"EXISTING DOCUMENT SUMMARIES:\n{truncated_summaries_text}"

            # Create agent and task directly
            if "document_comparer" not in self.agents:
                self.agents["document_comparer"] = self._create_agent("document_comparer")

            compare_task = Task(
                description=description,
                expected_output="A comprehensive legal comparison highlighting key similarities and differences",
                agent=self.agents["document_comparer"]
            )

            # Create crew for comparison
            crew = Crew(
                agents=[self.agents["document_comparer"]],
                tasks=[compare_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()

            # Convert CrewOutput to string
            comparison_text = str(result) if hasattr(result, '__str__') else "Error: Unable to convert result to string"

            # Format the output if it looks like JSON
            if comparison_text.strip().startswith('{') or comparison_text.strip().startswith('['):
                try:
                    import json
                    data = json.loads(comparison_text)

                    # Convert JSON to a more readable format
                    formatted_text = "# Document Comparison Results\n\n"

                    # Format according to expected structure
                    if isinstance(data, dict):
                        if "similarities" in data:
                            formatted_text += "## Similarities\n\n"
                            for item in data["similarities"]:
                                formatted_text += f"- {item}\n"
                            formatted_text += "\n"

                        if "differences" in data:
                            formatted_text += "## Differences\n\n"
                            for item in data["differences"]:
                                formatted_text += f"- {item}\n"
                            formatted_text += "\n"

                        if "analysis" in data:
                            formatted_text += "## Analysis\n\n"
                            formatted_text += data["analysis"]
                            formatted_text += "\n"

                        # Add any other sections
                        for key, value in data.items():
                            if key not in ["similarities", "differences", "analysis"]:
                                formatted_text += f"## {key.replace('_', ' ').title()}\n\n"
                                formatted_text += f"{value}\n\n"

                    return formatted_text
                except:
                    # If we can't parse as JSON, return the raw string
                    pass

            return comparison_text
        except Exception as e:
            print(f"Error in compare_with_summaries method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback if CrewAI fails
            return f"Error comparing documents: {str(e)}"

    def extract_legal_definitions(self, text: str) -> str:
        """Extract and analyze legal definitions from a document."""
        print(f"Extracting legal definitions, text length: {len(text)} characters")

        # Limit text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text

        try:
            # Create description directly without using task configuration
            description = "Extract and explain all defined legal terms from the following document text. "
            description += "Identify inconsistencies in definitions and potential legal ambiguities. "
            description += f"Document text: {truncated_text}"

            # Create agent and task directly
            if "legal_terminology_extractor" not in self.agents:
                self.agents["legal_terminology_extractor"] = self._create_agent("legal_terminology_extractor")

            extract_task = Task(
                description=description,
                expected_output="Glossary of legal terms with explanations and identified issues",
                agent=self.agents["legal_terminology_extractor"]
            )

            # Create crew for extraction
            crew = Crew(
                agents=[self.agents["legal_terminology_extractor"]],
                tasks=[extract_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()

            # Convert CrewOutput to string
            definitions_text = str(result) if hasattr(result,
                                                      '__str__') else "Error: Unable to convert result to string"

            return definitions_text
        except Exception as e:
            print(f"Error in extract_legal_definitions method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback if CrewAI fails
            return f"Error extracting legal definitions: {str(e)}"

    def extract_legal_definitions_improved(self, text: str) -> str:
        """Extract and analyze legal definitions from a document with improved accuracy."""
        # This is just a wrapper around extract_legal_definitions to match the API expected by the app
        return self.extract_legal_definitions(text)

    def assess_legal_risks(self, text: str, risk_categories: List[str] = None) -> str:
        """Assess legal risks in a document."""
        print(f"Assessing legal risks, text length: {len(text)} characters")

        if not risk_categories:
            risk_categories = ["contractual", "regulatory", "litigation", "intellectual property"]

        # Limit text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text

        try:
            # Create description directly
            description = "Assess the legal risks in the following document text, categorizing them by "
            description += ", ".join(risk_categories) + ". "
            description += "Provide a risk rating (High/Medium/Low) and potential mitigation strategies for each risk. "
            description += f"Document text: {truncated_text}"

            # Create agent and task directly
            if "legal_risk_assessor" not in self.agents:
                self.agents["legal_risk_assessor"] = self._create_agent("legal_risk_assessor")

            risk_task = Task(
                description=description,
                expected_output="Legal risk assessment report with risk ratings and mitigation recommendations",
                agent=self.agents["legal_risk_assessor"]
            )

            # Create crew for risk assessment
            crew = Crew(
                agents=[self.agents["legal_risk_assessor"]],
                tasks=[risk_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()

            # Convert CrewOutput to string
            risk_text = str(result) if hasattr(result, '__str__') else "Error: Unable to convert result to string"

            return risk_text
        except Exception as e:
            print(f"Error in assess_legal_risks method: {e}")
            import traceback
            traceback.print_exc()
            # Fallback if CrewAI fails
            return f"Error assessing legal risks: {str(e)}"

    def check_legal_compliance(self, text: str, regulatory_areas: List[str] = None) -> str:
        """Check document for compliance with regulations."""
        print(f"Checking legal compliance, text length: {len(text)} characters")

        if not regulatory_areas:
            regulatory_areas = ["data privacy", "consumer protection", "employment", "intellectual property"]

        # Limit text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text

        try:
            # Create task for compliance check using the tasks.yaml
            compliance_task = self._create_task(
                "check_legal_compliance",
                context={"regulatory_areas": regulatory_areas},
                doc_text=truncated_text
            )

            # Create crew for compliance check
            crew = Crew(
                agents=[self.agents.get("legal_compliance_checker", self._create_agent("legal_compliance_checker"))],
                tasks=[compliance_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"Error in check_legal_compliance method: {e}")
            # Fallback if CrewAI fails
            return f"Error checking legal compliance: {str(e)}"

    def analyze_governing_law(self, text: str) -> str:
        """Analyze governing law and jurisdiction clauses."""
        print(f"Analyzing governing law, text length: {len(text)} characters")

        # Limit text to avoid token limits
        truncated_text = text[:5000] if len(text) > 5000 else text

        try:
            # Create task for governing law analysis using the tasks.yaml
            law_task = self._create_task(
                "identify_governing_law",
                doc_text=truncated_text
            )

            # Create crew for governing law analysis
            crew = Crew(
                agents=[self.agents.get("legal_analyzer", self._create_agent("legal_analyzer"))],
                tasks=[law_task],
                verbose=True,
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"Error in analyze_governing_law method: {e}")
            # Fallback if CrewAI fails
            return f"Error analyzing governing law: {str(e)}"


def get_document_processor():
    """Get the OpenAI document processor based on settings."""
    # Check for OpenAI configuration
    config_path = "config/settings.json"
    api_key = None

    # First check environment variable
    if os.environ.get("OPENAI_API_KEY"):
        api_key = os.environ.get("OPENAI_API_KEY")
        print("Using API key from environment variable")

    # Then check settings file
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                settings = json.load(f)

                # If we didn't get a key from environment, try settings
                if not api_key and settings.get("openai_api_key"):
                    api_key = settings.get("openai_api_key")
                    print("Using API key from settings file")

                # Get endpoint and model
                endpoint = settings.get("openai_endpoint", "https://api.openai.com/v1")
                model = settings.get("openai_model", "gpt-4o")
        else:
            # Default values if no settings file
            endpoint = "https://api.openai.com/v1"
            model = "gpt-4o"
            print("No settings file found, using defaults")

        # Final check for API key
        if not api_key:
            # One more attempt to get from session state if code is running in streamlit
            try:
                import streamlit as st
                if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
                    api_key = st.session_state.openai_api_key
                    print("Using API key from session state")
            except:
                pass

        # Check if we have an API key
        if not api_key:
            raise ValueError("No OpenAI API key found in settings, environment variables, or session state.")

        # Create the processor
        print(f"Creating CrewAI processor with endpoint: {endpoint}, model: {model}")
        return CrewAIDocumentProcessor(
            api_key=api_key,
            base_url=endpoint,
            model=model
        )
    except Exception as e:
        print(f"Error setting up OpenAI document processor: {e}")
        raise ValueError(
            f"Unable to initialize OpenAI document processor: {e}. Please configure an OpenAI API key in the Settings tab.")


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

    # Set up storage
    print("Setting up ChromaDB storage...")
    chroma_storage = ChromaDBStorage(
        db_path="./chromadb",
        chunks_collection="document_chunks",
        summaries_collection="document_summaries"
    )

    # Get the appropriate document processor based on settings
    document_processor = get_document_processor()
    processor_type = document_processor.__class__.__name__
    print(f"Using document processor: {processor_type}")

    # Process all documents in the contracts folder
    contracts_folder = "/home/cdsw/02_application/contracts"
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