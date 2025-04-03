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
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
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

import streamlit as st
import os
import yaml
import tempfile
from pathlib import Path
import hashlib
from datetime import datetime
import json
import shutil
import pandas as pd

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Set page configuration
st.set_page_config(
    page_title="Contract Analysis Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'contracts_dir' not in st.session_state:
    st.session_state.contracts_dir = Path("./contracts")
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'llm_config' not in st.session_state:
    st.session_state.llm_config = {}

# Create contracts directory if it doesn't exist
if not os.path.exists(st.session_state.contracts_dir):
    os.makedirs(st.session_state.contracts_dir)

# Create configs directory if it doesn't exist
configs_dir = Path("./configs")
if not configs_dir.exists():
    os.makedirs(configs_dir)


# Function to delete contracts
def delete_contract(contract_path):
    """
    Delete a contract file from the contracts directory.

    Args:
        contract_path: Relative path to the contract to delete

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        # Construct the absolute path
        full_path = os.path.join(st.session_state.contracts_dir, contract_path)

        # Check if the file exists
        if os.path.exists(full_path):
            # Delete the file
            os.remove(full_path)

            # Update the uploaded_files list to remove the deleted file
            for i, file_info in enumerate(st.session_state.uploaded_files.copy()):
                if str(file_info["path"]) == full_path or os.path.basename(str(file_info["path"])) == os.path.basename(
                        contract_path):
                    st.session_state.uploaded_files.pop(i)
                    break

            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error deleting file: {str(e)}")
        return False


# File handling tools
class FileReadTool(BaseTool):
    name: str = "ReadContractFile"
    description: str = "Read the content of a contract file"

    def _run(self, file_path: str) -> str:
        """
        Read the content of a file, with improved error handling and path resolution.

        Args:
            file_path: Path to the file to read (can be absolute or relative)

        Returns:
            Content of the file or error message
        """
        try:
            # First try the path as provided
            original_path = Path(file_path)

            # If not found and not absolute, try to find in contracts directory
            if not original_path.exists() and not os.path.isabs(file_path):
                contracts_dir = st.session_state.contracts_dir
                possible_paths = []

                # Search the entire contracts directory for files with similar names
                for root, dirs, files in os.walk(contracts_dir):
                    for file in files:
                        # Check if the filename contains the key part of the provided path
                        # This helps with partial matches
                        if Path(file_path).name.lower() in file.lower():
                            possible_paths.append(os.path.join(root, file))

                # If we found possible matches, use the first one
                if possible_paths:
                    file_path = possible_paths[0]
                    print(f"Found alternative path: {file_path}")
                else:
                    # Try one more search approach - by keywords in filename
                    keywords = Path(file_path).stem.split('_')
                    if keywords:
                        for root, dirs, files in os.walk(contracts_dir):
                            for file in files:
                                # Check if any significant keyword is in the filename
                                if any(keyword.lower() in file.lower() for keyword in keywords if len(keyword) > 3):
                                    possible_paths.append(os.path.join(root, file))

                        if possible_paths:
                            file_path = possible_paths[0]
                            print(f"Found keyword match: {file_path}")

            # Final check with the resolved path
            resolved_path = Path(file_path)
            if not resolved_path.exists():
                return f"File not found: {original_path}. Tried searching in contracts directory but couldn't find a match."

            # Read the file based on its extension
            extension = resolved_path.suffix.lower()

            if extension == ".pdf":
                import PyPDF2
                with open(resolved_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        text += pdf_reader.pages[page_num].extract_text() + "\n"
                return text

            elif extension == ".docx":
                import docx
                doc = docx.Document(resolved_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text

            elif extension == ".txt":
                with open(resolved_path, "r", encoding="utf-8") as file:
                    return file.read()

            else:
                return f"Unsupported file extension: {extension}"

        except Exception as e:
            return f"Error reading file: {str(e)}"


class FileWriteTool(BaseTool):
    name: str = "WriteContractFile"
    description: str = "Write content to a file in the contracts directory"

    def _run(self, filename: str, content: str) -> str:
        try:
            file_path = Path("./contracts") / filename

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)

            return f"Successfully wrote content to {filename}"

        except Exception as e:
            return f"Error writing file: {str(e)}"


class ContractComparisonTool(BaseTool):
    name: str = "CompareContracts"
    description: str = "Compare multiple contracts for similarities or conflicts"

    def _run(self, contract_paths: list, comparison_type: str) -> str:
        try:
            file_read_tool = FileReadTool()
            contract_contents = {}

            for path in contract_paths:
                contract_name = os.path.basename(path)
                contract_contents[contract_name] = file_read_tool._run(path)

            comparison_result = f"Comparison of {len(contract_paths)} contracts for {comparison_type}:\n\n"

            # The actual comparison will be done by the agents
            # This tool just prepares the data structure

            return json.dumps({
                "comparison_type": comparison_type,
                "contracts": contract_contents
            })

        except Exception as e:
            return f"Error comparing contracts: {str(e)}"


# Load YAML configurations
def load_agents_config():
    agents_path = configs_dir / "agents.yaml"
    if agents_path.exists():
        with open(agents_path, 'r') as file:
            return yaml.safe_load(file)
    return {}


def load_tasks_config():
    tasks_path = configs_dir / "tasks.yaml"
    if tasks_path.exists():
        with open(tasks_path, 'r') as file:
            return yaml.safe_load(file)
    return {}


# Save configurations to YAML files
def save_agents_config(config):
    agents_path = configs_dir / "agents.yaml"
    with open(agents_path, 'w') as file:
        yaml.dump(config, file)


def save_tasks_config(config):
    tasks_path = configs_dir / "tasks.yaml"
    with open(tasks_path, 'w') as file:
        yaml.dump(config, file)


def save_llm_config(config):
    llm_path = configs_dir / "llm_config.yaml"
    with open(llm_path, 'w') as file:
        yaml.dump(config, file)
    st.session_state.llm_config = config


# Load LLM configuration
def load_llm_config():
    llm_path = configs_dir / "llm_config.yaml"
    if llm_path.exists():
        with open(llm_path, 'r') as file:
            config = yaml.safe_load(file)
            st.session_state.llm_config = config
            return config
    return {}


# Create agents from configuration
def create_agent_from_config(agent_config):
    # Initialize tools
    tools = [FileReadTool(), FileWriteTool(), ContractComparisonTool()]

    # Configure LLM settings
    llm_config = {
        "model": st.session_state.llm_config.get("model", "gpt-4"),
        "temperature": st.session_state.llm_config.get("temperature", 0.7),
        "max_tokens": st.session_state.llm_config.get("max_tokens", 500)
    }

    # Add API base URL if specified
    if 'api_endpoint' in st.session_state.llm_config and st.session_state.llm_config['api_endpoint']:
        llm_config["api_base"] = st.session_state.llm_config['api_endpoint']

    # Create agent with CrewAI
    agent = Agent(
        role=agent_config.get("role", "Contract Analyst"),
        goal=agent_config.get("goal", "Analyze contracts thoroughly"),
        backstory=agent_config.get("backstory", "Expert contract analyst with years of experience"),
        verbose=True,
        allow_delegation=agent_config.get("allow_delegation", False),
        tools=tools,
        llm_config=llm_config
    )

    return agent


# Create tasks from configuration
def create_task_from_config(task_config, agents, context=None):
    # Get the appropriate agent
    agent_name = task_config.get("agent", "default_agent")
    agent = next((a for a in agents if a.role == agent_name), agents[0])

    # Create task
    task = Task(
        description=task_config.get("description", "Analyze the contract"),
        expected_output=task_config.get("expected_output", "A detailed analysis of the contract"),
        agent=agent,
        async_execution=task_config.get("async_execution", False),
        context=context
    )

    return task


# Get all contract files, including in subfolders
def get_all_contracts():
    contracts = []
    contracts_dir = st.session_state.contracts_dir

    # Walk through all subdirectories recursively
    for root, dirs, files in os.walk(contracts_dir):
        for file in files:
            # Filter for supported file types
            if file.lower().endswith(('.pdf', '.docx', '.txt')):
                # Get the relative path for display
                rel_path = os.path.relpath(os.path.join(root, file), contracts_dir)
                contracts.append(rel_path)

    return sorted(contracts)


def truncate_contract_content(content, max_chars=50000):
    """
    Truncate contract content to avoid context window issues.
    Keeps the beginning and end of the document which often contain the most important information.

    Args:
        content: The full contract text
        max_chars: Maximum characters to keep

    Returns:
        Truncated content with note about truncation
    """
    if len(content) <= max_chars:
        return content

    # Keep the first 60% and last 40% of the allowed length
    first_part_len = int(max_chars * 0.6)
    last_part_len = max_chars - first_part_len

    first_part = content[:first_part_len]
    last_part = content[-last_part_len:]

    truncated = first_part + "\n\n[...CONTENT TRUNCATED TO AVOID CONTEXT WINDOW LIMITS...]\n\n" + last_part

    return truncated


def analyze_contracts_against_existing(uploaded_paths, existing_paths, analysis_type="conflicts"):
    """
    Analyze newly uploaded contracts against existing ones.

    Args:
        uploaded_paths: List of paths to newly uploaded contracts
        existing_paths: List of paths to existing contracts to compare against
        analysis_type: Type of analysis to perform (conflicts, similarities, clauses)

    Returns:
        Analysis results as text
    """
    # Make sure we have LLM config
    if not st.session_state.llm_config or not st.session_state.llm_config.get('api_key'):
        return "LLM not configured. Please set up the API key in the LLM Configuration tab."

    # Load agent configurations
    agents_config = load_agents_config()

    # Create file reading tool
    file_read_tool = FileReadTool()

    # Read the contract contents and truncate if necessary
    uploaded_contents = {}
    existing_contents = {}

    # Read uploaded contracts
    for path in uploaded_paths:
        content = file_read_tool._run(path)
        if not content.startswith("File not found:") and not content.startswith("Error reading file:"):
            truncated_content = truncate_contract_content(content)
            uploaded_contents[os.path.basename(path)] = truncated_content

    # Read existing contracts
    for path in existing_paths:
        content = file_read_tool._run(path)
        if not content.startswith("File not found:") and not content.startswith("Error reading file:"):
            truncated_content = truncate_contract_content(content)
            existing_contents[os.path.basename(path)] = truncated_content

    # Check if we have valid contracts to compare
    if not uploaded_contents:
        return "No valid uploaded contracts found. Please check the file paths."

    if not existing_contents:
        return "No valid existing contracts found. Please check the file paths."

    # Create agents
    contract_analyst = create_agent_from_config(agents_config.get("contract_analyst", {
        "role": "Contract Analyst",
        "goal": "Analyze contracts thoroughly and compare new contracts with existing ones",
        "backstory": "Expert contract analyst with years of experience in legal document analysis"
    }))

    legal_expert = create_agent_from_config(agents_config.get("legal_expert", {
        "role": "Legal Expert",
        "goal": "Identify legal implications and potential issues in contracts",
        "backstory": "Senior legal consultant with expertise in contract law and document analysis"
    }))

    # Prepare task details
    # Display file names (not full paths) for better readability
    uploaded_names = list(uploaded_contents.keys())
    existing_names = list(existing_contents.keys())

    # Instead of including full contract text in the prompt, provide a summary and key excerpts
    # The full text is available via FileReadTool if needed

    if analysis_type == "conflicts":
        description = f"""
        Analyze the following NEWLY UPLOADED contracts: {', '.join(uploaded_names)}

        Compare them against these EXISTING contracts: {', '.join(existing_names)}

        Identify any potential conflicts of interest, contradictions, or legal issues that could
        arise from having these contracts together. Focus on:

        1. Conflicting obligations, timelines, or requirements
        2. Exclusivity clauses or non-compete agreements that might conflict
        3. Confidentiality provisions that might create conflicts
        4. Any legal risks or issues that might arise from these contracts coexisting

        Use the ReadContractFile tool to access the contract content as needed. Focus on the most important
        clauses and provisions rather than analyzing every detail, as some contracts may be lengthy.

        Provide a detailed analysis with specific references to the relevant clauses.
        """
        expected_output = "A comprehensive analysis of potential conflicts between new and existing contracts"

    elif analysis_type == "similarities":
        description = f"""
        Analyze the following NEWLY UPLOADED contracts: {', '.join(uploaded_names)}

        Compare them against these EXISTING contracts: {', '.join(existing_names)}

        Identify key similarities, patterns, and common elements between these contracts. Focus on:

        1. Similar clauses, terms, or provisions
        2. Common legal structures or frameworks
        3. Matching obligations, requirements, or standards
        4. Implications of these similarities

        Use the ReadContractFile tool to access the contract content as needed. Focus on the most important
        clauses and provisions rather than analyzing every detail, as some contracts may be lengthy.

        Provide a detailed analysis with specific references to the relevant clauses.
        """
        expected_output = "A comprehensive analysis of similarities between new and existing contracts"

    else:  # clauses
        description = f"""
        Analyze the following NEWLY UPLOADED contracts: {', '.join(uploaded_names)}

        Compare them against these EXISTING contracts: {', '.join(existing_names)}

        Perform a detailed clause-by-clause comparison focusing on:

        1. Key differences in important clauses (liability, confidentiality, termination, etc.)
        2. Stronger or weaker protections in different versions of similar clauses
        3. Unusual or unique clauses that appear in some contracts but not others
        4. Recommendations for standardization or improvements

        Use the ReadContractFile tool to access the contract content as needed. Focus on the most important
        clauses and provisions rather than analyzing every detail, as some contracts may be lengthy.

        Provide a structured analysis organized by clause type.
        """
        expected_output = "A detailed clause-by-clause comparison between new and existing contracts"

    # Create task
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=legal_expert
    )

    # Create crew and run
    crew = Crew(
        agents=[contract_analyst, legal_expert],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()
    return result

# Compare contracts using CrewAI
def compare_contracts(contract_paths, comparison_type):
    """
    Compare multiple contracts using CrewAI with improved file handling.

    Args:
        contract_paths: List of paths to contracts to compare
        comparison_type: Type of comparison to perform

    Returns:
        Comparison results as text
    """
    # Make sure we have LLM config
    if not st.session_state.llm_config or not st.session_state.llm_config.get('api_key'):
        return "LLM not configured. Please set up the API key in the LLM Configuration tab."

    # Load agent configurations
    agents_config = load_agents_config()

    # Create improved file reading tool
    file_read_tool = FileReadTool()

    # Verify all files exist and can be read
    contract_contents = {}
    missing_files = []

    for path in contract_paths:
        # Try to read the file
        content = file_read_tool._run(path)

        # Check if the result indicates an error
        if content.startswith("File not found:") or content.startswith("Error reading file:"):
            missing_files.append((path, content))
        else:
            # Store the content if successful - truncate to avoid context window issues
            file_name = os.path.basename(str(path))
            truncated_content = truncate_contract_content(content)
            contract_contents[file_name] = truncated_content

    # If any files are missing, return an error
    if missing_files:
        error_message = "The following files could not be read:\n\n"
        for path, error in missing_files:
            error_message += f"- {path}: {error}\n"

        if not contract_contents:
            return error_message + "\nNo contracts could be read. Please check the file paths and try again."
        else:
            # Continue with available files but warn about missing ones
            error_message += f"\nProceeding with {len(contract_contents)} available contracts."
            st.warning(error_message)

    # Create agents
    contract_analyst = create_agent_from_config(agents_config.get("contract_analyst", {
        "role": "Contract Analyst",
        "goal": "Analyze contracts thoroughly to find important information",
        "backstory": "Expert contract analyst with years of experience in legal document analysis"
    }))

    legal_expert = create_agent_from_config(agents_config.get("legal_expert", {
        "role": "Legal Expert",
        "goal": "Identify legal implications and potential issues in contracts",
        "backstory": "Senior legal consultant with expertise in contract law and document analysis"
    }))

    # If we have at least two contracts, proceed with the comparison
    if len(contract_contents) >= 2:
        # Create task based on comparison type with file info
        if comparison_type == "Similarities":
            task_description = f"""
            Compare the following contracts and identify key similarities between them: {', '.join(contract_contents.keys())}

            Use the ReadContractFile tool to access the contract content as needed. Focus on the most important
            clauses and provisions rather than analyzing every detail, as some contracts may be lengthy.

            Focus on identifying:
            1. Similar or identical clauses and provisions
            2. Common legal structures or frameworks
            3. Matching terms, conditions, and requirements
            4. Consistent patterns in language or formatting
            5. Shared parties, entities, or references
            """
        elif comparison_type == "Conflicts of Interest":
            task_description = f"""
            Analyze the following contracts and identify potential conflicts of interest: {', '.join(contract_contents.keys())}

            Use the ReadContractFile tool to access the contract content as needed. Focus on the most important
            clauses and provisions rather than analyzing every detail, as some contracts may be lengthy.

            Focus on identifying:
            1. Conflicting terms, obligations, or requirements
            2. Potential conflicts of interest between parties
            3. Contradictory clauses or provisions
            4. Incompatible timelines, deliverables, or expectations
            5. Exclusivity or non-compete clauses that may conflict
            """
        else:
            task_description = f"""
            Compare specific clauses across the following contracts: {', '.join(contract_contents.keys())}

            Use the ReadContractFile tool to access the contract content as needed. Focus on the most important
            clauses and provisions rather than analyzing every detail, as some contracts may be lengthy.

            Focus on comparing these common clause types:
            1. Indemnification clauses
            2. Limitation of liability provisions
            3. Confidentiality and non-disclosure requirements
            4. Termination conditions
            5. Payment terms and conditions
            6. Intellectual property rights
            7. Dispute resolution mechanisms
            8. Governing law and jurisdiction
            """

        # Create the task
        task = Task(
            description=task_description,
            expected_output=f"A detailed {comparison_type.lower()} analysis of the contracts",
            agent=legal_expert
        )

        # Create crew and run
        crew = Crew(
            agents=[contract_analyst, legal_expert],
            tasks=[task],
            verbose=True
        )

        result = crew.kickoff()
        return result
    else:
        return f"Unable to compare contracts. Need at least 2 valid contracts, but only found {len(contract_contents)}."

# Process uploaded files
def process_files(uploaded_files):
    results = []

    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()

            # Generate a unique filename
            file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = Path(uploaded_file.name).suffix
            new_filename = f"{Path(uploaded_file.name).stem}_{timestamp}_{file_hash}{extension}"

            # Create a folder structure based on the current day
            current_date = datetime.now().strftime("%Y-%m-%d")
            subfolder_path = st.session_state.contracts_dir / current_date
            os.makedirs(subfolder_path, exist_ok=True)

            # Save file to contracts directory
            target_path = subfolder_path / new_filename

            # Use shutil.copy2 to copy across devices
            shutil.copy2(temp_file.name, target_path)
            os.unlink(temp_file.name)  # Clean up the temp file

            # Add to uploaded files list
            st.session_state.uploaded_files.append({
                "name": uploaded_file.name,
                "path": target_path,
                "timestamp": datetime.now()
            })

            # Get relative path for result
            rel_path = os.path.relpath(target_path, st.session_state.contracts_dir)

            results.append({
                "original_name": uploaded_file.name,
                "saved_as": new_filename,
                "path": rel_path
            })

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            # Continue with other files even if one fails
            continue

    return results


# Run a crew with the given agents and tasks
def run_crew(agents, tasks, process_type="sequential"):
    # Set process type
    if process_type.lower() == "hierarchical":
        process = Process.hierarchical
    else:
        process = Process.sequential

    # Create and run crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=process
    )

    return crew.kickoff()


# Main application layout
st.sidebar.title("Contract Analysis")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a function:", ["Upload & Analyze", "Chat with Contracts", "Compare Contracts"])

    st.header("Existing Contracts")
    contracts = get_all_contracts()
    if contracts:
        selected_contracts = st.multiselect(
            "Select contracts for analysis or comparison:",
            contracts
        )
    else:
        st.info("No contracts available in the repository.")
        selected_contracts = []

# Create tabs in the main area
tab1, tab2 = st.tabs(["Contract Analysis", "LLM Configuration"])

# Contract Analysis Tab
with tab1:
    if page == "Upload & Analyze":
        st.header("Upload Contracts")

        uploaded_files = st.file_uploader("Upload contract files (PDF, DOCX, TXT)",
                                          type=["pdf", "docx", "txt"],
                                          accept_multiple_files=True)

        # Option to compare with existing contracts
        st.subheader("Analysis Options")
        compare_with_existing = st.checkbox("Compare uploaded contracts with existing contracts")

        if compare_with_existing:
            existing_contracts = get_all_contracts()
            if existing_contracts:
                selected_existing = st.multiselect(
                    "Select existing contracts to compare with:",
                    existing_contracts
                )

                analysis_type = st.radio(
                    "Select analysis type:",
                    ["Conflicts of Interest", "Similarities", "Clause Comparison"],
                    key="upload_analysis_type"
                )
            else:
                st.info("No existing contracts available for comparison.")
                selected_existing = []

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process Uploaded Contracts"):
                if uploaded_files:
                    with st.spinner("Processing contracts..."):
                        results = process_files(uploaded_files)

                        # Display the upload results in a nice format
                        st.success(f"✅ Successfully processed {len(results)} contracts!")

                        # Show details of uploaded files in an expander
                        with st.expander("View Details", expanded=True):
                            for i, result in enumerate(results):
                                st.markdown(f"""
                                **File {i + 1}:** {result['original_name']}
                                - **Saved as:** {result['saved_as']}
                                - **Location:** {result['path']}
                                """)

                        # If comparison with existing is selected
                        if compare_with_existing and selected_existing and results:
                            st.info("Comparing uploaded contracts with selected existing contracts...")

                            # Get paths for new contracts
                            uploaded_paths = [result["path"] for result in results]

                            # Get appropriate analysis type
                            if "upload_analysis_type" in st.session_state:
                                analysis_type = st.session_state.upload_analysis_type
                            else:
                                analysis_type = "Conflicts of Interest"

                            analysis_map = {
                                "Conflicts of Interest": "conflicts",
                                "Similarities": "similarities",
                                "Clause Comparison": "clauses"
                            }

                            # Execute analysis against existing contracts
                            with st.spinner("Analyzing contracts - this may take a few minutes..."):
                                comparison_results = analyze_contracts_against_existing(
                                    uploaded_paths=uploaded_paths,
                                    existing_paths=selected_existing,
                                    analysis_type=analysis_map.get(analysis_type, "conflicts")
                                )

                                # Store results
                                st.session_state.comparison_results = comparison_results

                            # Display the results
                            st.subheader("Analysis Results")
                            with st.container():
                                st.text_area("Analysis Results", comparison_results, height=500)

                            # Format and display the results in a card-like container
                            st.markdown("""
                            <style>
                            .results-container {
                                background-color: #f8f9fa;
                                border-radius: 10px;
                                padding: 20px;
                                margin: 10px 0;
                                border-left: 5px solid #4CAF50;
                                color: #333333 !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)

                            st.markdown(f'<div class="results-container">{comparison_results}</div>',
                                        unsafe_allow_html=True)
                else:
                    st.warning("Please upload at least one contract.")

        # Display recently uploaded contracts with delete buttons
        if st.session_state.uploaded_files:
            st.header("Recently Uploaded Contracts")

            # Create columns for the contract list and delete buttons
            columns = st.columns([0.7, 0.3])

            with columns[0]:
                st.subheader("Contracts")
            with columns[1]:
                st.subheader("Actions")

            # Create a nice table for displaying the uploaded files with delete buttons
            for i, file_info in enumerate(
                    st.session_state.uploaded_files.copy()):  # Use copy to avoid modification during iteration
                col1, col2 = st.columns([0.7, 0.3])

                # Format the timestamp
                if isinstance(file_info["timestamp"], datetime):
                    timestamp = file_info["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = str(file_info["timestamp"])

                # Display contract info
                with col1:
                    st.markdown(f"""
                    **{file_info["name"]}**  
                    Uploaded: {timestamp}  
                    Path: {file_info["path"]}
                    """)

                # Add delete button
                with col2:
                    # Use a unique key for each button based on the file path
                    if st.button("🗑️ Delete", key=f"delete_{i}_{hash(str(file_info['path']))}"):
                        # Get the relative path
                        rel_path = os.path.relpath(file_info["path"], st.session_state.contracts_dir) if isinstance(
                            file_info["path"], Path) else file_info["path"]

                        # Delete the contract
                        if delete_contract(rel_path):
                            st.success(f"Successfully deleted {file_info['name']}.")
                            # Force refresh
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {file_info['name']}.")

                # Add a separator
                st.markdown("---")

    elif page == "Chat with Contracts":
        st.header("Chat with Contracts")

        # Display chat history with better formatting
        if st.session_state.chat_history:
            st.markdown("""
            <style>
            .user-message {
                background-color: #e6f7ff;
                border-radius: 10px;
                padding: 10px;
                margin: 5px 0;
                border-left: 5px solid #1890ff;
            }
            .ai-message {
                background-color: #f6ffed;
                border-radius: 10px;
                padding: 10px;
                margin: 5px 0;
                border-left: 5px solid #52c41a;
            }
            </style>
            """, unsafe_allow_html=True)

            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {message}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ai-message"><strong>AI:</strong>:black[{message}]</div>', unsafe_allow_html=True)

        # Chat input
        user_query = st.text_input("Ask about your contracts:", key="chat_input")

        if st.button("Send"):
            if user_query:
                if not st.session_state.llm_config or not st.session_state.llm_config.get('api_key'):
                    st.error("LLM not configured. Please set up the API key in the LLM Configuration tab.")
                else:
                    # Add user message to chat history
                    st.session_state.chat_history.append(("user", user_query))

                    with st.spinner("Thinking..."):
                        # Load agent configurations
                        agents_config = load_agents_config()

                        # Create agents
                        contract_analyst = create_agent_from_config(agents_config.get("contract_analyst", {
                            "role": "Contract Analyst",
                            "goal": "Answer questions about contracts accurately",
                            "backstory": "Expert in analyzing and explaining contract details"
                        }))

                        # Create task
                        task = Task(
                            description=f"Answer the following question about contracts: {user_query}",
                            expected_output="A clear, detailed answer to the user's question",
                            agent=contract_analyst
                        )

                        # Create crew and run
                        crew = Crew(
                            agents=[contract_analyst],
                            tasks=[task],
                            verbose=True
                        )

                        response = crew.kickoff()

                        # Add AI response to chat history
                        st.session_state.chat_history.append(("ai", response))

                    # This will refresh the UI without losing the state
                    st.rerun()
            else:
                st.warning("Please enter a query.")

    elif page == "Compare Contracts":
        st.header("Compare Contracts")

        # Add a button to delete contracts
        with st.expander("Manage Selected Contracts"):
            # List selected contracts with delete buttons
            if selected_contracts:
                st.subheader("Selected Contracts")

                for i, contract in enumerate(selected_contracts):
                    col1, col2 = st.columns([0.7, 0.3])

                    with col1:
                        st.write(f"📄 {i + 1}. {contract}")

                    with col2:
                        # Use a unique key for each delete button
                        if st.button("🗑️ Delete", key=f"delete_compare_{i}_{hash(contract)}"):
                            if delete_contract(contract):
                                st.success(f"Successfully deleted {contract}.")
                                # Force refresh
                                st.rerun()
                            else:
                                st.error(f"Failed to delete {contract}.")

        if len(selected_contracts) < 2:
            st.warning("Please select at least two contracts for comparison.")
        else:
            comparison_type = st.radio(
                "Select comparison type:",
                ["Similarities", "Conflicts of Interest", "Clause Comparison"]
            )

            # Add a testing mode checkbox
            test_mode = st.checkbox("Test file reading only")

            if test_mode:
                if st.button("Test Reading Selected Contracts"):
                    with st.spinner("Testing file reading..."):
                        # Create a file reading tool
                        file_reader = FileReadTool()

                        # Test each selected contract
                        for contract in selected_contracts:
                            st.subheader(f"Testing: {contract}")
                            result = file_reader._run(contract)

                            if result.startswith("File not found:") or result.startswith("Error reading file:"):
                                st.error(result)
                            else:
                                st.success(f"Successfully read file ({len(result)} characters)")
                                st.text_area("Preview:", result[:500] + "..." if len(result) > 500 else result,
                                             height=150)
            else:
                if st.button("Compare Selected Contracts"):
                    if not st.session_state.llm_config or not st.session_state.llm_config.get('api_key'):
                        st.error("LLM not configured. Please set up the API key in the LLM Configuration tab.")
                    else:
                        with st.spinner("Comparing contracts..."):
                            # Get full paths for selected contracts - they are already relative paths
                            contract_paths = selected_contracts

                            # Execute comparison
                            comparison_results = compare_contracts(
                                contract_paths=contract_paths,
                                comparison_type=comparison_type
                            )

                            # Store results
                            st.session_state.comparison_results = comparison_results

                            # Show the comparison results in a nicely formatted way
                            st.subheader("Comparison Results")
                            with st.container():
                                st.text_area("Comparison Results", comparison_results, height=500)

                            # Format and display the results in a card-like container
                            st.markdown("""
                            <style>
                            .comparison-container {
                                background-color: #f0f7ff;
                                border-radius: 10px;
                                padding: 20px;
                                margin: 10px 0;
                                border-left: 5px solid #007BFF;
                                color: #333333 !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)

                            st.markdown(f'<div class="comparison-container">{comparison_results}</div>',
                                        unsafe_allow_html=True)

            # Display currently selected contracts
            with st.expander("Selected Contracts", expanded=True):
                for i, contract in enumerate(selected_contracts):
                    st.write(f"📄 {i + 1}. {contract}")

# LLM Configuration Tab
with tab2:
    st.header("LLM Configuration")

    # Load existing config
    llm_config = load_llm_config()

    api_key = st.text_input("API Key (Optional)",
                            value=llm_config.get("api_key", ""),
                            type="password")

    api_endpoint = st.text_input("API Endpoint",
                                 value=llm_config.get("api_endpoint", "https://api.openai.com/v1"))

    st.subheader("Model Settings")
    col1, col2 = st.columns(2)

    with col1:
        model = st.text_input(
            "Model Name",
            value=llm_config.get("model", "gpt-4")
        )

        temperature = st.slider("Temperature", 0.0, 1.0, llm_config.get("temperature", 0.7), 0.1)
        max_tokens = st.slider("Max Tokens", 100, 2000, llm_config.get("max_tokens", 500), 50)

    with col2:
        top_p = st.slider("Top P", 0.0, 1.0, llm_config.get("top_p", 0.9), 0.1)
        frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, llm_config.get("frequency_penalty", 0.0), 0.1)
        presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, llm_config.get("presence_penalty", 0.0), 0.1)

    if st.button("Save LLM Configuration"):
        # Save configuration to YAML file
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "api_key": api_key,
            "api_endpoint": api_endpoint
        }

        save_llm_config(config)

        # Set appropriate environment variables if API key is provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            if api_endpoint:
                os.environ["OPENAI_API_BASE"] = api_endpoint

            st.success("LLM Configuration saved successfully!")
        else:
            # Allow configuration without API key
            st.success(
                "LLM Configuration saved successfully! Note: API Key is optional but required for most LLM operations.")

# First-time setup: Create default agent and task configurations
if not (configs_dir / "agents.yaml").exists():
    default_agents = {
        "contract_analyst": {
            "role": "Contract Analyst",
            "goal": "Analyze contracts thoroughly to find important information",
            "backstory": "Expert contract analyst with years of experience in legal document analysis",
            "allow_delegation": False
        },
        "legal_expert": {
            "role": "Legal Expert",
            "goal": "Identify legal implications and potential issues in contracts",
            "backstory": "Senior legal consultant with expertise in contract law and document analysis",
            "allow_delegation": True
        }
    }
    save_agents_config(default_agents)

if not (configs_dir / "tasks.yaml").exists():
    default_tasks = {
        "contract_analysis": {
            "description": "Analyze the provided contract and extract key information",
            "expected_output": "A detailed analysis of the contract including key clauses, parties, obligations, and potential issues",
            "agent": "Contract Analyst",
            "async_execution": False
        },
        "contract_comparison": {
            "description": "Compare multiple contracts to identify similarities, differences, or conflicts",
            "expected_output": "A detailed comparison highlighting important similarities and differences between the contracts",
            "agent": "Legal Expert",
            "async_execution": False
        }
    }
    save_tasks_config(default_tasks)