# CrewAI Contract Analysis Application

A powerful, AI-driven contract analysis system built with CrewAI that enables users to upload, compare, and interact with contracts in an intelligent way.

## Features

* **Contract Management**
  * Upload PDF, DOCX, and TXT contracts
  * Automatic organization in date-based folders
  * Support for complex folder hierarchies
* **Intelligent Analysis**
  * Compare newly uploaded contracts against existing contracts
  * Identify potential conflicts of interest
  * Find similarities across contracts
  * Perform clause-by-clause comparisons
* **Interactive Interface**
  * Chat with your contracts to ask specific questions
  * Select and compare multiple contracts
  * View detailed analysis reports
* **Flexible Configuration**
  * Configure any LLM model by name
  * Customize API endpoints
  * Adjust generation parameters
  * Optional API key for flexibility

## Installation

### Prerequisites

* Python 3.10 or higher (compatible with CrewAI requirements)
* API key for your preferred LLM provider (optional but recommended)

### Setup

1. Clone this repository
   ```bash
   git clone <repository-url>
   cd crewai-contract-analysis
   ```
2. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```
3. Run the setup script to create necessary directories and default configurations
   ```bash
   python setup.py
   ```
4. Launch the application
   ```bash
   streamlit run app.py
   ```

## Usage Guide

### Uploading Contracts

1. Navigate to the "Upload & Analyze" section
2. Upload one or more contract files
3. Optional: Enable "Compare with existing contracts" to analyze against your existing repository
4. Select which existing contracts to compare against
5. Choose the analysis type: Conflicts of Interest, Similarities, or Clause Comparison
6. Click "Process Uploaded Contracts"

### Comparing Existing Contracts

1. Select contracts from the sidebar
2. Navigate to the "Compare Contracts" section
3. Select comparison type
4. Click "Compare Selected Contracts"
5. View the detailed comparison report

### Chatting with Contracts

1. Navigate to the "Chat with Contracts" section
2. Type your question about any contracts in your repository
3. View AI-generated responses based on the contract content

### Configuring LLM

1. Navigate to the "LLM Configuration" tab
2. Enter your API key (optional)
3. Configure your preferred API endpoint
4. Enter your desired model name
5. Adjust temperature and other generation parameters
6. Click "Save LLM Configuration"

## Project Structure

```
.
├── app.py                 # Main application file
├── setup.py               # Setup script
├── requirements.txt       # Required packages
├── contracts/             # Directory for stored contracts (with subfolders)
└── configs/               # Configuration files
    ├── agents.yaml        # Agent definitions
    ├── tasks.yaml         # Task definitions
    └── llm_config.yaml    # LLM configuration
```

## How It Works

This application uses CrewAI to create specialized AI agents that collaborate to analyze contracts:

1. **Contract Analyst Agent**: Focuses on extracting and understanding key information from contracts
2. **Legal Expert Agent**: Specializes in identifying legal implications, conflicts, and comparing contract clauses

These agents use a variety of tools to read and process contract files and work together to provide comprehensive analysis.

## Configuration Files

### agents.yaml

Defines the characteristics of AI agents:

```yaml
contract_analyst:
  role: "Contract Analyst"
  goal: "Analyze contracts thoroughly to find important information"
  backstory: "Expert contract analyst with years of experience"
  allow_delegation: false

legal_expert:
  role: "Legal Expert"
  goal: "Identify legal implications and potential issues in contracts"
  backstory: "Senior legal consultant with expertise in contract law"
  allow_delegation: true
```

### tasks.yaml

Defines the tasks agents can perform:

```yaml
contract_analysis:
  description: "Analyze the provided contract and extract key information"
  expected_output: "A detailed analysis of the contract"
  agent: "Contract Analyst"
  async_execution: false
```

### llm\_config.yaml

Configures the LLM:

```yaml
model: "gpt-4"
temperature: 0.7
max_tokens: 500
top_p: 0.9
frequency_penalty: 0.0
presence_penalty: 0.0
api_key: ""
api_endpoint: "https://api.openai.com/v1"
```

## Acknowledgments

* Built with [CrewAI](https://github.com/crewAIInc/crewAI)
* User interface powered by [Streamlit](https://streamlit.io/)

## Hardware Recommendations for Cloudera AI

### Minimum Requirements

* **CPU**: 4+ cores for optimal performance
* **Memory**: 8 GB RAM minimum
*

### GPU Recommendations

* **NVIDIA GPU Support**: NVIDIA A100 or A30 Tensor Core GPUs for best performance
* **GPU Memory**: Minimum 16 GB VRAM per GPU
* **Configuration**: For production ML workloads, a minimum of 4 NVIDIA-Certified servers with 1-2 GPUs per server is recommended

The hardware recommendations are based on Cloudera's documentation for production deployments. For development or testing environments, requirements may be lower but SSD storage is always required for optimal performance. GPU requirements will vary based on specific workloads and model complexity with users able to request a specific number of GPU instances up to the total available.
