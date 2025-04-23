# Legal Document Analysis System

A powerful AI-driven tool for analyzing, comparing, and managing legal documents using OpenAI's GPT models.

## Features

- **Document Processing**: Upload and process legal documents (PDF, DOCX, TXT)
- **Legal Analysis**: Generate summaries, analyze provisions, extract definitions
- **Risk Assessment**: Identify and visualize legal risks in documents
- **Document Comparison**: Compare multiple documents to identify similarities and differences
- **Vector Search**: Find similar documents or passages using semantic search
- **Customizable AI Backend**: Configure OpenAI API settings or use compatible alternatives

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key (see Configuration section)
4. Run the application:
   ```
   streamlit run app.py
   ```

## Configuration

### Environment Variables

You can set configuration using environment variables or a `.env` file:

1. Create a `.env` file in the project root (use `sample.env` as a template)
2. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. (Optional) Customize API endpoint or model:
   ```
   OPENAI_API_BASE=https://your-custom-endpoint.com/v1
   OPENAI_API_MODEL=gpt-4o-latest
   ```

### API Settings Tab

The application includes an API Settings tab where you can configure:

- **API Key**: Set your OpenAI API key for the current session
- **API Endpoint**: Change the API endpoint URL (for custom deployments or compatible APIs)
- **Model Selection**: Choose from available models or specify a custom model

These settings can be adjusted at runtime without restarting the application.

## Usage

### Document Processing

1. Upload one or more legal documents using the file uploader in the sidebar
2. Click "Process Legal Documents" to analyze the documents
3. View the summary and analysis in the main interface

### Legal Analysis

Each processed document includes several analysis tabs:

- **Summary & Analysis**: Overview of the document and key legal elements
- **Legal Provisions**: Extraction of obligations, rights, and key provisions
- **Legal Definitions**: Formal terms defined in the document
- **Risk Assessment**: Identification and visualization of legal risks
- **Similar Documents**: Documents with similar content in the database

### Search

Use the search functionality in the sidebar to find relevant documents or passages based on semantic similarity.

### Document Database

The Document Database tab provides access to all processed documents in the system and allows batch processing of documents in the `contracts` folder.

## Advanced Configuration

### Custom API Endpoints

You can use alternative API endpoints compatible with the OpenAI API format:

1. Self-hosted models (e.g., LiteLLM, OpenAI-compatible servers)
2. Enterprise deployments with custom endpoints
3. Azure OpenAI Service with appropriate configuration

### Model Selection

The application supports various OpenAI models:

- `gpt-4o-latest`: Latest GPT-4o model (recommended)
- `gpt-4o`: GPT-4o model
- `gpt-4-turbo`: GPT-4 Turbo model
- `gpt-4`: GPT-4 model
- `gpt-3.5-turbo`: GPT-3.5 Turbo model

You can also specify custom model identifiers for compatible deployments.

## Troubleshooting

### API Connection Issues

If you experience issues connecting to the API:

1. Verify your API key is correct
2. Check that your API endpoint URL is properly formatted and accessible
3. Ensure the selected model is available in your OpenAI subscription or endpoint
4. Check the system messages section for detailed error information

### Processing Errors

If document processing fails:

1. Check if the file format is supported (PDF, DOCX, TXT)
2. Ensure the document is not corrupted or password-protected
3. For large documents, try processing in smaller chunks
4. Check system messages for specific error details

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.