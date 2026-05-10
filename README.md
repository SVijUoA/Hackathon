# IMAC Immunisation Advisor Agent

A Streamlit-based AI assistant that provides clinical guidance on immunisation using Retrieval-Augmented Generation (RAG) with Azure OpenAI and Azure AI Search.

## 🚀 Quick Start

### Development Mode (No Azure Setup Required)
The app includes mock responses for testing the UI without Azure credentials:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open http://localhost:8501 in your browser

The app will automatically detect missing Azure credentials and provide mock responses for common queries.

## 🔧 Production Setup (Azure Services)

### Prerequisites
- Azure subscription
- Azure OpenAI resource
- Azure AI Search resource

### Environment Variables
Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_MODEL=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-06-01-preview

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_ADMIN_KEY=your-search-admin-key
AZURE_SEARCH_INDEX_NAME=imac-guidelines
```

### Setting up Azure Resources

1. **Azure OpenAI**:
   - Create an Azure OpenAI resource in the Azure portal
   - Deploy a GPT-4o-mini model
   - Copy the endpoint and API key

2. **Azure AI Search**:
   - Create an Azure AI Search resource
   - Create an index named `imac-guidelines` with fields for `content` and `source`
   - Upload IMAC guidelines documents
   - Copy the endpoint and admin key

## 📁 Project Structure

```
├── app.py                 # Streamlit UI (frontend)
├── rag_engine.py          # RAG backend with Azure integrations
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
└── README.md             # This file
```

## 🛡️ Safety Features

- **Clinical Guardrails**: Only uses official IMAC guidelines
- **Low Confidence Detection**: Warns when information isn't found
- **Source Citations**: Provides references for all answers
- **User Feedback**: Collects feedback to improve responses

## 🔍 How It Works

1. **User Query**: Clinician asks an immunisation question
2. **Retrieval**: Azure AI Search finds relevant IMAC guidelines
3. **Augmentation**: Context is injected into the prompt
4. **Generation**: Azure OpenAI generates clinically accurate response
5. **Safety Check**: Response is validated against guidelines

## 🧪 Testing

The app includes comprehensive mock responses for development testing. Common test queries:

- "What is the standard vaccine schedule for children?"
- "What are the contraindications for vaccines?"
- "How do I handle catch-up vaccinations?"

## 📊 Architecture

- **Frontend**: Streamlit (app.py) - Clean UI focused on user experience
- **Backend**: RAG Engine (rag_engine.py) - Handles Azure services and AI logic
- **Data**: Azure AI Search index with IMAC guidelines
- **AI**: Azure OpenAI GPT-4o-mini for response generation

## 🚨 Important Notes

- **Clinical Use**: This is an AI assistant designed to support, not replace, clinical judgment
- **Verification**: Always verify information against official IMAC guidelines
- **Development Mode**: Mock responses are for UI testing only - not for clinical use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both mock and real Azure services
5. Submit a pull request