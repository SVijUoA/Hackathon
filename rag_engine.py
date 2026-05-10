import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve Azure Variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini") # Use 4o-mini if available
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01-preview")

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
search_index = os.getenv("AZURE_SEARCH_INDEX_NAME", "imac-guidelines")

# Initialize Azure OpenAI Client
openai_client = None
if endpoint and api_key:
    # Check if endpoint is a placeholder
    if "your-resource.openai.azure.com" in endpoint:
        print("Warning: Azure OpenAI endpoint appears to be a placeholder. Please update your .env file with real Azure OpenAI credentials.")
    else:
        try:
            openai_client = AzureOpenAI(
                api_key=api_key,
                api_version=azure_openai_api_version,
                azure_endpoint=endpoint
            )
        except Exception as e:
            print(f"Failed to initialize Azure OpenAI client: {e}")

# Configure Azure AI Search
search_client = None
if search_endpoint and search_key:
    # Check if endpoint is a placeholder
    if "your-search.search.windows.net" in search_endpoint or search_key == "your-search-key-here":
        print("Warning: Azure Search endpoint/key appears to be a placeholder. Please update your .env file with real Azure Search credentials.")
    else:
        try:
            search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=search_index,
                credential=AzureKeyCredential(search_key)
            )
        except Exception as e:
            print(f"Failed to initialize Azure Search client: {e}")

def generate_response(user_prompt):
    """
    The core RAG backend function.
    1. Retrieves documents from Azure AI Search.
    2. Injects them into the Prompt.
    3. Calls Azure OpenAI.
    """
    # Check if we have valid Azure clients
    has_openai = openai_client is not None
    has_search = search_client is not None

    # If no Azure services are configured, provide mock responses for development
    if not has_openai:
        return get_mock_response(user_prompt)

    retrieved_chunks = []
    citations = []

    # --- STEP A: RETRIEVAL ---
    if has_search:
        try:
            # Search the index for the top 3 most relevant paragraphs
            results = search_client.search(search_text=user_prompt, top=3)
            for doc in results:
                # NOTE: Update "content" and "source" based on what your scraper teammate names the columns!
                chunk = doc.get("content", "")
                source = doc.get("source", "Official IMAC Guidance")

                if chunk:
                    retrieved_chunks.append(chunk)
                if source and source not in citations:
                    citations.append(source)
        except Exception as e:
            print(f"Search connection error: {e}")
            # Continue with empty context if search fails

    # Combine the found text into one block for the AI to read
    context = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant guidance found."

    # --- STEP B: AUGMENTATION & GUARDRAILS ---
    system_message = (
        "You are an IMAC immunisation advisor. Answer clinical immunisation questions "
        "using ONLY the provided Context.\n"
        "If the answer cannot be explicitly found in the Context, "
        "you MUST reply EXACTLY with: 'I couldn't find a clear answer in approved guidance.'\n"
        "Do not use your internal knowledge. Do not guess or infer."
    )

    # --- STEP C: GENERATION ---
    try:
        response = openai_client.chat.completions.create(
            model=azure_openai_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_prompt}"}
            ],
            max_tokens=500,
            temperature=0.0  # CRITICAL: 0.0 enforces 'Accuracy over Recall'
        )

        ai_answer = response.choices[0].message.content.strip()

        # Final safety check to trigger the Front-end UI warning
        if "couldn't find a clear answer" in ai_answer.lower():
            return "I couldn't find a clear answer in approved guidance.", []

        return ai_answer, citations

    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
            return "System Error: Unable to connect to Azure OpenAI service. Please check your internet connection and Azure service status.", []
        elif "authentication" in error_msg or "unauthorized" in error_msg or "invalid" in error_msg:
            return "System Error: Azure OpenAI authentication failed. Please check your API key and endpoint in the .env file.", []
        else:
            return f"System Error: {str(e)}", []


def get_mock_response(user_prompt):
    """
    Provides mock responses for development when Azure services are not configured.
    This allows testing the UI without requiring Azure credentials.
    """
    prompt_lower = user_prompt.lower()

    if "vaccine schedule" in prompt_lower or "schedule" in prompt_lower:
        return (
            "The standard vaccine schedule for children in New Zealand follows the National Immunisation Schedule. Key milestones include:\n\n"
            "- 6 weeks: DTaP-IPV-HepB/Hib, PCV13, Rotavirus\n"
            "- 3 months: DTaP-IPV-HepB/Hib, PCV13, Rotavirus\n"
            "- 5 months: DTaP-IPV-HepB/Hib, PCV13\n"
            "- 12 months: MMR, PCV13, Meningococcal B\n"
            "- 15 months: MMR, Varicella\n\n"
            "Please consult the latest IMAC guidelines for complete details.",
            [
                "[NZ Immunisation Handbook 2023](https://www.health.govt.nz/publication/nz-immunisation-handbook-2023)",
                "[IMAC National Immunisation Schedule](https://immune.org.nz/immunisation/programmes/national-immunisation-schedule)"
            ]
        )

    elif "contraindications" in prompt_lower:
        return (
            "Common contraindications for vaccines include:\n\n"
            "- Severe allergic reaction to a previous dose or vaccine component\n"
            "- Immunodeficiency states (for live vaccines)\n"
            "- Pregnancy (for certain live vaccines)\n"
            "- Current febrile illness\n\n"
            "Always assess individual patient circumstances before vaccination.",
            [
                "[NZ Immunisation Handbook - Contraindications](https://www.health.govt.nz/publication/nz-immunisation-handbook-2023)",
                "[IMAC Clinical Guidelines](https://www.tewhatuora.govt.nz/for-health-professionals/clinical-guidance/immunisation-handbook)"
            ]
        )

    elif "catch-up" in prompt_lower or "catch up" in prompt_lower or "missed" in prompt_lower:
        return (
            "Catch-up vaccination is important for children who have missed scheduled doses. Key principles include:\n\n"
            "**General Approach:**\n"
            "- Administer all vaccines appropriate for current age, regardless of when previous doses were given\n"
            "- Minimum intervals must still be observed between doses\n"
            "- No need to restart the series if the interval has been exceeded\n\n"
            "**Spacing:**\n"
            "- Most vaccines can be given simultaneously at different injection sites\n"
            "- If multiple live vaccines are needed, they should be given on the same day or 28 days apart\n\n"
            "**Special Considerations:**\n"
            "- Review the child's immunisation history carefully\n"
            "- Assess for any contraindications before catch-up vaccination\n"
            "- Consider the child's age and which vaccines are due\n\n"
            "Refer to the NZ Immunisation Handbook for specific catch-up schedules by age.",
            [
                "[NZ Immunisation Handbook - Catch-up Schedules](https://www.health.govt.nz/publication/nz-immunisation-handbook-2023)",
                "[IMAC Catch-up Vaccination Guidelines](https://immune.org.nz/immunisation/catch-schedules)"
            ]
        )

    else:
        return (
            "This is a development mock response. To get real IMAC guidance, please configure your Azure OpenAI and Azure Search credentials in the .env file.\n\n"
            "The question you asked would normally be answered using official IMAC guidelines retrieved from our knowledge base.",
            ["[IMAC Website](https://immune.org.nz)"]
        )