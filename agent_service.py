import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, FunctionTool

# 1. Load environment variables
load_dotenv()
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
# Fallback to AZURE_SEARCH_ADMIN_KEY to align with existing .env setups
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY") or os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME", "imac-guidelines")
AGENT_NAME = "IMAC_Immunisation_Advisor"

# 2. Connect to Microsoft Foundry
project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

# 3. Define the Search Tool (This gives the agent access to the Handbook!)
def search_clinical_guidelines(query: str) -> str:
    """
    Searches the official New Zealand Immunisation Handbook for clinical guidance.
    The agent will automatically use this tool when a user asks a medical question.
    """
    print(f"🔍 Agent is searching the handbook for: '{query}'...")
    
    try:
        credential = AzureKeyCredential(SEARCH_KEY)
        search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX, credential=credential)
        
        # Perform a simple search (you can upgrade this to vector search later)
        results = search_client.search(search_text=query, top=3)
        
        # Combine the top results into a single text block for the AI to read
        retrieved_text = "\n\n".join([f"Source: {doc.get('source_url', doc.get('title', 'Unknown Source'))}\nContent: {doc.get('content', '')}" for doc in results])
        return retrieved_text
        
    except Exception as e:
        return f"Error connecting to the clinical database: {str(e)}"

# Register the Python function as an AI Tool
handbook_search_tool = FunctionTool(user_functions={search_clinical_guidelines})

def setup_agent():
    """Run this function to define the agent's strict clinical rules and equip its tools."""
    print("Creating/Updating Agent...")
    
    strict_instructions = (
        "You are an Immunisation Guidelines Advisor Agent for IMAC. "
        "Your role is to retrieve and summarize official New Zealand immunisation guidance. "
        "CLINICAL SAFETY RULE: You must never provide professional medical diagnosis or independent treatment suggestions. "
        "Always use the 'search_clinical_guidelines' tool to find answers. "
        "If the answer is not in the search results, state clearly: 'I cannot find the answer in the official guidance.' "
        "CITATION RULE: Always explicitly list the sources you used at the bottom of your response, based on the 'Source' data provided by your search tool."
    )

    agent = project.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(
            model="gpt-4o",
            instructions=strict_instructions,
            tools=[handbook_search_tool] # <--- We gave the agent the tool here!
        ),
    )
    print(f"✅ Agent created successfully! Version: {agent.version}")
    return agent

def chat_with_advisor(user_input: str, conversation_id: str = None):
    """Run this function every time the user asks a question."""
    openai = project.get_openai_client()

    if not conversation_id:
        conversation = openai.conversations.create()
        conversation_id = conversation.id
        print(f"\nStarted new conversation thread: {conversation_id}")

    print(f"User: {user_input}")
    
    # Send the question to the agent
    response = openai.responses.create(
        conversation=conversation_id,
        extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"}},
        input=user_input,
    )
    
    print(f"Agent: {response.output_text}\n")
    return conversation_id, response.output_text

# ==========================================
if __name__ == "__main__":
    setup_agent()
    
    print("\n🤖 Agent Service Test Terminal (Type 'exit' to quit)")
    convo_id = None
    
    while True:
        user_msg = input("\nYou: ")
        if user_msg.lower() in ['exit', 'quit']:
            break
            
        convo_id, reply = chat_with_advisor(user_msg, convo_id)