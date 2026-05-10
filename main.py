# 1. IMPORTS ALWAYS GO FIRST
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st # Or your specific framework

# 2. THE BOILERPLATE GOES HERE (Right after imports)
load_dotenv() 

# 3. RETRIEVE YOUR VARIABLES
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# 4. REST OF YOUR CODE (App logic, functions, etc.)
def run_adviser_agent():
    # Application logic starts here...
    pass