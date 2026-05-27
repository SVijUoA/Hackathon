"""
Foundry-compatible RAG backend for IMAC immunisation advisor.
Uses Microsoft Foundry for model inference (chat completions and embeddings).
Compatible with Streamlit app via the same generate_response() interface.
"""

import os
import re
import time
import uuid
import hashlib
from io import BytesIO
from typing import List, Dict, Tuple
import json
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()


def warn_placeholder_config(foundry_endpoint, foundry_key, foundry_chat_model, foundry_embedding_model):
    """Warn at startup when .env contains placeholder Foundry credentials."""
    warnings = []

    if not foundry_endpoint or "your-foundry.azure.com" in foundry_endpoint:
        warnings.append("Foundry endpoint is not configured or is still using the placeholder value.")
    if not foundry_key or foundry_key.startswith("your-"):
        warnings.append("Foundry API key is not configured or is still using the placeholder value.")
    if not foundry_chat_model or foundry_chat_model.startswith("your-"):
        warnings.append("Foundry chat model name is not configured or is still using the placeholder value.")
    if not foundry_embedding_model or foundry_embedding_model.startswith("your-"):
        warnings.append("Foundry embedding model name is not configured or is still using the placeholder value.")

    if warnings:
        print("WARNING: Invalid Foundry configuration detected in .env. Please update the following settings:")
        for warning in warnings:
            print(f" - {warning}")
        print("The app will fall back to mock mode unless valid Foundry credentials are provided.")


# Retrieve Foundry Variables
foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")
foundry_api_key = os.getenv("FOUNDRY_API_KEY")
foundry_chat_model = os.getenv("FOUNDRY_CHAT_MODEL", "gpt-5-mini")
foundry_embedding_model = os.getenv("FOUNDRY_EMBEDDING_MODEL", "text-embedding-3-small")
foundry_api_version = os.getenv("FOUNDRY_API_VERSION", "2025-08-07")

VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./chroma_store")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "imac_guidance")

warn_placeholder_config(foundry_endpoint, foundry_api_key, foundry_chat_model, foundry_embedding_model)

# Security configuration
MAX_USER_PROMPT_LENGTH = 1000
MAX_SEARCH_RESULTS = 3
MAX_VECTOR_RESULTS = 5
MAX_FOUNDARY_OUTPUT_TOKENS = 1024
MAX_FOUNDARY_RETRY_OUTPUT_TOKENS = 2048

# Data sources for ingestion
SOURCE_URLS = [
    "https://immune.org.nz",
    "https://www.health.govt.nz/publication/nz-immunisation-handbook-2023",
    "https://immune.org.nz/immunisation/programmes/national-immunisation-schedule",
    "https://www.tewhatuora.govt.nz/for-health-professionals/clinical-guidance/immunisation-handbook",
    "https://static.info.content.health.nz/docs/health-pros/topics/immunisations/immunisation-handbook-2026-v2.pdf",
]

HEADERS = {
    "User-Agent": "IMAC-RAG-Agent/1.0 (+https://immune.org.nz)"
}


class FoundryClient:
    """Wrapper for Foundry inference API calls (chat completions and embeddings)."""
    
    def __init__(self, endpoint: str, api_key: str, api_version: str = foundry_api_version):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.parsed_endpoint = urlparse(self.endpoint)
        self.service_root = f"{self.parsed_endpoint.scheme}://{self.parsed_endpoint.netloc}"
        self.use_responses_api = self.endpoint.endswith("/responses") or "/responses/" in self.endpoint
        self.base_endpoint = self.endpoint
        if self.use_responses_api:
            self.base_endpoint = self.endpoint[: self.endpoint.rfind("/responses")]

        if self.use_responses_api:
            self.embeddings_url = f"{self.service_root}/openai/v1/embeddings"
        elif self.endpoint.endswith("/embeddings"):
            self.embeddings_url = self.endpoint
        elif self.endpoint.endswith("/openai/v1"):
            self.embeddings_url = f"{self.endpoint}/embeddings"
        elif "/openai/v1/" in self.endpoint:
            self.embeddings_url = re.sub(r"/openai/v1/.*$", "/openai/v1/embeddings", self.endpoint)
        else:
            self.embeddings_url = f"{self.endpoint}/openai/v1/embeddings"
    
    def _build_url(self, path: str) -> str:
        if path == "embeddings":
            return self.embeddings_url

        if self.use_responses_api:
            if path == "responses":
                return self.endpoint
            return f"{self.base_endpoint}/{path}"
        return f"{self.endpoint}/{path}"

    def _request(self, path: str, payload: dict):
        url = self._build_url(path)
        if self.use_responses_api or "/openai/v1/" in self.endpoint:
            params = {}
        else:
            params = {"api-version": self.api_version} if self.api_version else {}

        response = requests.post(url, headers=self.headers, params=params, json=payload, timeout=30)
        if response.status_code >= 400:
            if response.status_code == 404 and path == "embeddings" and self.use_responses_api:
                fallback_urls = [
                    f"{self.base_endpoint}/{path}",
                    f"{self.endpoint}/{path}",
                ]
                for fallback_url in fallback_urls:
                    if fallback_url == url:
                        continue
                    print(f"Warning: embeddings endpoint 404; trying fallback {fallback_url}")
                    alt_response = requests.post(fallback_url, headers=self.headers, params=params, json=payload, timeout=30)
                    if alt_response.status_code < 400:
                        return alt_response.json()
            raise RuntimeError(f"Foundry API error {response.status_code}: {response.text or response.reason} (url={url})")
        return response.json()
    
    def chat_completions_create(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.0):
        """Call Foundry chat completions API."""
        if self.use_responses_api:
            serialized_input = "\n".join(msg["content"] for msg in messages)
            payload = {
                "model": model,
                "input": serialized_input,
                "max_output_tokens": max_tokens,
            }
            return self._request("responses", payload)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._request("chat/completions", payload)
    
    def _extract_embeddings(self, response: dict) -> List[List[float]]:
        if not isinstance(response, dict):
            raise ValueError("Embeddings response is not a dict")

        if "data" in response and isinstance(response["data"], list):
            embeddings = []
            for item in response["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
            if embeddings:
                return embeddings

        if "output" in response and isinstance(response["output"], list):
            embeddings = []
            for item in response["output"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
            if embeddings:
                return embeddings

        if "embeddings" in response and isinstance(response["embeddings"], list):
            embeddings = [item for item in response["embeddings"] if isinstance(item, list)]
            if embeddings:
                return embeddings

        raise KeyError("Unable to extract embeddings from Foundry response")

    def embeddings_create(self, model: str, input_texts: List[str]):
        """Call Foundry embeddings API."""
        payload = {
            "model": model,
            "input": input_texts,
        }
        path = "embeddings" if self.use_responses_api else "embeddings"
        response = self._request(path, payload)
        return {"data": [{"embedding": emb} for emb in self._extract_embeddings(response)]}


# Initialize Foundry Client
foundry_client = None
if foundry_endpoint and foundry_api_key and "your-foundry.azure.com" not in foundry_endpoint:
    try:
        foundry_client = FoundryClient(
            endpoint=foundry_endpoint,
            api_key=foundry_api_key,
            api_version=foundry_api_version,
        )
    except Exception as e:
        print(f"Failed to initialize Foundry client: {e}")


def create_vector_store(persist_directory: str = VECTOR_STORE_DIR) -> chromadb.api.client.Client:
    return chromadb.Client(
        Settings(
            persist_directory=persist_directory,
            is_persistent=True,
        )
    )


def get_collection_name(collection_item):
    """Extract collection name from either dict or Collection object."""
    if isinstance(collection_item, dict):
        return collection_item.get("name")
    return getattr(collection_item, "name", None)


def vector_collection_exists(collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> bool:
    client = create_vector_store(persist_directory)
    return collection_name in [get_collection_name(collection) for collection in client.list_collections()]


def vector_collection_has_data(collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> bool:
    client = create_vector_store(persist_directory)
    if collection_name not in [get_collection_name(collection) for collection in client.list_collections()]:
        return False

    collection = client.get_collection(collection_name)
    try:
        return collection.count() > 0
    except Exception:
        documents = collection.get().get('documents', [])
        return bool(documents)


def fetch_page(url: str, timeout: int = 20) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def fetch_pdf_text(url: str, timeout: int = 30) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    reader = PdfReader(BytesIO(response.content))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_visible_text_from_html(html: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style tags to avoid parsing them
    for tag in soup.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()
    
    title = (soup.title.string or "").strip() if soup.title else "IMAC Guidance"

    candidates: List[str] = []
    
    # Primary: try semantic content tags
    for selector in ["article", "main", "section[role='main']", "div[role='main']"]:
        for element in soup.select(selector):
            text = element.get_text(separator="\n", strip=True)
            if len(text) > 120:
                candidates.append(text)
    
    # Secondary: if semantic tags found minimal content, extract paragraphs and headings
    if len(candidates) < 2:
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"]):
            text = element.get_text(separator="\n", strip=True)
            if len(text) >= 30:  # Lower threshold for individual elements
                candidates.append(text)
    
    # Tertiary: if still minimal, grab substantial divs
    if len(candidates) < 3:
        for div in soup.find_all("div", class_=None):
            text = div.get_text(separator="\n", strip=True)
            if 200 <= len(text) <= 2000:
                candidates.append(text)

    cleaned: List[str] = []
    for fragment in candidates:
        normalized = re.sub(r"\s+", " ", fragment).strip()
        if len(normalized) >= 50:  # Slightly relaxed threshold
            cleaned.append(normalized)

    return title or "IMAC Guidance", cleaned


def chunk_text(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_documents_from_url(url: str) -> List[Dict[str, str]]:
    if url.lower().endswith(".pdf"):
        raw_text = fetch_pdf_text(url)
        title = "NZ Immunisation Handbook PDF"
        text_blocks = [raw_text]
    else:
        html = fetch_page(url)
        title, text_blocks = extract_visible_text_from_html(html)

    documents: List[Dict[str, str]] = []
    for section_index, block in enumerate(text_blocks, start=1):
        for chunk in chunk_text(block):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            documents.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#{section_index}-{chunk_hash}")),
                    "title": title,
                    "content": chunk,
                    "url": url,
                    "source": url,
                    "section": title,
                }
            )
    print(f"Built {len(documents)} chunk documents from {url}")
    return documents


def embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    if foundry_client is None:
        raise RuntimeError("Foundry client is not configured for embeddings.")

    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = foundry_client.embeddings_create(
                model=foundry_embedding_model,
                input_texts=batch,
            )
            # Extract embeddings from Foundry API response
            for item in response.get("data", []):
                embeddings.append(item["embedding"])
            time.sleep(0.2)
        except Exception as e:
            print(f"Warning: embedding batch failed: {e}")
            # Return dummy embeddings on failure
            embeddings.extend([[0.0] * 1536 for _ in batch])
    return embeddings


def ingest_sources(collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> int:
    if foundry_client is None:
        raise RuntimeError("Foundry client is not configured. Cannot ingest vector data.")

    client = create_vector_store(persist_directory)
    existing = [get_collection_name(collection) for collection in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.get_or_create_collection(name=collection_name)

    documents: List[Dict[str, str]] = []
    for url in SOURCE_URLS:
        try:
            documents.extend(build_documents_from_url(url))
            time.sleep(1.0)
        except Exception as exc:
            print(f"Warning: failed to ingest {url}: {exc}")

    if not documents:
        print("No documents were created for ingestion.")
        return 0

    contents = [doc["content"] for doc in documents]
    embeddings = embed_texts(contents)
    ids = [doc["id"] for doc in documents]
    metadatas = [
        {
            "title": doc["title"],
            "url": doc["url"],
            "source": doc["source"],
            "section": doc["section"],
        }
        for doc in documents
    ]

    collection.add(
        ids=ids,
        documents=contents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(documents)} chunk embeddings into collection '{collection_name}'.")
    return len(documents)


def normalize_query_results(results):
    """Handle various Chroma query result shapes."""
    if isinstance(results, dict):
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
    else:
        documents = getattr(results, "documents", [[]])
        metadatas = getattr(results, "metadatas", [[]])

    documents = documents[0] if documents and isinstance(documents[0], list) else documents
    metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
    return documents, metadatas


def query_vector_store(question: str, collection_name: str = VECTOR_COLLECTION_NAME, persist_directory: str = VECTOR_STORE_DIR) -> Tuple[List[str], List[str]]:
    client = create_vector_store(persist_directory)
    if collection_name not in [get_collection_name(collection) for collection in client.list_collections()]:
        return [], []

    collection = client.get_collection(collection_name)
    
    try:
        response = foundry_client.embeddings_create(
            model=foundry_embedding_model,
            input_texts=[question],
        )
        query_embedding = response["data"][0]["embedding"]
    except Exception as e:
        print(f"Embedding generation failed: {e}")
        return [], []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=MAX_VECTOR_RESULTS,
        include=["documents", "metadatas", "distances"],
    )

    documents, metadatas = normalize_query_results(results)
    chunks: List[str] = []
    citations: List[str] = []
    for doc, metadata in zip(documents, metadatas):
        chunks.append(doc)
        source = metadata.get("source") or metadata.get("url") or "Official IMAC Guidance"
        if source not in citations:
            citations.append(source)

    return chunks, citations


def ensure_vector_index(persist_directory: str = VECTOR_STORE_DIR, collection_name: str = VECTOR_COLLECTION_NAME) -> bool:
    if vector_collection_has_data(collection_name, persist_directory):
        client = create_vector_store(persist_directory)
        collection = client.get_collection(collection_name)
        try:
            count = collection.count()
        except Exception:
            count = len(collection.get().get("documents", []))
        print(f"Vector collection '{collection_name}' already exists with {count} documents.")
        return True

    if vector_collection_exists(collection_name, persist_directory):
        print(f"Existing vector collection '{collection_name}' is empty or invalid. Recreating it.")
        client = create_vector_store(persist_directory)
        client.delete_collection(collection_name)

    try:
        ingest_sources(collection_name, persist_directory)
        return vector_collection_has_data(collection_name, persist_directory)
    except Exception as e:
        print(f"Failed to create vector index: {e}")
        return False


def get_mock_response(user_prompt):
    """
    Provides mock responses for development when Foundry is not configured.
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
            ["[NZ Immunisation Handbook 2023](https://www.health.govt.nz/publication/nz-immunisation-handbook-2023)"]
        )
    elif "catch-up" in prompt_lower or "missed" in prompt_lower:
        return (
            "**Catch-up Vaccination Guidelines:**\n\n"
            "- Give vaccines appropriate for current age, regardless of when previous doses were given\n"
            "- Minimum intervals must still be observed between doses\n"
            "- No need to restart the series if the interval has been exceeded\n\n"
            "Refer to the NZ Immunisation Handbook for specific catch-up schedules by age.",
            ["[NZ Immunisation Handbook - Catch-up Schedules](https://www.health.govt.nz/publication/nz-immunisation-handbook-2023)"]
        )
    else:
        return (
            "This is a secure development mock response. To get real IMAC guidance, please configure your Foundry credentials in the .env file and build the local vector store.\n\n"
            "The question you asked would normally be answered using official IMAC guidelines retrieved from our knowledge base.",
            ["[IMAC Website](https://immune.org.nz)"]
        )


def generate_response(user_prompt):
    """
    Secure RAG backend function using Foundry for inference.
    Compatible with Streamlit app interface.
    """
    user_prompt = user_prompt.strip()
    if not user_prompt:
        return "Security Alert: Empty query received. Please provide a valid clinical question.", []

    if len(user_prompt) > MAX_USER_PROMPT_LENGTH:
        return f"Security Alert: Query exceeds maximum allowed length ({MAX_USER_PROMPT_LENGTH} characters).", []

    if not foundry_client:
        return get_mock_response(user_prompt)

    retrieved_chunks: List[str] = []
    citations: List[str] = []

    if ensure_vector_index(VECTOR_STORE_DIR, VECTOR_COLLECTION_NAME):
        try:
            retrieved_chunks, citations = query_vector_store(user_prompt, VECTOR_COLLECTION_NAME, VECTOR_STORE_DIR)
            if not retrieved_chunks:
                return (
                    "System Error: No relevant guidance was retrieved from the vector store. "
                    "Please verify the vector index has been created and that the Foundry embedding model is supported.",
                    []
                )
        except Exception as e:
            return (f"System Error: Vector store retrieval failed: {e}", [])
    else:
        return (
            "System Error: The vector index is missing or could not be created. Please run ingestion again.",
            []
        )

    context = "\n\n".join(retrieved_chunks)

    prompt_text = (
        "Answer using only the context below. "
        "If the answer cannot be found in the context, reply exactly: I couldn't find a clear answer in approved guidance.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{user_prompt}"
    )

    def _extract_response_text(api_response: dict) -> str:
        def _extract_from_content(content):
            texts = []
            if isinstance(content, str):
                return [content]
            if isinstance(content, dict):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return [content["text"]]
                if isinstance(content.get("text"), str):
                    return [content["text"]]
                if isinstance(content.get("content"), str):
                    return [content["content"]]
                if isinstance(content.get("content"), list):
                    for item in content["content"]:
                        texts.extend(_extract_from_content(item))
                return texts
            if isinstance(content, list):
                for item in content:
                    texts.extend(_extract_from_content(item))
                return texts
            return texts

        def _collect_candidate_texts(value):
            candidates = []
            if isinstance(value, str):
                stripped = value.strip()
                if len(stripped) > 30 and " " in stripped:
                    candidates.append(stripped)
                return candidates
            if isinstance(value, dict):
                for key, subvalue in value.items():
                    if key in {"text", "output_text", "content"}:
                        candidates.extend(_collect_candidate_texts(subvalue))
                return candidates
            if isinstance(value, list):
                for item in value:
                    candidates.extend(_collect_candidate_texts(item))
                return candidates
            return candidates

        if not isinstance(api_response, dict):
            raise ValueError("API response is not a dict")

        if "choices" in api_response:
            choice = api_response["choices"][0]
            if isinstance(choice, dict) and "message" in choice:
                return choice["message"].get("content", "").strip()

        if "output_text" in api_response and isinstance(api_response["output_text"], str):
            return api_response["output_text"].strip()

        output = api_response.get("output")
        if isinstance(output, list):
            texts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    texts.extend(_extract_from_content(item.get("content")))
                elif isinstance(item.get("text"), str):
                    texts.append(item["text"])
            filtered = [t.strip() for t in texts if t and t.strip()]
            if filtered:
                return "\n".join(filtered).strip()

        if "text" in api_response and isinstance(api_response["text"], dict):
            nested = api_response["text"].get("text")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

        candidates = _collect_candidate_texts(api_response)
        if candidates:
            return candidates[0]

        print("DEBUG: Unable to extract text from Foundry response:", json.dumps(api_response, indent=2)[:3000])
        raise KeyError("Unable to extract text from Foundry response")

    try:
        response = foundry_client.chat_completions_create(
            model=foundry_chat_model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=MAX_FOUNDARY_OUTPUT_TOKENS,
            temperature=0.0,
        )

        if (
            isinstance(response, dict)
            and response.get("status") == "incomplete"
            and response.get("incomplete_details", {}).get("reason") == "max_output_tokens"
        ):
            print("DEBUG: Foundry response was incomplete due to max_output_tokens; retrying with larger output budget.")
            response = foundry_client.chat_completions_create(
                model=foundry_chat_model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=MAX_FOUNDARY_RETRY_OUTPUT_TOKENS,
                temperature=0.0,
            )

        ai_answer = _extract_response_text(response)
        if "couldn't find a clear answer" in ai_answer.lower():
            return "I couldn't find a clear answer in approved guidance.", citations
        return ai_answer, citations

    except Exception as e:
        return f"System Error: {str(e)}", []
