import os
import re
import traceback
from typing import Dict, List, Optional, Tuple

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


def get_agent_client(
    project_endpoint: Optional[str] = None,
    agent_name: Optional[str] = None,
    allow_preview: bool = True,
):
    """Return an OpenAI-compatible client pointed at a deployed Foundry agent."""
    resolved_endpoint = project_endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    resolved_agent_name = agent_name or os.getenv("FOUNDRY_AGENT_NAME", "imac-agent-new")

    print(f"[DEBUG] Foundry agent setup: endpoint={'set' if resolved_endpoint else 'unset'}, agent_name={resolved_agent_name}")

    if not resolved_endpoint:
        print("[DEBUG] Missing AZURE_AI_PROJECT_ENDPOINT environment variable")
        raise RuntimeError(
            "AZURE_AI_PROJECT_ENDPOINT is required. It should point to your Azure AI Project endpoint."
        )

    print(f"[DEBUG] Creating AIProjectClient with endpoint: {resolved_endpoint}")
    project_client = AIProjectClient(
        endpoint=resolved_endpoint,
        credential=DefaultAzureCredential(),
        allow_preview=allow_preview,
    )

    print(f"[DEBUG] Requesting OpenAI-compatible client for agent: {resolved_agent_name}")
    return project_client.get_openai_client(agent_name=resolved_agent_name)


def build_messages(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict]:
    """
    Convert stored chat history + current prompt into the typed input format
    expected by the Responses API.

    The Responses API requires each item in `input` to be a typed message dict:
        {"role": "user"|"assistant", "content": [{"type": "input_text", "text": "..."}]}
    Plain {"role": ..., "content": "..."} dicts are rejected with a 400.
    System messages must NOT go here – they belong in the top-level `instructions`
    parameter of responses.create().
    """
    messages: List[Dict] = []

    for message in conversation_history or []:
        role = message.get("role")
        content = message.get("content")
        # Skip system turns – handled via instructions= parameter
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        stripped = content.strip()
        if not stripped:
            continue
        content_type = "input_text" if role == "user" else "output_text"
        messages.append({
            "role": role,
            "content": [{"type": content_type, "text": stripped}],
        })

    if prompt:
        stripped_prompt = prompt.strip()
        if stripped_prompt:
            # Avoid duplicating the last user turn if history already ends with it
            last = messages[-1] if messages else None
            last_text = (
                last["content"][0]["text"]
                if last and last.get("role") == "user" and last.get("content")
                else None
            )
            if last_text != stripped_prompt:
                messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": stripped_prompt}],
                })

    return messages


def _debug_dump_output_items(output_items: list) -> None:
    """
    Deep-print every output item so we can see exactly what the Responses API
    returns (item types, content block types, annotation types, tool results).
    This is intentionally verbose – it is the primary diagnostic for citation issues.
    """
    print(f"[DEBUG] === Response output dump ({len(output_items)} items) ===")
    for i, item in enumerate(output_items):
        item_type = getattr(item, "type", "<no type>")
        print(f"[DEBUG]   item[{i}] type={item_type}")

        # --- message items: walk content blocks and their annotations ---
        if item_type == "message":
            for j, block in enumerate(getattr(item, "content", []) or []):
                block_type = getattr(block, "type", "<no type>")
                text_preview = repr(getattr(block, "text", "")[:120]) if hasattr(block, "text") else ""
                annotations  = getattr(block, "annotations", None) or []
                print(f"[DEBUG]     content[{j}] type={block_type} text={text_preview} annotations={len(annotations)}")
                for k, ann in enumerate(annotations):
                    ann_type = getattr(ann, "type", "<no type>")
                    ann_text = getattr(ann, "text", "")
                    print(f"[DEBUG]       annotation[{k}] type={ann_type} marker={repr(ann_text)}")
                    if ann_type == "file_citation":
                        print(f"[DEBUG]         file_citation: filename={getattr(ann, 'filename', None)} file_id={getattr(ann, 'file_id', None)}")
                    elif ann_type == "url_citation":
                        uc = getattr(ann, "url_citation", None)
                        print(f"[DEBUG]         url_citation: url={getattr(uc, 'url', None)} title={getattr(uc, 'title', None)}")

        # --- file_search_call: list result filenames as fallback citations ---
        elif item_type == "file_search_call":
            results = getattr(item, "results", None) or []
            status  = getattr(item, "status", "<no status>")
            print(f"[DEBUG]     file_search_call status={status} results={len(results)}")
            for r in results:
                print(f"[DEBUG]       result: filename={getattr(r, 'filename', None)} file_id={getattr(r, 'file_id', None)} score={getattr(r, 'score', None)}")

        # --- web_search_call ---
        elif item_type == "web_search_call":
            status = getattr(item, "status", "<no status>")
            print(f"[DEBUG]     web_search_call status={status}")

        # --- anything else ---
        else:
            print(f"[DEBUG]     (raw) {repr(item)[:200]}")

    print("[DEBUG] === end output dump ===")


def extract_agent_text_and_citations(response) -> Tuple[str, List[Dict]]:
    """
    Extract the final answer text and structured citations from the agent response.

    Returns (answer_text, citations) where citations is a list of dicts:
        {"filename": str, "file_id": str, "index": int, "ref_num": int}

    The index field (character offset) is used to insert [N] superscript markers
    into the text at the exact position each citation was made.  Multiple
    annotations pointing to the same file get the same ref_num so the reference
    list stays compact.

    NOTE: response.output_text is a plain-string convenience attribute that
    carries NO annotation objects – we must always walk response.output instead.
    """
    answer_text: str = ""
    # Map filename -> ref_num for deduplication
    filename_to_ref: Dict[str, int] = {}
    # Fallback pool: filenames surfaced by file_search_call result items
    file_search_fallback: List[str] = []

    output_items = getattr(response, "output", []) or []
    _debug_dump_output_items(output_items)

    for item in output_items:
        item_type = getattr(item, "type", None)

        if item_type == "file_search_call":
            for result in getattr(item, "results", []) or []:
                filename = getattr(result, "filename", None)
                if filename and filename not in file_search_fallback:
                    file_search_fallback.append(filename)
            continue

        if item_type != "message":
            continue

        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) != "output_text":
                continue

            raw_text: str = getattr(block, "text", "") or ""
            annotations   = getattr(block, "annotations", None) or []

            for annotation in annotations:
                ann_type = getattr(annotation, "type", "")
                marker   = getattr(annotation, "text", "") or ""

                if ann_type == "file_citation":
                    filename = (
                        getattr(annotation, "filename", None) or
                        getattr(annotation, "title",    None) or
                        getattr(annotation, "name",     None)
                    )
                    if not filename:
                        fid = getattr(annotation, "file_id", None)
                        filename = f"File: {fid}" if fid else "Source"

                    # Assign a ref number the first time we see this filename
                    if filename not in filename_to_ref:
                        filename_to_ref[filename] = len(filename_to_ref) + 1
                    ref_num = filename_to_ref[filename]

                elif ann_type == "url_citation":
                    uc    = getattr(annotation, "url_citation", None)
                    url   = getattr(uc, "url",   None) if uc else None
                    title = getattr(uc, "title", None) if uc else None
                    label = title or url or "Source"
                    if label not in filename_to_ref:
                        filename_to_ref[label] = len(filename_to_ref) + 1

            # Strip any leftover 【n:m†…】 markers that had no annotation object
            raw_text = re.sub(r"【\d+:\d+†[^】]*】", "", raw_text)

            if raw_text.strip():
                answer_text += ("\n" if answer_text else "") + raw_text.strip()

    if not answer_text:
        fallback_text = getattr(response, "output_text", None)
        if isinstance(fallback_text, str) and fallback_text.strip():
            print("[DEBUG] extract: fell back to response.output_text (no annotations available)")
            answer_text = fallback_text.strip()
        else:
            raise RuntimeError("Foundry agent returned no usable text content.")

    # ------------------------------------------------------------------
    # Build the structured citations list ordered by ref_num
    # ------------------------------------------------------------------
    citations: List[Dict] = []
    if filename_to_ref:
        for filename, ref_num in sorted(filename_to_ref.items(), key=lambda x: x[1]):
            citations.append({"filename": filename, "ref_num": ref_num})
    elif file_search_fallback:
        print(f"[DEBUG] extract: using {len(file_search_fallback)} file_search_call filename(s) as fallback citations")
        for i, filename in enumerate(file_search_fallback, 1):
            citations.append({"filename": filename, "ref_num": i})

    print(f"[DEBUG] extract: answer_len={len(answer_text)} citations={citations}")
    return answer_text.strip(), citations


def ask_foundry_agent(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    project_endpoint: Optional[str] = None,
    agent_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Ask a deployed Foundry agent and return (answer_text, citations)."""
    print("[DEBUG] ask_foundry_agent starting")
    agent_client = get_agent_client(
        project_endpoint=project_endpoint,
        agent_name=agent_name,
    )

    resolved_model = model or os.getenv("FOUNDRY_AGENT_MODEL", "gpt-5-mini")
    resolved_max_tokens = max_tokens or int(os.getenv("FOUNDRY_AGENT_MAX_TOKENS", "2000"))
    prepared_messages = build_messages(prompt, conversation_history)

    # `instructions=` is rejected when an agent is specified – the agent owns
    # its system prompt in Azure AI Foundry and the API forbids overriding it.
    # Instead, append a citation directive to the final user message so the
    # model is reminded to use FileSearch and cite sources on every turn.
    CITATION_SUFFIX = (
        "\n\n[IMPORTANT: Search your indexed guidance documents before answering. "
        "After each factual claim or bullet point, add the chapter and section in parentheses, "
        "for example: (Chapter 4, Section 4.2) or (Appendix B). "
        "Do not use numbered footnotes or superscripts — write the reference inline in the sentence. "
        "At the end of your response list the source filenames you used.]"
    )
    if prepared_messages and prepared_messages[-1].get("role") == "user":
        last = prepared_messages[-1]
        last["content"][0]["text"] = last["content"][0]["text"] + CITATION_SUFFIX

    print(f"[DEBUG] Agent request: model={resolved_model}, max_output_tokens={resolved_max_tokens}, message_count={len(prepared_messages)}")
    print(f"[DEBUG] Agent messages preview: {prepared_messages[:3]}")

    try:
        response = agent_client.responses.create(
            model=resolved_model,
            input=prepared_messages,
            max_output_tokens=resolved_max_tokens,
        )
    except Exception:
        print("[DEBUG] Foundry agent responses API call failed")
        traceback.print_exc()
        raise

    status = getattr(response, "status", None)
    print(f"[DEBUG] Agent response object type: {type(response).__name__}")
    print(f"[DEBUG] Agent response status: {status}")
    print(f"[DEBUG] Agent response output item count: {len(getattr(response, 'output', []) or [])}")

    # "incomplete" means the response was cut off by max_output_tokens.
    # We still attempt to extract whatever text and citations came back, but
    # log a warning so the token budget can be raised if answers are truncated.
    if status == "incomplete":
        incomplete_reason = getattr(response, "incomplete_details", None)
        print(f"[WARNING] Agent response is INCOMPLETE (truncated). "
              f"Consider raising FOUNDRY_AGENT_MAX_TOKENS. Details: {incomplete_reason}")

    answer, citations = extract_agent_text_and_citations(response)
    if not answer:
        print("[DEBUG] Foundry agent returned empty content")
        raise RuntimeError("Foundry agent returned an empty response.")

    print(f"[DEBUG] Foundry agent returned a response with {len(citations)} citation(s)")
    return answer, citations


def streamlit_agent_response(
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    project_endpoint: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Convenience wrapper for a Streamlit frontend.

    Returns ``(answer, citations)`` so the frontend can render both the
    response text and any source citations produced by the agent.
    """
    print("[DEBUG] streamlit_agent_response invoked")
    try:
        answer, citations = ask_foundry_agent(
            prompt=prompt,
            conversation_history=conversation_history,
            project_endpoint=project_endpoint,
            agent_name=agent_name,
        )
        print(f"[DEBUG] streamlit_agent_response succeeded with {len(citations)} citation(s)")
        return answer, citations
    except Exception:
        print("[DEBUG] streamlit_agent_response failed")
        traceback.print_exc()
        raise

