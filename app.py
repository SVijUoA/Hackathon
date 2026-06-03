import traceback

import streamlit as st
from foundry_agent_bridge import streamlit_agent_response
from foundry_rag_engine import generate_response

MAX_HISTORY_MESSAGES = 10


def build_conversation_history():
    history = []
    for message in st.session_state.messages[-MAX_HISTORY_MESSAGES:]:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        history.append({"role": role, "content": content.strip()})
    return history


def _is_refusal_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    normalized = text.strip().lower()
    refusal_phrases = [
        "i'm sorry, but i cannot assist with that request",
        "i'm sorry but i cannot assist with that request",
        "i'm sorry, but i cannot help with that request",
        "i'm sorry but i cannot help with that request",
        "i'm sorry, but i cannot answer that request",
        "i'm sorry but i cannot answer that request",
        "cannot assist with that request",
        "cannot help with that request",
        "cannot answer that request",
    ]
    return any(phrase in normalized for phrase in refusal_phrases)


def render_citations(citations):
    """
    Render a structured citations list from the Foundry agent.
    Each entry is either a dict {"filename": str, "ref_num": int}
    (new structured format) or a plain string (legacy fallback).

    Known filenames are resolved to their canonical public URLs so the
    user sees a clickable link rather than a raw filename.
    """
    # Map agent-returned filenames to their public URLs.
    # Add new entries here whenever a new source file is indexed.
    FILENAME_TO_URL = {
        "immunisation-handbook-2026-v2.pdf": (
            "https://static.info.content.health.nz/docs/health-pros/topics/"
            "immunisations/immunisation-handbook-2026-v2.pdf"
        ),
        "vaccines.json": "https://immune.org.nz/vaccines-and-diseases/vaccines",
    }

    if not citations:
        return
    with st.expander("📚 View Source Citations"):
        for citation in citations:
            if isinstance(citation, dict):
                ref_num  = citation.get("ref_num", "")
                filename = citation.get("filename", "Source")
                url = FILENAME_TO_URL.get(filename)
                if url:
                    st.markdown(f"**[{ref_num}]** [{url}]({url})")
                else:
                    st.markdown(f"**[{ref_num}]** `{filename}`")
            else:
                # Plain string citation (local RAG backend). Check if any known
                # filename appears in the path and resolve to its public URL.
                resolved_url = None
                for known_name, known_url in FILENAME_TO_URL.items():
                    if known_name in citation:
                        resolved_url = known_url
                        break
                if resolved_url:
                    st.markdown(f"- [{resolved_url}]({resolved_url})")
                else:
                    st.markdown(f"- {citation}")


def generate_app_response(prompt, conversation_history=None):
    try:
        # Unpack both text answer and citations from the deployed agent
        answer, citations = streamlit_agent_response(
            prompt=prompt,
            conversation_history=conversation_history,
        )
        if _is_refusal_text(answer):
            print("[DEBUG] Deployed agent returned refusal text; falling back to local backend")
            raise RuntimeError("Deployed agent refusal")
        return answer, citations, "deployed agent"
    except Exception:
        print("[DEBUG] Deployed agent failed or refused; falling back to local backend")
        traceback.print_exc()
        answer, citations = generate_response(prompt, conversation_history=conversation_history)
        if _is_refusal_text(answer):
            print("[DEBUG] Local fallback returned refusal text; replacing with safe guidance message")
            answer = "I couldn't find a clear answer in approved guidance."
        return answer, citations, "existing local backend"


# Initialize session state early
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1. PAGE CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="IMAC Advisor Agent", page_icon="🛡️", layout="centered")

# --- SIDEBAR WITH LINKS AND FUNCTIONS ---
with st.sidebar:
    st.header("🛡️ IMAC Resources")
    st.markdown("""
    **Official Links:**
    - [IMAC Website](https://immune.org.nz)
    - [NZ Immunisation Handbook](https://www.health.govt.nz/publication/nz-immunisation-handbook-2023)
    - [Vaccine Timetable](https://immune.org.nz/immunisation/programmes/national-immunisation-schedule)
    - [Clinical Guidelines](https://www.tewhatuora.govt.nz/for-health-professionals/clinical-guidance/immunisation-handbook)
    """)
    
    st.divider()
    st.subheader("Quick Actions")
    if st.button("📋 Vaccine Schedule"):
        st.session_state.messages.append({"role": "user", "content": "What is the standard vaccine schedule for children?"})
    if st.button("💉 Catch-up Vaccines"):
        st.session_state.messages.append({"role": "user", "content": "How do I handle catch-up vaccinations?"})
    if st.button("⚠️ Contraindications"):
        st.session_state.messages.append({"role": "user", "content": "What are the contraindications for vaccines?"})
    
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
        st.divider()
    st.subheader("Audit")
    if st.session_state.messages:
        # Create a simple text log of the chat
        chat_log = "IMAC AI Advisor - Audit Log\n\n"
        for msg in st.session_state.messages:
            role = "Advisor" if msg["role"] == "user" else "AI System"
            chat_log += f"{role}: {msg['content']}\n\n"
        
        st.download_button(
            label="💾 Download Audit Log",
            data=chat_log,
            file_name="clinical_chat_audit.txt",
            mime="text/plain"
        )
    st.divider()
    st.subheader("About")
    st.markdown("""
    This AI assistant provides guidance based on official IMAC guidelines.
    Always consult healthcare professionals for clinical decisions.
    """)

st.title("🛡️ IMAC Immunisation Advisor Agent")

# SECURITY: THREAT 09 - Overreliance on AI
st.markdown("""
**Clinical Safety Notice:** This tool is an AI assistant designed to support, not replace, clinical judgment. 
Always verify information against official IMAC guidelines.
""")
st.divider()

# --- 2. RENDER PAST CONVERSATION ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            render_citations(message["citations"])

# --- 3. USER INPUT & CHAT LOGIC ---
# SECURITY: THREAT 04 - MODEL DoS (Added max_chars=1000 limit)
if prompt := st.chat_input("Ask a clinical question regarding immunisation...", max_chars=1000):
    
    # Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and Display Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Querying the deployed IMAC agent..."):
            ai_answer, citations, response_source = generate_app_response(
                prompt,
                conversation_history=build_conversation_history(),
            )
        
        # Render the text answer FIRST
        message_placeholder.markdown(ai_answer)
        
        # Safety / security banners (independent of citations)
        if ai_answer == "I couldn't find a clear answer in approved guidance.":
            st.error("🛑 Clinical Safety Protocol Engaged")
            st.warning("Low Confidence: Information not found in the indexed guidance. Please refer to senior clinical staff.")
        elif "Security Alert" in ai_answer:
            st.error("🛑 Security Protocol Engaged: Query Rejected.")

        # Citations – always shown when present, regardless of safety banners
        if citations:
            render_citations(citations)

        # Feedback – shown for every normal (non-safety, non-security) response
        if ai_answer not in ("I couldn't find a clear answer in approved guidance.",) and "Security Alert" not in ai_answer:
            st.write("Was this response helpful?")
            feedback = st.feedback("thumbs")
            if feedback is not None:
                st.toast("Thank you for your feedback! This helps improve our clinical agent.")

    # Save Assistant Response to Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_answer,
        "citations": citations,
        "source": response_source,
    })

# --- 4. HANDLE QUICK ACTIONS (Robust version handling multiple clicks) ---
if st.session_state.messages:
    messages_needing_responses = []
    # Find all user messages that do not have a corresponding assistant response
    for i in range(len(st.session_state.messages)):
        if st.session_state.messages[i]["role"] == "user":
            has_response = i + 1 < len(st.session_state.messages) and st.session_state.messages[i + 1]["role"] == "assistant"
            if not has_response:
                messages_needing_responses.append(i)
    
    # Process any pending quick actions
    for idx in messages_needing_responses:
        user_message = st.session_state.messages[idx]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Querying the deployed IMAC agent..."):
                ai_answer, citations, response_source = generate_app_response(
                    user_message["content"],
                    conversation_history=build_conversation_history(),
                )
            
            message_placeholder.markdown(ai_answer)
            
            # Safety / security banners
            if ai_answer == "I couldn't find a clear answer in approved guidance.":
                st.error("🛑 Clinical Safety Protocol Engaged")
                st.warning("Low Confidence: Information not found in the indexed guidance. Please refer to senior clinical staff.")
            elif "Security Alert" in ai_answer:
                st.error("🛑 Security Protocol Engaged: Query Rejected.")

            # Citations – always shown when present
            if citations:
                render_citations(citations)

            # Feedback – shown for every normal response
            if ai_answer not in ("I couldn't find a clear answer in approved guidance.",) and "Security Alert" not in ai_answer:
                st.write("Was this response helpful?")
                feedback = st.feedback("thumbs")
                if feedback is not None:
                    st.toast("Thank you for your feedback! This helps improve our clinical agent.")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_answer,
            "citations": citations,
            "source": response_source,
        })