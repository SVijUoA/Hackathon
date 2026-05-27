import streamlit as st
from agent_service import setup_agent, chat_with_advisor

# Cache agent creation so it doesn't create a new version on every UI interaction
@st.cache_resource
def init_agent():
    return setup_agent()

init_agent()

# Initialize session state early
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

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
        st.session_state.conversation_id = None
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

# --- 3. USER INPUT & CHAT LOGIC ---
# SECURITY: THREAT 04 - MODEL DoS (Added max_chars=1000 limit)
if prompt := st.chat_input("Ask a clinical question regarding immunisation...", max_chars=1000):
    
    # Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and Display Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Agent is searching official IMAC guidelines..."):
            convo_id, ai_answer = chat_with_advisor(prompt, st.session_state.conversation_id)
            st.session_state.conversation_id = convo_id
        
        # Render the text answer FIRST
        message_placeholder.markdown(ai_answer)
        
        # Then handle safety warnings
        if "I cannot find the answer in the official guidance" in ai_answer:
            st.error("🛑 Clinical Safety Protocol Engaged")
            st.warning("Low Confidence: Information not found in the indexed guidance. Please refer to senior clinical staff.")
        else:
            # Feedback Loop
            st.write("Was this response helpful?")
            feedback = st.feedback("thumbs")
            if feedback is not None:
                st.toast("Thank you for your feedback! This helps improve our clinical agent.")

    # Save Assistant Response to Session State
    st.session_state.messages.append({
        "role": "assistant", 
        "content": ai_answer
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
            
            with st.spinner("Agent is searching official IMAC guidelines..."):
                convo_id, ai_answer = chat_with_advisor(user_message["content"], st.session_state.conversation_id)
                st.session_state.conversation_id = convo_id
            
            message_placeholder.markdown(ai_answer)
            
            if "I cannot find the answer in the official guidance" in ai_answer:
                st.error("🛑 Clinical Safety Protocol Engaged")
                st.warning("Low Confidence: Information not found in the indexed guidance. Please refer to senior clinical staff.")
            else:
                st.write("Was this response helpful?")
                feedback = st.feedback("thumbs")
                if feedback is not None:
                    st.toast("Thank you for your feedback! This helps improve our clinical agent.")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_answer
        })