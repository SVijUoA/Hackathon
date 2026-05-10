import streamlit as st
from rag_engine import generate_response # <--- Connects to your back-end engine

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
    st.subheader("About")
    st.markdown("""
    This AI assistant provides guidance based on official IMAC guidelines.
    Always consult healthcare professionals for clinical decisions.
    """)

st.title("🛡️ IMAC Immunisation Advisor Agent")

# Clinical Safety Guardrail: Explicit disclaimer on the UI
st.markdown("""
**Clinical Safety Notice:** This tool is an AI assistant designed to support, not replace, clinical judgment. 
Always verify information against official IMAC guidelines.
""")
st.divider()

# --- 2. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. RENDER PAST CONVERSATION ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("View Source Citations"):
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")

# --- 4. USER INPUT & CHAT LOGIC ---
if prompt := st.chat_input("Ask a clinical question regarding immunisation..."):
    
    # 4a. Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 4b. Generate and Display Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching official IMAC guidelines..."):
            # --- THIS IS WHERE THE FRONT-END TALKS TO THE BACK-END ---
            ai_answer, citations = generate_response(prompt)
        
        # Render the text answer FIRST
        message_placeholder.markdown(ai_answer)
        
        # Then handle citations or warnings
        if ai_answer == "I couldn't find a clear answer in approved guidance.":
            st.error("🛑 Clinical Safety Protocol Engaged")
            st.warning("Low Confidence: Information not found in the indexed guidance. Please refer to senior clinical staff.")
        elif citations:
            with st.expander("View Source Citations"):
                for citation in citations:
                    st.markdown(f"- {citation}")
            
            # Feedback Loop
            st.write("Was this response helpful?")
            feedback = st.feedback("thumbs")
            if feedback is not None:
                st.toast("Thank you for your feedback! This helps improve our clinical agent.")

    # 4c. Save Assistant Response to Session State
    st.session_state.messages.append({
        "role": "assistant", 
        "content": ai_answer,
        "citations": citations
    })

# --- 5. HANDLE QUICK ACTIONS ---
# Find all user messages that need responses (for handling multiple quick actions at once)
if st.session_state.messages:
    messages_needing_responses = []
    for i in range(len(st.session_state.messages)):
        if st.session_state.messages[i]["role"] == "user":
            has_response = i + 1 < len(st.session_state.messages) and st.session_state.messages[i + 1]["role"] == "assistant"
            if not has_response:
                messages_needing_responses.append(i)
    
    # Generate responses for all pending messages (shows all quick actions at once)
    for idx in messages_needing_responses:
        user_message = st.session_state.messages[idx]
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Searching official IMAC guidelines..."):
                ai_answer, citations = generate_response(user_message["content"])
            
            message_placeholder.markdown(ai_answer)
            
            if ai_answer == "I couldn't find a clear answer in approved guidance.":
                st.error("🛑 Clinical Safety Protocol Engaged")
                st.warning("Low Confidence: Information not found in the indexed guidance. Please refer to senior clinical staff.")
            elif citations:
                with st.expander("View Source Citations"):
                    for citation in citations:
                        st.markdown(f"- {citation}")
                
                st.write("Was this response helpful?")
                feedback = st.feedback("thumbs")
                if feedback is not None:
                    st.toast("Thank you for your feedback! This helps improve our clinical agent.")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_answer,
            "citations": citations
        })