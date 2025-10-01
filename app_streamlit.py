import streamlit as st
from src.retriever import retrieve_docs
from src.prompts import build_prompt
from src.llm_client import generate_answer
from src.guardrails import check_safe
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(page_title='RAG Chatbot', layout='centered')
st.title('RAG Chatbot')

if 'history' not in st.session_state:
    st.session_state['history'] = []


query = st.text_input('Ask a question about the documents', '')

if st.button('Send') and query:
    # 1. Guardrail check
    ok, reason = check_safe(query)
    
    if not ok:
        st.error(f'Blocked by guardrail: {reason}')
    else:
        with st.spinner('Retrieving relevant documents...'):
            # Retrieve relevant documents
            docs = retrieve_docs(query, k=4)
            
            if not docs:
                # If no documents found, return fallback answer
                ans = "I don't know. No relevant documents found in the knowledge base."
            else:
                # Build prompt with retrieved docs
                prompt = build_prompt(query, docs)
                
                # Generate answer from LLM
                with st.spinner('Generating answer...'):
                    try:
                        ans = generate_answer(prompt)
                    except Exception as e:
                        st.error(f'LLM generation failed: {e}')
                        ans = "Error generating response"

        # Append to session history
        st.session_state.history.append((query, ans))

# Display conversation history
st.header("Conversation History")
for q, a in reversed(st.session_state.history[-10:]):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.write('---')
