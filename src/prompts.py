# In your prompts.py - FIX THE SYNTAX
PROMPT_TEMPLATE = """
You are an assistant that answers user questions using ONLY the provided context below.
CRITICAL INSTRUCTION: If the answer is NOT present in the provided context, you MUST reply with the exact phrase "I don't know." Do not use any external knowledge.

Context:
{context}

Question: {question}

Answer concisely and truthfully, citing context if possible.
"""

def build_prompt(query, docs):
    """Build a prompt using retrieved documents."""
    if not docs:
        return "I don't know..."  # ← Only returns if no docs
    else:
    # This executes when docs ARE available
        context = ""
        for i, doc in enumerate(docs):
            context += f"[Document {i+1}]: {doc.page_content}\n"
    return PROMPT_TEMPLATE.format(context=context, question=query)