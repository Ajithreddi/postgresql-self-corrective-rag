import streamlit as st
import os
import re
from typing import TypedDict, List

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="PostgreSQL Self-Corrective RAG",
    layout="wide"
)

st.title("📘 PostgreSQL Self-Corrective RAG")
st.caption(
    "A confidence-aware, self-corrective RAG system built with LangGraph, FAISS, and Groq."
)

# -------------------------------------------------
# LOAD ENV / SECRETS
# -------------------------------------------------
load_dotenv()  # works locally; ignored on Hugging Face

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Add it in Hugging Face Secrets or .env for local use.")
    st.stop()

# -------------------------------------------------
# LOAD MODELS (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_system():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.load_local(
        "faiss_pg_10_11",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return llm, vectorstore


llm, vectorstore = load_system()

# -------------------------------------------------
# LANGGRAPH STATE
# -------------------------------------------------
class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    score: float
    retry_count: int


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------
# LANGGRAPH NODES
# -------------------------------------------------
def retrieve(state: RAGState):
    docs = vectorstore.similarity_search(state["question"], k=4)
    return {"docs": docs}


def generate(state: RAGState):
    context = format_docs(state["docs"])

    prompt = f"""
You are a PostgreSQL expert.

Answer ONLY using the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{state["question"]}
"""
    answer = llm.invoke(prompt).content
    return {"answer": answer}


def evaluate(state: RAGState):
    context = format_docs(state["docs"])

    eval_prompt = f"""
You are an evaluator.

Question:
{state["question"]}

Answer:
{state["answer"]}

Context:
{context}

Rate the answer from 0 to 1 based on:
- factual correctness
- completeness
- grounding in context

Respond ONLY with a number.
"""
    response = llm.invoke(eval_prompt).content.strip()
    match = re.search(r"\d*\.?\d+", response)
    score = float(match.group()) if match else 0.0

    return {"score": score}


def refine(state: RAGState):
    refine_prompt = f"""
Rewrite this question to retrieve better PostgreSQL documentation.

Original question:
{state["question"]}
"""
    new_question = llm.invoke(refine_prompt).content.strip()

    return {
        "question": new_question,
        "retry_count": state["retry_count"] + 1
    }


def decide(state: RAGState):
    score = state["score"]
    retries = state["retry_count"]

    if score >= 0.7:
        return "accept"

    if 0.3 <= score < 0.7 and retries < 1:
        return "refine"

    return "reject"

# -------------------------------------------------
# BUILD LANGGRAPH
# -------------------------------------------------
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("evaluate", evaluate)
graph.add_node("refine", refine)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    decide,
    {
        "accept": END,
        "refine": "refine",
        "reject": END
    }
)

graph.add_edge("refine", "retrieve")

rag_graph = graph.compile()

# -------------------------------------------------
# STREAMLIT UI (CONFIDENCE-AWARE)
# -------------------------------------------------
question = st.text_input(
    "Ask a PostgreSQL documentation question",
    placeholder="How does logical replication differ from physical replication?"
)

if question:
    with st.spinner("Thinking..."):
        result = rag_graph.invoke({
            "question": question,
            "retry_count": 0
        })

    score = float(result.get("score", 0.0))
    answer = result.get("answer", "")

    st.markdown("### 📊 Confidence Score")
    st.progress(min(score, 1.0))
    st.write(score)

    if score >= 0.7:
        st.markdown("### 📌 Answer")
        st.write(answer)

    elif 0.3 <= score < 0.7:
        st.warning(
            "⚠️ The answer may be partially supported or weakly grounded in the documentation."
        )
        st.markdown("### 📌 Partial Answer")
        st.write(answer)

    else:
        st.error(
            "❌ This question cannot be answered using the indexed PostgreSQL documentation."
        )