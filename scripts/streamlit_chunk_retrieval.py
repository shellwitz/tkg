"""Streamlit app for chunk retrieval and Cypher agent inspection."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tkg_rag.cypher_agent as cypher_agent
from tkg_rag.answer import generate_answer
from tkg_rag.retrieve import retrieve


st.set_page_config(
    page_title="TKG Retrieval Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .result-card {
        background-color: #1b1f2a;
        color: #e5e7eb;
        padding: 16px;
        border-radius: 10px;
        margin: 12px 0;
        border-left: 4px solid #5a8dee;
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.35);
    }
    .score-badge {
        background-color: #5a8dee;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9em;
    }
    .source-badge {
        background-color: #2b3350;
        color: #dbeafe;
        padding: 4px 8px;
        border-radius: 6px;
        margin: 2px;
        font-size: 0.85em;
    }
    .highlight {
        background-color: #3b2f14;
        color: #fde68a;
        padding: 1px 3px;
        border-radius: 3px;
    }
    .metric-box {
        background-color: #1f2937;
        color: #e5e7eb;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def highlight_query_terms(text: str, query: str) -> str:
    if not query:
        return text
    words = query.lower().split()
    highlighted = text
    import re

    for word in words:
        if len(word) < 3:
            continue
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f'<span class="highlight">{m.group()}</span>',
            highlighted,
        )
    return highlighted


def format_chunk_card(result: Dict[str, object], rank: int, query: str) -> str:
    text = result.get("text", "")
    source_name = result.get("source_name") or "Unknown Source"
    score = float(result.get("score", 0.0))
    chunk_id = result.get("chunk_id") or "?"

    highlighted_text = highlight_query_terms(text, query)
    preview = highlighted_text[:500] + ("..." if len(text) > 500 else "")

    return f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h4 style="margin: 0;">#{rank} - Chunk {chunk_id}</h4>
            <span class="score-badge">Score: {score:.4f}</span>
        </div>
        <p style="margin: 10px 0;">{preview}</p>
        <div style="margin-top: 10px; font-size: 0.9em; color: #555;">
            <strong>Source:</strong> {source_name}
        </div>
    </div>
    """


def format_edge_card(edge: Dict[str, object], rank: int) -> str:
    rel_text = edge.get("relation_text", "")
    score = float(edge.get("edge_score", edge.get("similarity", 0.0)))
    start_date = edge.get("start_date") or ""
    end_date = edge.get("end_date") or ""
    time_str = f"{start_date} → {end_date}".strip(" →")
    source_name = edge.get("source_name") or ""
    target_name = edge.get("target_name") or ""

    return f"""
    <div class="result-card" style="border-left-color: #f59e0b;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h4 style="margin: 0;">#{rank} - Edge {edge.get("rel_id", "?")}</h4>
            <span class="score-badge" style="background-color: #f59e0b;">Score: {score:.4f}</span>
        </div>
        <p style="margin: 10px 0;">{rel_text}</p>
        <div style="margin-top: 10px; font-size: 0.9em; color: #555;">
            <strong>Entities:</strong> {source_name} → {target_name}<br>
            <strong>Time:</strong> {time_str or "N/A"}
        </div>
    </div>
    """


def perform_retrieval(
    query: str,
    use_only_vec_search: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    start_time = time.time()
    payload = retrieve(query, use_only_vec_search=use_only_vec_search)
    chunks = payload.get("chunks") or []
    edges = payload.get("edges") or []
    context = payload.get("context")

    metadata = {
        "elapsed_time": time.time() - start_time,
        "num_chunks": len(chunks),
        "num_edges": len(edges),
        "context": context,
    }
    return chunks, edges, metadata


def render_retrieval_results(
    result: Dict[str, object],
    meta: Dict[str, object],
    query: str,
    key_prefix: str,
) -> None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks", meta.get("num_chunks", 0))
    with col2:
        st.metric("Edges", meta.get("num_edges", 0))
    with col3:
        st.metric("Time", f"{meta.get('elapsed_time', 0):.3f}s")

    st.markdown("---")
    if result.get("edges"):
        st.subheader("Edges")
        for idx, edge in enumerate(result["edges"], 1):
            st.markdown(format_edge_card(edge, idx), unsafe_allow_html=True)
            with st.expander("Edge Details"):
                st.json(edge)

    if result.get("chunks"):
        st.subheader("Chunks")
        for idx, chunk in enumerate(result["chunks"], 1):
            st.markdown(format_chunk_card(chunk, idx, query), unsafe_allow_html=True)
            with st.expander("Chunk Details"):
                st.write("Chunk ID:", chunk.get("chunk_id"))
                st.write("Source:", chunk.get("source_name"))
                st.write("Score:", chunk.get("score"))
                st.text_area(
                    "Text",
                    value=chunk.get("text", ""),
                    height=140,
                    key=f"{key_prefix}_chunk_text_{idx}",
                    disabled=True,
                )

    if meta.get("context"):
        with st.expander("Fused Context"):
            st.text_area(
                "Context",
                value=meta["context"],
                height=200,
                key=f"{key_prefix}_context",
                disabled=True,
            )


st.title("🔎 TKG RAG Lab")
st.markdown("Inspect RAG and Cypher agent results side-by-side.")

tabs = st.tabs(["RAG Mode", "Cypher Agent"])

with tabs[0]:
    query = st.text_input("Enter your query", key="retrieval_query")
    use_only_vec_search = st.selectbox(
        "Retrieval mode",
        ["Edges + Vector", "Vector Only"],
        key="retrieval_mode",
    )
    rendered_inline = False
    col1, col2 = st.columns([2, 1])
    with col1:
        run_btn = st.button("Run Retrieval", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear", use_container_width=True)
    st.subheader("Answer")
    answer_container = st.empty()
    retrieval_container = st.container()

    if clear_btn:
        st.session_state.pop("retrieval_result", None)
        st.session_state.pop("retrieval_meta", None)
        st.rerun()

    if run_btn and query:
        with answer_container.container():
            st.write("Generating answer...")
        with st.spinner("Running retrieval..."):
            chunks, edges, meta = perform_retrieval(query, use_only_vec_search == "Vector Only")
        st.session_state["retrieval_result"] = {"chunks": chunks, "edges": edges}
        st.session_state["retrieval_meta"] = meta
        st.session_state["rag_answer"] = None
        st.session_state["rag_answer_time"] = None
        with retrieval_container:
            result = st.session_state.get("retrieval_result")
            meta = st.session_state.get("retrieval_meta")
            if result and meta:
                render_retrieval_results(result, meta, query, key_prefix="inline")
                rendered_inline = True

        if meta.get("context"):
            with st.spinner("Generating answer..."):
                start_time = time.time()
                answer = generate_answer(query, meta["context"])
                st.session_state["rag_answer"] = answer
                st.session_state["rag_answer_time"] = time.time() - start_time
            with answer_container.container():
                st.write(answer)
                st.metric("Answer Time", f"{st.session_state['rag_answer_time']:.3f}s")

    result = st.session_state.get("retrieval_result")
    meta = st.session_state.get("retrieval_meta")
    if result and meta and not rendered_inline:
        rag_answer = st.session_state.get("rag_answer")
        rag_answer_time = st.session_state.get("rag_answer_time")
        with answer_container.container():
            if rag_answer:
                st.write(rag_answer)
                if rag_answer_time is not None:
                    st.metric("Answer Time", f"{rag_answer_time:.3f}s")
            else:
                st.write("Generating answer...")
        with retrieval_container:
            render_retrieval_results(result, meta, query, key_prefix="stored")

with tabs[1]:
    st.markdown("Use the Cypher agent and inspect generated queries and results.")
    cypher_query = st.text_input("Question for Cypher agent", key="cypher_query")

    default_uri = os.getenv("NEO4J_URI", "bolt://localhost:7688")
    default_user = os.getenv("TKG_NEO4J_USER", "neo4j")
    default_password = os.getenv("TKG_NEO4J_PASSWORD", "passworty")
    default_container = os.getenv("TKG_NEO4J_CONTAINER", "tkg-neo4j")

    with st.expander("Connection settings"):
        col1, col2 = st.columns(2)
        with col1:
            neo4j_uri = st.text_input("Neo4j URI", value=default_uri)
            neo4j_user = st.text_input("Neo4j user", value=default_user)
        with col2:
            neo4j_password = st.text_input("Neo4j password", value=default_password, type="password")
            container_name = st.text_input("Docker container", value=default_container)

        max_steps = st.slider("Max steps", min_value=1, max_value=20, value=10)
        timeout_s = st.slider("Step timeout (s)", min_value=5, max_value=60, value=15)

    run_agent = st.button("Run Cypher Agent", type="primary", use_container_width=True)
    if run_agent and cypher_query:
        st.session_state["cypher_result"] = None
        st.session_state["cypher_live_rendered"] = False
        steps_container = st.empty()
        answer_container = st.empty()
        last_result = None
        start_time = time.time()
        stream_fn = getattr(cypher_agent, "run_cypher_agent_stream", None)
        if stream_fn is None:
            st.error("Streaming Cypher agent not available. Restart Streamlit after updating the repo.")
        else:
            with st.spinner("Running Cypher agent..."):
                for event in stream_fn(
                    question=cypher_query,
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    container=container_name,
                    timeout_s=timeout_s,
                    max_steps=max_steps,
                ):
                    if event.get("event") == "step":
                        steps = (last_result or {}).get("steps", [])
                        steps = steps + [{"cypher": event.get("cypher"), "rows": event.get("rows", [])}]
                        last_result = {
                            "steps": steps,
                            "cypher": event.get("cypher", ""),
                            "rows": event.get("rows", []),
                            "answer": "",
                        }
                        with steps_container.container():
                            st.subheader("Cypher Steps (live)")
                            for idx, step in enumerate(steps, 1):
                                with st.expander(f"Step {idx}", expanded=(idx == len(steps))):
                                    st.code(step.get("cypher", ""), language="cypher")
                                    st.json(step.get("rows", []))
                    elif event.get("event") == "final":
                        elapsed = time.time() - start_time
                        last_result = {
                            "answer": event.get("answer", ""),
                            "cypher": event.get("cypher", ""),
                            "rows": event.get("rows", []),
                            "steps": event.get("steps", []),
                            "elapsed_time": elapsed,
                        }
                        with answer_container.container():
                            st.subheader("Answer")
                            st.write(last_result.get("answer", ""))
                        st.session_state["cypher_live_rendered"] = True

        st.session_state["cypher_result"] = last_result or cypher_agent.run_cypher_agent(
            question=cypher_query,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            container=container_name,
            timeout_s=timeout_s,
            max_steps=max_steps,
        )

    cypher_result = st.session_state.get("cypher_result")
    live_rendered = st.session_state.get("cypher_live_rendered", False)
    if cypher_result:
        st.markdown("---")
        if not live_rendered:
            if cypher_result.get("answer"):
                st.subheader("Answer")
                st.write(cypher_result.get("answer", ""))
        if cypher_result.get("elapsed_time") is not None:
            st.metric("Agent Time", f"{cypher_result.get('elapsed_time', 0):.3f}s")

            steps = cypher_result.get("steps", [])
            if steps:
                st.subheader("Cypher Steps")
                for idx, step in enumerate(steps, 1):
                    with st.expander(f"Step {idx}"):
                        st.code(step.get("cypher", ""), language="cypher")
                        rows = step.get("rows", [])
                        st.json(rows)
            else:
                st.info("No Cypher queries were executed.")

        st.subheader("Last Query")
        st.code(cypher_result.get("cypher", ""), language="cypher")

        st.subheader("Last Result")
        st.json(cypher_result.get("rows", []))

        st.markdown("---")
        if st.button("Export Agent Trace"):
            export_payload = {
                "question": cypher_query,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "answer": cypher_result.get("answer"),
                "steps": cypher_result.get("steps"),
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(export_payload, indent=2),
                file_name=f"cypher_agent_trace_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
