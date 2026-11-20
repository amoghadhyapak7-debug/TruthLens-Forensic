"""Text & Document Forensics View for TruthLens Monitor."""
from __future__ import annotations

import requests
import streamlit as st


def render_text_view(api_url: str, all_records: list[dict]):
    st.header("📄 Text & Context Forensics Analyzer")
    st.caption("Document extraction, AI-generated text analysis, claim extraction & linguistic integrity")

    text_records = [r for r in all_records if r.get("file_type") in ("text", "document", "structured_data")]
    if not text_records:
        st.info("No text or document evidence registered yet. Upload .pdf, .docx, .txt, .md, or .json in the **Evidence Ingest** tab first.")
        return

    ev_options = {f"{r.get('evidence_id')} - {r.get('original_filename')}": r for r in text_records}
    selected_label = st.selectbox("Select Text/Document Evidence", options=list(ev_options.keys()), key="text_evidence_select")
    record = ev_options[selected_label]
    evidence_id = record.get("evidence_id")

    st.write(f"**Evidence ID:** `{evidence_id}` &bull; **Filename:** `{record.get('original_filename')}` &bull; **Type:** `{record.get('file_type')}`")

    res_key = f"text_analysis_result_{evidence_id}"
    if res_key not in st.session_state:
        try:
            r = requests.get(f"{api_url}/api/v1/analyzers/text/{evidence_id}", timeout=5)
            if r.status_code == 200:
                st.session_state[res_key] = r.json()
        except Exception:
            pass

    if st.button("🚀 Run Text & Context Analysis", type="primary", key=f"run_txt_{evidence_id}"):
        with st.spinner("Analyzing document text, claims, entities, and AI-generation markers..."):
            try:
                resp = requests.post(f"{api_url}/api/v1/analyzers/text/{evidence_id}", timeout=120)
                if resp.status_code == 200:
                    st.session_state[res_key] = resp.json()
                    st.success("Analysis complete!")
                else:
                    st.error(f"Analysis failed: {resp.text}")
            except Exception as exc:
                st.error(f"Error during text analysis: {exc}")

    result = st.session_state.get(res_key)
    if not result:
        st.info("Click **Run Text & Context Analysis** to parse claims and AI generation markers.")
        return

    ai_data = result.get("ai_detection", {})
    ai_prob = ai_data.get("ai_probability", 0.0)

    st.subheader("1. AI-Generated Text Assessment")
    m1, m2 = st.columns(2)
    with m1:
        if isinstance(ai_prob, (int, float)) and ai_prob >= 0.70:
            st.error(f"🚨 **PROBABLE AI-GENERATED TEXT ({ai_prob:.1%})**")
        elif isinstance(ai_prob, (int, float)) and ai_prob <= 0.30:
            st.success(f"✅ **PROBABLE HUMAN-AUTHORED TEXT ({ai_prob:.1%})**")
        else:
            st.warning(f"⚠️ **INCONCLUSIVE / MIXED AUTHORED TEXT**")
    with m2:
        if isinstance(ai_prob, (int, float)):
            st.progress(min(max(float(ai_prob), 0.0), 1.0))
        st.caption(f"Indicators evaluated: perplexity, burstiness, repetitive phrasing, and syntactic consistency.")

    st.markdown("---")
    st.subheader("2. Extracted Claims & Entities")
    t1, t2, t3 = st.tabs(["Claims & Statements", "Entities Identified", "Document Preview"])

    with t1:
        claims = result.get("claims", [])
        if claims:
            for idx, c in enumerate(claims):
                st.write(f"**Claim {idx+1}:** {c.get('claim_text', c)}")
        else:
            st.write("No distinct factual claims extracted.")

    with t2:
        entities = result.get("entities", {})
        if entities and isinstance(entities, dict):
            st.json(entities)
        elif entities and isinstance(entities, list):
            st.write(entities)
        else:
            st.write("No key named entities cataloged.")

    with t3:
        doc_text = result.get("extracted_text") or result.get("text_preview", "")
        if doc_text:
            st.text_area("Extracted Content", value=str(doc_text)[:5000], height=250, disabled=True)
        else:
            st.caption("No text preview available.")

    with st.expander("Full Text Analytics Payload", expanded=False):
        st.json(result, expanded=False)
