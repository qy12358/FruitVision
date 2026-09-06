"""Prepare reports on request without rerunning the assessment page."""

import streamlit as st


@st.fragment
def render_pdf_download(report_key, build_pdf, file_name):
    cached = st.session_state.get("prepared_pdf")
    if cached is None or cached[0] != report_key:
        if not st.button("Prepare PDF", key="prepare_pdf", width="stretch"):
            return
        try:
            with st.spinner("Preparing PDF report..."):
                pdf = build_pdf()
            if not pdf:
                raise ValueError("The PDF report could not be generated.")
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
            return
        cached = (report_key, pdf)
        st.session_state.prepared_pdf = cached

    st.download_button(
        "Export PDF",
        data=cached[1],
        file_name=file_name,
        mime="application/pdf",
        on_click="ignore",
        width="stretch",
    )
