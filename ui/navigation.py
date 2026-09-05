"""ui / navigation for ManGo or Stay."""

from config import PAGES
from datetime import datetime
import streamlit as st

def generate_batch_id() -> str:
    """Generate a readable batch identifier for a new assessment session."""
    return f"BATCH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def go_to(page_name: str):
    st.session_state["_pending_nav"] = page_name
    st.rerun()

def initialise_session():
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None

    if "current_image_hash" not in st.session_state:
        st.session_state.current_image_hash = None

    if "current_saved_id" not in st.session_state:
        st.session_state.current_saved_id = None

    if "selected_assessment_id" not in st.session_state:
        st.session_state.selected_assessment_id = None

    if "batch_id_input" not in st.session_state:
        st.session_state.batch_id_input = generate_batch_id()

    if "_pending_nav" in st.session_state:
        st.session_state.page = st.session_state.pop("_pending_nav")


def render_navigation(history_df):
    st.markdown(
        f"""
        <div class="top-appbar">
            <div>
                <div class="top-brand-name">ManGo or Stay</div>
                <div class="top-brand-caption">Harumanis mango ripeness and surface quality assessment</div>
            </div>
            <div class="top-record-count">{len(history_df):,} saved assessment(s)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_col, action_col = st.columns([5.6, 1.4])

    with nav_col:
        page = st.radio(
            "Navigate",
            PAGES,
            horizontal=True,
            label_visibility="collapsed",
            key="page",
        )

    with action_col:
        if page != "Assess Mango":
            if st.button("New assessment", use_container_width=True, type="primary", key="top_new_assessment"):
                st.session_state.current_analysis = None
                st.session_state.current_image_hash = None
                st.session_state.current_saved_id = None
                st.session_state.batch_id_input = generate_batch_id()
                go_to("Assess Mango")
        else:
            st.caption("Harumanis mango only")

    st.markdown('<div class="top-nav-divider"></div>', unsafe_allow_html=True)

    return page

