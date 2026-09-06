"""ui / theme for ManGo or Stay."""

from config import BG_LIGHT
from config import GREEN
from config import PRIMARY
from config import PRIMARY_LIGHT
from config import RED
from config import SECONDARY
from config import YELLOW
import streamlit as st

def configure_page():
    st.set_page_config(
        page_title="ManGo or Stay",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

def apply_theme():
    st.markdown(
        f"""
        <style>
        :root {{
            --ink: #1E2924;
            --muted: #6F7773;
            --line: #E0E2DE;
            --surface: #FFFFFF;
            --surface-soft: #F0F2EE;
            --primary: {PRIMARY};
            --primary-soft: #E8EFEB;
        }}

        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: var(--ink);
        }}

        [data-testid="stAppViewContainer"] {{ background: {BG_LIGHT}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1220px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }}

        /* The app uses a top navigation bar rather than Streamlit's side panel. */
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"] {{
            display: none !important;
        }}

        .top-appbar {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 24px;
            padding: 0 0 14px 0;
            margin-bottom: 2px;
        }}
        .top-brand-name {{
            font-size: 25px;
            line-height: 1.08;
            font-weight: 740;
            letter-spacing: -0.04em;
            color: var(--primary);
        }}
        .top-brand-caption {{
            margin-top: 7px;
            font-size: 12px;
            line-height: 1.5;
            color: #69716D;
        }}
        .top-record-count {{
            flex: 0 0 auto;
            padding: 7px 10px;
            border: 1px solid #D8DDD8;
            border-radius: 7px;
            background: #FFFFFF;
            font-size: 10px;
            font-weight: 680;
            color: #56635D;
            white-space: nowrap;
        }}
        .top-nav-divider {{
            height: 1px;
            background: #DDE1DC;
            margin: 4px 0 28px 0;
        }}

        /* Make Streamlit's horizontal radio behave like a quiet product navigation bar. */
        div[role="radiogroup"] {{
            gap: 6px !important;
            flex-wrap: wrap !important;
            align-items: center !important;
        }}
        div[role="radiogroup"] > label {{
            border: 1px solid #DDE1DC !important;
            border-radius: 7px !important;
            background: #FFFFFF !important;
            padding: 7px 11px !important;
            margin: 0 !important;
            transition: background 120ms ease, border-color 120ms ease;
        }}
        div[role="radiogroup"] > label:hover {{
            border-color: #BFC9C2 !important;
            background: #F4F7F4 !important;
        }}
        div[role="radiogroup"] > label:has(input:checked) {{
            border-color: var(--primary) !important;
            background: var(--primary-soft) !important;
        }}
        div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}
        div[role="radiogroup"] p {{
            font-size: 12px !important;
            font-weight: 650 !important;
            color: #33423B !important;
        }}

        .page-kicker {{
            margin-bottom: 7px;
            font-size: 10px;
            font-weight: 720;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--primary);
        }}
        .page-title {{
            margin: 0;
            font-size: clamp(29px, 3vw, 39px);
            line-height: 1.12;
            font-weight: 720;
            letter-spacing: -0.04em;
            color: #17211D;
        }}
        .page-subtitle {{
            max-width: 760px;
            margin: 10px 0 26px 0;
            font-size: 14px;
            line-height: 1.65;
            color: var(--muted);
        }}

        .section-title {{
            font-size: 18px;
            font-weight: 680;
            color: #1B2722;
            margin: 24px 0 12px 0;
            letter-spacing: -0.018em;
        }}

        .metric-card {{
            min-height: 112px;
            background: var(--surface);
            border-radius: 12px;
            padding: 18px 19px;
            border: 1px solid var(--line);
            box-shadow: 0 3px 14px rgba(29,43,37,0.035);
        }}
        .metric-title {{
            font-size: 11px;
            line-height: 1.35;
            color: #737B77;
            font-weight: 650;
            margin-bottom: 11px;
        }}
        .metric-value {{
            font-size: 29px;
            line-height: 1;
            font-weight: 720;
            letter-spacing: -0.038em;
            color: #1B2722;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 10px;
            font-weight: 720;
            color: white;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        .badge-ripe {{ background-color: {GREEN}; }}
        .badge-semi {{ background-color: {YELLOW}; color:#242A27; }}
        .badge-unripe {{ background-color: {SECONDARY}; }}
        .badge-rotten {{ background-color: {RED}; }}
        .badge-A {{ background-color: {GREEN}; }}
        .badge-B {{ background-color: #6F8F69; }}
        .badge-C {{ background-color: {YELLOW}; color:#242A27; }}
        .badge-D {{ background-color: {RED}; }}
        .badge-premium {{ background-color: {GREEN}; }}
        .grading-result {{
            padding: 24px;
            margin-bottom: 24px;
            background: #F2F6F3;
            border: 1px solid #D4DED8;
            border-radius: 12px;
        }}
        .grading-result-label {{
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 12px;
            color: #37413C;
        }}
        .grading-result .badge {{
            font-size: clamp(28px, 4vw, 42px);
            font-weight: 750;
            line-height: 1.2;
            padding: 12px 22px;
            text-transform: none;
            letter-spacing: normal;
        }}
        .badge-grade-1 {{ background-color: #6F8F69; }}
        .badge-grade-2 {{ background-color: {YELLOW}; color:#242A27; }}
        .badge-reject {{ background-color: {RED}; }}
        .badge-unavailable {{ background-color: #777D79; }}

        .fixed-produce {{
            min-height: 74px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            background: #EEF2EE;
            border: 1px solid #D8E0DA;
            border-radius: 10px;
            padding: 12px 15px;
        }}
        .fixed-produce-label {{
            font-size: 10px;
            font-weight: 650;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #78817C;
            margin-bottom: 5px;
        }}
        .fixed-produce-value {{
            font-size: 15px;
            font-weight: 700;
            color: var(--primary);
        }}

        .guide-card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 9px;
        }}
        .guide-number {{
            font-size: 10px;
            font-weight: 750;
            letter-spacing: 0.08em;
            color: var(--primary);
            margin-bottom: 6px;
        }}
        .guide-title {{
            font-size: 13px;
            font-weight: 700;
            color: #25302B;
            margin-bottom: 4px;
        }}
        .guide-copy {{
            font-size: 12px;
            line-height: 1.55;
            color: #6E7772;
        }}

        .notice-box {{
            background: #EFF3EF;
            border: 1px solid #DCE4DE;
            border-left: 3px solid var(--primary);
            border-radius: 8px;
            padding: 13px 15px;
            font-size: 12px;
            line-height: 1.6;
            color: #4F5B55;
            margin: 8px 0 14px 0;
        }}

        .hero-card {{
            background: var(--surface);
            border-radius: 14px;
            padding: 22px 24px;
            border: 1px solid var(--line);
            border-left: 4px solid var(--hero-color, {PRIMARY});
            box-shadow: 0 4px 18px rgba(29,43,37,0.035);
            margin-bottom: 18px;
        }}
        .hero-title {{
            font-size: 10px;
            color: #7A827E;
            font-weight: 720;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 8px;
        }}
        .hero-advice {{
            font-size: 13px;
            line-height: 1.55;
            color: #59635E;
            margin-top: 11px;
        }}

        .empty-state {{
            text-align: center;
            padding: 44px 22px;
            background: #FBFBF9;
            border: 1px dashed #D6DAD5;
            border-radius: 12px;
            color: #7A817D;
        }}
        .empty-state-mark {{
            width: 28px;
            height: 4px;
            border-radius: 999px;
            background: #C7D0CA;
            margin: 0 auto 14px auto;
        }}

        .step-indicator {{
            display: flex;
            align-items: center;
            gap: 7px;
            margin: 4px 0 22px 0;
            flex-wrap: wrap;
        }}
        .step {{
            display: flex;
            align-items: center;
            gap: 7px;
            padding: 6px 11px 6px 7px;
            border-radius: 7px;
            font-size: 11px;
            font-weight: 650;
        }}
        .step-num {{
            width: 20px;
            height: 20px;
            border-radius: 5px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 720;
        }}
        .step-todo {{ background:#ECEDEA; color:#949A96; }}
        .step-todo .step-num {{ background:#DADDD9; color:#777D79; }}
        .step-active {{ background:#E8EFEB; color:{PRIMARY}; }}
        .step-active .step-num {{ background:{PRIMARY}; color:white; }}
        .step-done {{ background:#EEF2EF; color:{PRIMARY}; }}
        .step-done .step-num {{ background:{PRIMARY_LIGHT}; color:white; }}
        .step-connector {{ width:22px; height:1px; background:#D7DAD6; }}

        div.stButton > button,
        div.stDownloadButton > button {{
            min-height: 42px;
            border-radius: 8px;
            font-weight: 650;
            border: 1px solid #D9DDD9;
            box-shadow: none;
        }}

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div {{
            border-radius: 8px !important;
            border-color: #D7DBD7 !important;
            background: #FFFFFF !important;
        }}

        .confidence-gauge {{ width: 100%; background: white; }}
        .confidence-gauge svg {{
            display: block;
            width: 100%;
            max-width: 460px;
            height: auto;
            margin-inline: auto;
        }}

        /* Keep photo previews compact while showing the entire image. */
        [data-testid="stImage"] {{
            width: min(100%, 480px) !important;
            margin-inline: auto;
        }}

        [data-testid="stImage"] img {{
            display: block;
            width: auto !important;
            max-width: 100% !important;
            height: auto !important;
            max-height: min(360px, 45vh);
            object-fit: contain;
            margin-inline: auto;
        }}

        [data-testid="stImageContainer"] {{
            width: 100%;
            max-width: 100%;
            min-width: 0;
        }}

        /* Fit the full photo to the result area without changing image data. */
        .st-key-assessment_original_photo [data-testid="stImage"],
        .st-key-assessment_original_photo [data-testid="stImageContainer"] {{
            width: 100% !important;
            min-width: 0;
        }}

        .st-key-assessment_original_photo [data-testid="stImage"] img {{
            width: 100% !important;
            height: clamp(360px, 65vh, 640px) !important;
            max-height: none;
            object-fit: contain;
            object-position: center;
        }}

        .st-key-assessment_original_photo [data-testid="stImageCaption"] {{
            width: 100%;
            text-align: center;
            white-space: normal;
            word-break: normal;
        }}

        [data-testid="stCameraInput"] {{
            max-width: 480px;
            margin-inline: auto;
        }}

        [data-testid="stCameraInput"] video,
        [data-testid="stCameraInput"] img {{
            max-height: min(360px, 45vh);
            object-fit: contain;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: #FBFBF9;
            border: 1px dashed #C9CFCA;
            border-radius: 12px;
            padding: 1.3rem;
        }}

        [data-baseweb="tab-list"] {{
            gap: 18px;
            border-bottom: 1px solid #DFE1DD;
        }}
        [data-baseweb="tab"] {{
            height: 42px;
            padding-left: 0;
            padding-right: 0;
            background: transparent;
        }}

        [data-testid="stExpander"] {{
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 8px;
        }}

        hr {{
            border: none;
            border-top: 1px solid #E1E2DE;
            margin: 1.2rem 0;
        }}

        /* Shared responsive layout for all four pages and their detail panels. */
        [data-testid="stColumn"] {{ min-width: 0; }}
        .metric-value, .hero-advice, .top-brand-caption {{
            overflow-wrap: anywhere;
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            max-width: 100%;
            overflow-x: auto;
        }}

        @media (max-width: 850px) {{
            [data-testid="stImage"] {{ width: 100% !important; }}
            [data-testid="stImage"] img {{
                margin-inline: auto !important;
                max-width: 100% !important;
            }}
            [data-testid="stMainBlockContainer"] {{
                padding: 1.5rem 1rem 2.5rem;
            }}
            [data-testid="stHorizontalBlock"] {{
                flex-direction: column !important;
                gap: 1rem !important;
            }}
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
                width: 100% !important;
                flex: 1 1 auto !important;
                min-width: 0 !important;
            }}
            .top-appbar {{ flex-wrap: wrap; gap: 12px; }}
            .top-nav-divider {{ margin-bottom: 20px; }}
            .page-title {{ font-size: clamp(26px, 6vw, 34px); }}
            .page-subtitle {{ margin-bottom: 18px; }}
            .metric-card {{ min-height: 88px; padding: 15px; }}
            .metric-value {{ font-size: 26px; line-height: 1.15; }}
            .hero-card, .grading-result {{ padding: 18px; }}
            div.stButton > button, div.stDownloadButton > button {{
                min-height: 44px;
                width: 100%;
            }}
            .st-key-page div[role="radiogroup"] {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                width: 100%;
            }}
            .st-key-page div[role="radiogroup"] > label {{
                min-height: 44px;
                box-sizing: border-box;
            }}
            [data-baseweb="tab-list"] {{
                gap: 16px;
                overflow-x: auto;
            }}
            [data-baseweb="tab"] {{ flex-shrink: 0; min-height: 44px; }}
            [data-testid="stFileUploaderDropzone"] {{
                flex-wrap: wrap;
                padding: 1rem;
            }}
            .st-key-assessment_original_photo [data-testid="stImage"] img {{
                width: 100% !important;
                height: clamp(260px, 60svh, 520px) !important;
                max-height: none;
                object-fit: contain;
            }}
            [data-testid="stCameraInput"] {{ width: 100%; max-width: 100%; }}
            [data-testid="stCameraInput"] video {{ max-width: 100%; }}
        }}

        @media (max-width: 480px) {{
            [data-testid="stMainBlockContainer"] {{ padding-inline: 0.75rem; }}
            .top-brand-name {{ font-size: 23px; }}
            .top-record-count {{ font-size: 11px; }}
            .step-indicator {{ gap: 4px; justify-content: space-between; }}
            .step {{ gap: 4px; padding: 6px; font-size: 10px; }}
            .step-connector {{ display: none; }}
            .empty-state {{ padding: 28px 16px; }}
            .guide-card {{ padding: 15px; }}
            [data-baseweb="input"] input,
            [data-baseweb="textarea"] textarea {{ font-size: 16px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

