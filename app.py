"""ManGo or Stay — run with: streamlit run app.py."""
from services.storage import initialise_database, load_history_dataframe
from ui.navigation import initialise_session, render_navigation
from importlib import import_module
from ui.theme import apply_theme, configure_page


def main():
    configure_page()
    apply_theme()
    initialise_session()
    initialise_database()
    history_df = load_history_dataframe()
    page = render_navigation(history_df)
    pages = {
        "Home": "home",
        "Assess Mango": "assess",
        "Assessment Details": "details",
        "History & Reports": "history",
    }
    import_module(f"ui.pages.{pages[page]}").render(history_df)


if __name__ == "__main__":
    main()
