"""ui / pages / history for ManGo or Stay."""

from datetime import datetime
from ui.components import empty_state
from ui.components import page_header
from ui.navigation import go_to
import streamlit as st

def render(history_df):
    page_header(
        "Records",
        "History and reports",
        "Search saved assessments and export filtered records as CSV.",
    )

    if history_df.empty:
        empty_state(
            "No assessment history yet",
            "Completed mango assessments are saved automatically and will appear here for review and export.",
        )
        if st.button("Assess a mango", type="primary"):
            go_to("Assess Mango")
    else:
        f1, f2 = st.columns([2, 1.5])
        search = f1.text_input("Search by assessment ID or batch ID")
        date_range = f2.date_input("Date range", value=())

        filtered = history_df.copy()
        if search.strip():
            text = search.strip()
            filtered = filtered[
                filtered["ID"].str.contains(text, case=False, na=False)
                | filtered["Batch ID"].fillna("").str.contains(text, case=False, na=False)
            ]

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered = filtered[
                (filtered["Date"].dt.date >= start_date)
                & (filtered["Date"].dt.date <= end_date)
            ]
        elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
            selected_date = date_range[0]
            filtered = filtered[filtered["Date"].dt.date == selected_date]

        export_columns = [
            "ID", "Batch ID", "Fruit Type", "Ripeness", "Confidence", "Grade",
            "Defect %", "Blemish %", "Damage %", "Severity", "Defect Types", "Date",
        ]
        export_df = filtered[export_columns].copy()
        export_df["Defect Types"] = export_df["Defect Types"].apply(
            lambda values: ", ".join(values) if isinstance(values, list) else str(values)
        )
        export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export filtered records as CSV",
            data=csv_data,
            file_name=f"mango-assessment-history-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=filtered.empty,
        )

        if filtered.empty:
            empty_state("No matching assessments", "Change the search text or date range to see saved records.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Assessments found", len(filtered))
            m2.metric("Average confidence", f"{filtered['Confidence'].mean():.1f}%")
            m3.metric("Average defect coverage", f"{filtered['Defect %'].mean():.2f}%")
            m4.metric("High severity", int((filtered["Severity"] == "High").sum()))

            display = filtered.copy()
            display["Confidence"] = display["Confidence"].map(lambda value: f"{value:.1f}%")
            display["Defect %"] = display["Defect %"].map(lambda value: f"{value:.2f}%")
            display["Date"] = display["Date"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(
                display[["ID", "Batch ID", "Ripeness", "Confidence", "Grade", "Defect %", "Severity", "Date"]],
                hide_index=True,
                use_container_width=True,
            )
