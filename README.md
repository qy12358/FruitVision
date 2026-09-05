# ManGo or Stay

Harumanis mango ripeness and surface quality assessment with Streamlit.

Run from the project directory using an environment with the application dependencies installed:

```powershell
streamlit run app.py
```

## Code structure

| Location | Responsibility |
| --- | --- |
| `app.py` | Application startup and page routing |
| `config.py` | Shared colours, labels, advice, and database paths |
| `ui/pages/home.py` | Overview dashboard |
| `ui/pages/assess.py` | Photo input and assessment workflow |
| `ui/pages/details.py` | Saved assessment details |
| `ui/pages/history.py` | History filtering and report dashboard |
| `ui/theme.py` | Streamlit page configuration and CSS |
| `ui/navigation.py` | Navigation and session state initialization |
| `ui/components.py` | Reusable headers, badges, cards, and gauges |
| `ui/assessment_results.py` | Live assessment results and analysis charts |
| `ui/education.py` | Histogram explanations and model information |
| `services/assessment.py` | Assessment pipeline and per-object result aggregation |
| `services/models.py` | Processing resources and cached model loaders |
| `services/storage.py` | SQLite persistence, JSON serialization, and image encoding |
| `services/reports.py` | Report generation from stored assessments |
| `services/histograms.py` | Label formatting and histogram calculations |
| `modules/` | Image preprocessing, identification, ripeness, blemish detection, grading, and report algorithms |
| `training/` | Dataset auditing, model training, and evaluation scripts |
| `models/` | Trained models and evaluation artifacts |
| `data/` | Local assessment database |

Pages call services and render results using shared UI components. Services use the existing algorithms in `modules/` and do not import the UI. Model loading retains Streamlit resource caching. Session initialization and database initialization run explicitly from `app.main()` on each Streamlit rerun.

To change a screen, edit its file in `ui/pages/`. To change assessment orchestration or persistence, edit the corresponding service. To change an image-processing or classification algorithm, edit its existing file in `modules/`.
