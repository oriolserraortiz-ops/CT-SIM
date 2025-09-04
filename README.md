# Protocol Similarity & Timeline Classifier (Streamlit)

A lightweight web app to upload clinical trial protocols (PDF), auto-extract key fields, compute a similarity score
(using a transparent 12-dimension rubric), and classify study pairs as **Competitive / Complementary / Non-complementary**.
It also plots **Similarity vs Execution Timelines**.

## Features
- Multi-file PDF upload
- Heuristic extraction (indication, phase, MoA/class, comparator, endpoints, durations, N). All fields are editable in-UI.
- Configurable weights per dimension
- Pairwise similarity matrix with timeline overlap ratio
- Classification rule that incorporates score + timeline overlap + endpoint/MoA alignment
- Interactive Plotly timeline visualization
- CSV export

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed in your terminal.

## Deploy to Streamlit Community Cloud
1. Push this folder to a new GitHub repo.
2. Go to https://share.streamlit.io , sign in, and **New app** → point to your repo and `app.py`.
3. Add the same Python version as your local environment if you have issues (e.g., 3.10/3.11).
4. Click **Deploy**.

## Deploy to Hugging Face Spaces (Gradio/Streamlit)
1. Create a new Space → **Streamlit** template.
2. Upload all files here (`app.py`, `requirements.txt`).
3. Spaces will build and serve automatically.

## Notes & Limitations
- PDF parsing is heuristic; use the expandable editor to correct fields (especially **start/end dates**).
- You can tune scoring **weights** in the left sidebar to match your portfolio lens.
- Extend dictionaries in `app.py` (e.g., `DRUG_CLASS_KEYWORDS`, `ENDPOINT_KEYWORDS`) for your TA.
