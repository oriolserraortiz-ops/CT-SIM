
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
from datetime import date, datetime
from dateutil import parser as dateparser
import plotly.graph_objects as go

st.set_page_config(page_title="Protocol Similarity & Timeline Classifier", layout="wide")

# -------------------------
# Utility & Parsing helpers
# -------------------------

DEFAULT_WEIGHTS = {
    "indication": 15,
    "population": 10,
    "phase_intent": 10,
    "intervention_class": 15,
    "comparator": 5,
    "primary_endpoint": 15,
    "design": 10,
    "sample_size": 5,
    "setting": 2,
    "concomitant": 3,
    "assessments": 5,
    "timing": 5
}

CLASS_RULES = {
    "competitive": {"similarity_min": 60, "overlap_ratio_min": 0.30},
    "complementary": {"similarity_range": (30, 59.999)},
    "non_complementary": {"similarity_max": 30}
}

DRUG_CLASS_KEYWORDS = {
    "GLP-1": ["glp-1", "glp1", "efpeglenatide", "liraglutide", "semaglutide", "dulaglutide", "exenatide"],
    "SGLT2": ["sglt2", "dapagliflozin", "empagliflozin", "canagliflozin", "ertugliflozin"],
    "PD-1/PD-L1": ["pd-1", "pd-l1", "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab"],
    "EGFR": ["egfr", "osimertinib", "gefitinib", "erlotinib"]
    # Extend as needed
}

INDICATION_KEYWORDS = {
    "Type 2 Diabetes Mellitus": ["type 2 diabetes", "t2dm"],
    "Type 1 Diabetes Mellitus": ["type 1 diabetes", "t1dm"],
    "Obesity": ["obesity", "overweight"],
    # Extend as needed
}

ENDPOINT_KEYWORDS = {
    "HbA1c": ["hba1c"],
    "EGP": ["endogenous glucose production", "egp", "hepatic glucose production", "hgp"],
    "PFS": ["progression-free survival", "pfs"],
    "OS": ["overall survival", "os"],
    # Extend as needed
}

def extract_text_from_pdf(file):
    try:
        text_parts = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception as e:
        return ""

def find_match_keyword(text, keyword_map):
    t = text.lower()
    for label, kws in keyword_map.items():
        for kw in kws:
            if kw in t:
                return label
    return None

def regex_search(text, patterns, flags=re.IGNORECASE|re.MULTILINE):
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m
    return None

def detect_phase(text):
    m = regex_search(text, [r'\bphase\s*(i{1,3}v?|[1-4])\b'])
    if not m:
        return None
    val = m.group(0).lower()
    # Normalize
    roman_to_arabic = {"i":1,"ii":2,"iii":3,"iv":4}
    m2 = re.search(r'(i{1,3}v?)|([1-4])', val)
    if not m2:
        return None
    token = m2.group(0)
    if token.isdigit():
        return f"Phase {token}"
    return f"Phase {roman_to_arabic.get(token, token)}"

def detect_randomization_blinding(text):
    t = text.lower()
    randomized = "random" in t
    double_blind = "double-blind" in t or "double blind" in t
    single_blind = "single-blind" in t or "single blind" in t
    open_label = "open-label" in t or "open label" in t
    return {
        "randomized": randomized,
        "blind": "double" if double_blind else ("single" if single_blind else ("open-label" if open_label else None))
    }

def detect_duration_weeks(text):
    # Try patterns like "Week 56" or "56 weeks"
    m1 = regex_search(text, [r'week\s*([0-9]{1,3})', r'([0-9]{1,3})\s*weeks?'])
    if m1:
        try:
            # Try to get the number from either group 1 or the whole match
            num = None
            g = re.findall(r'([0-9]{1,3})', m1.group(0))
            if g:
                num = int(g[-1])
            if num:
                return num
        except:
            pass
    return None

def detect_sample_size(text):
    # naive: look for "n=###" or "approximately ### participants"
    m = regex_search(text, [r'\bn\s*=\s*([0-9]{2,5})',
                            r'approximately\s+([0-9]{2,5})\s+(participants|subjects)',
                            r'([0-9]{2,5})\s+(participants|subjects)'])
    if m:
        try:
            for g in m.groups():
                if g and re.match(r'^[0-9]{2,5}$', str(g)):
                    return int(g)
        except:
            pass
    return None

def detect_comparator(text):
    t = text.lower()
    if "placebo" in t:
        return "Placebo"
    if "active-controlled" in t or "active controlled" in t:
        # Rough extraction of the active comparator name if present
        m = regex_search(t, [r'compared to ([a-z0-9\-\s]+)'])
        if m:
            return f"Active ({m.group(1).strip()})"
        return "Active"
    if "standard of care" in t:
        return "Standard of care"
    return None

def find_dates(text):
    # Heuristics to find dates like "31-Jul-2019", "July 31, 2019", "2019-07-31"
    patterns = [
        r'\b(\d{1,2}[-/ ]?[A-Za-z]{3,9}[-/ ]?\d{2,4})\b',
        r'\b([A-Za-z]{3,9}[-/ ]?\d{1,2},[-/ ]?\s*\d{4})\b',
        r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
    ]
    matches = []
    for pat in patterns:
        matches += re.findall(pat, text)
    parsed = []
    for m in matches:
        try:
            parsed.append(dateparser.parse(m, dayfirst=False, fuzzy=True).date())
        except:
            pass
    parsed = sorted(set(parsed))
    return parsed

def extract_protocol_fields(file, name_hint=None):
    text = extract_text_from_pdf(file)
    fields = {
        "file_name": getattr(file, "name", name_hint or "protocol.pdf"),
        "title": None,
        "nct_number": None,
        "indication": find_match_keyword(text, INDICATION_KEYWORDS),
        "phase": detect_phase(text),
        "drug_class": find_match_keyword(text, DRUG_CLASS_KEYWORDS),
        "primary_endpoint": find_match_keyword(text, ENDPOINT_KEYWORDS),
        "comparator": detect_comparator(text),
        "randomized": detect_randomization_blinding(text)["randomized"],
        "blinding": detect_randomization_blinding(text)["blind"],
        "duration_weeks": detect_duration_weeks(text),
        "sample_size": detect_sample_size(text),
        "dates_detected": find_dates(text)
    }
    # Title heuristic
    m_title = regex_search(text, [r'protocol title:\s*(.+)\n', r'official study title:\s*(.+)\n'])
    if m_title:
        fields["title"] = m_title.group(1).strip()
    # NCT heuristic
    m_nct = regex_search(text, [r'\bNCT0?\d{7,8}\b', r'\bNCT\s*[:#]?\s*0?\d{7,8}\b'])
    if m_nct:
        fields["nct_number"] = m_nct.group(0).replace(" ", "").replace(":", "")
    return fields, text

def degree_match(a, b):
    # Convert to lower and handle None
    if not a or not b:
        return 0.0
    a, b = str(a).lower(), str(b).lower()
    if a == b:
        return 1.0
    # loose contains
    if a in b or b in a:
        return 0.75
    # related classes (e.g., GLP-1 vs GLP-1/GLP-1 RA strings)
    if any(tok in a for tok in ["glp"]) and any(tok in b for tok in ["glp"]):
        return 1.0
    return 0.5 if (a.split()[0] == b.split()[0]) else 0.25

def score_similarity(p1, p2, weights):
    # Each field -> degree (0..1), then multiply by weight
    # Fallbacks for missing values are handled by degree_match (0.0)
    dims = {}
    dims["indication"] = degree_match(p1.get("indication"), p2.get("indication"))
    # population: crude proxy using sample_size closeness and metformin keywords (not perfect)
    pop1 = p1.get("population_hint") or ""
    pop2 = p2.get("population_hint") or ""
    dims["population"] = 0.5 if (p1.get("sample_size") and p2.get("sample_size")) else 0.25
    dims["phase_intent"] = degree_match(p1.get("phase"), p2.get("phase"))
    dims["intervention_class"] = degree_match(p1.get("drug_class"), p2.get("drug_class"))
    dims["comparator"] = degree_match(p1.get("comparator"), p2.get("comparator"))
    dims["primary_endpoint"] = degree_match(p1.get("primary_endpoint"), p2.get("primary_endpoint"))
    # design
    d1 = f'{"rand" if p1.get("randomized") else "non-rand"}|{p1.get("blinding") or "none"}'
    d2 = f'{"rand" if p2.get("randomized") else "non-rand"}|{p2.get("blinding") or "none"}'
    dims["design"] = degree_match(d1, d2)
    # sample size closeness
    s1, s2 = p1.get("sample_size"), p2.get("sample_size")
    if s1 and s2:
        ratio = min(s1, s2) / max(s1, s2)
        dims["sample_size"] = 1.0 if ratio >= 0.8 else (0.75 if ratio >= 0.5 else (0.5 if ratio >= 0.25 else 0.25))
    else:
        dims["sample_size"] = 0.25
    # setting proxy: multicenter vs single center keyword heuristics
    # If sample size > 100 assume multicenter, else single-center
    set1 = "multicenter" if (s1 or 0) > 100 else "single-center"
    set2 = "multicenter" if (s2 or 0) > 100 else "single-center"
    dims["setting"] = degree_match(set1, set2)
    # concomitant policy proxy: look for "rescue" word occurrence counts
    dims["concomitant"] = 0.75 if all(k in p1.get("full_text", "").lower() for k in ["rescue","concomitant"]) and all(k in p2.get("full_text", "").lower() for k in ["rescue","concomitant"]) else 0.25
    # procedure/assessments proxy: endpoint overlap already captured; assume partial
    dims["assessments"] = 0.5 if (dims["primary_endpoint"] >= 0.5) else 0.25
    # timing/duration: compare duration_weeks
    w1, w2 = p1.get("duration_weeks"), p2.get("duration_weeks")
    if w1 and w2:
        ratio = min(w1, w2) / max(w1, w2)
        dims["timing"] = 1.0 if ratio >= 0.8 else (0.75 if ratio >= 0.5 else (0.5 if ratio >= 0.25 else 0.25))
    else:
        dims["timing"] = 0.25

    # Weighted sum
    score = 0.0
    breakdown = []
    for k, deg in dims.items():
        w = weights.get(k, 0)
        val = deg * w
        score += val
        breakdown.append({"dimension": k, "degree": round(deg,2), "weight": w, "weighted": round(val,2)})
    return round(score, 2), breakdown

def compute_overlap_ratio(a_start, a_end, b_start, b_end):
    # return overlap duration / min(duration_a, duration_b)
    if not (a_start and a_end and b_start and b_end):
        return 0.0
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    overlap_days = (end - start).days
    if overlap_days <= 0:
        return 0.0
    a_days = (a_end - a_start).days
    b_days = (b_end - b_start).days
    min_days = max(min(a_days, b_days), 1)
    return overlap_days / min_days

def classify_pair(score, overlap_ratio, endpoint_match, moa_match):
    # Endpoint & MoA booleans can tilt to competition
    if score >= CLASS_RULES["competitive"]["similarity_min"] and overlap_ratio >= CLASS_RULES["competitive"]["overlap_ratio_min"] and (endpoint_match or moa_match):
        return "Competitive"
    if 30 <= score < 60:
        return "Complementary"
    if score < 30:
        return "Non-complementary"
    # default
    return "Complementary"

def to_date(obj):
    if isinstance(obj, (date, datetime)):
        return obj.date() if isinstance(obj, datetime) else obj
    if isinstance(obj, str) and obj.strip():
        try:
            return dateparser.parse(obj).date()
        except:
            return None
    return None

# -------------------------
# Sidebar - Weights
# -------------------------

st.sidebar.header("Scoring Weights")
weights = {}
for k in DEFAULT_WEIGHTS:
    weights[k] = st.sidebar.slider(f"{k}", min_value=0, max_value=20, value=DEFAULT_WEIGHTS[k], step=1)
st.sidebar.caption("Weights sum is unconstrained here for flexibility.")

# -------------------------
# Main UI
# -------------------------

st.title("Protocol Similarity & Execution Timeline")
st.write("Upload **two or more** protocols (PDF). We’ll auto-extract fields, let you edit, then score similarity and classify as Competitive / Complementary / Non-complementary.")

files = st.file_uploader("Upload Protocol PDFs", type=["pdf"], accept_multiple_files=True)

protocols = []
if files:
    cols = st.columns(2)
    for idx, f in enumerate(files):
        with cols[idx % 2]:
            with st.spinner(f"Reading {f.name} ..."):
                fields, full_text = extract_protocol_fields(f, name_hint=f.name)
                fields["full_text"] = full_text
            st.subheader(f"Protocol {idx+1}: {fields['file_name']}")

            with st.expander("Parsed fields (edit as needed)"):
                fields["title"] = st.text_input("Title", value=fields.get("title") or "", key=f"title_{idx}")
                fields["nct_number"] = st.text_input("NCT Number", value=fields.get("nct_number") or "", key=f"nct_{idx}")
                fields["indication"] = st.text_input("Indication", value=fields.get("indication") or "", key=f"indi_{idx}")
                fields["phase"] = st.text_input("Phase", value=fields.get("phase") or "", key=f"phase_{idx}")
                fields["drug_class"] = st.text_input("Intervention class / MoA", value=fields.get("drug_class") or "", key=f"moa_{idx}")
                fields["primary_endpoint"] = st.text_input("Primary endpoint (keyword)", value=fields.get("primary_endpoint") or "", key=f"ep_{idx}")
                fields["comparator"] = st.text_input("Comparator", value=fields.get("comparator") or "", key=f"comp_{idx}")
                fields["randomized"] = st.checkbox("Randomized?", value=bool(fields.get("randomized")), key=f"rand_{idx}")
                fields["blinding"] = st.selectbox("Blinding", options=["none","single","double","open-label"], index=["none","single","double","open-label"].index(fields.get("blinding") or "none"), key=f"blind_{idx}")
                fields["duration_weeks"] = st.number_input("Planned treatment duration (weeks)", min_value=0, value=int(fields.get("duration_weeks") or 0), step=1, key=f"dur_{idx}")
                fields["sample_size"] = st.number_input("Planned sample size", min_value=0, value=int(fields.get("sample_size") or 0), step=1, key=f"n_{idx}")
                # Execution window
                st.markdown("**Execution window (Start/End dates)**")
                detected_dates = fields.get("dates_detected") or []
                dates_suggestion = ", ".join([d.isoformat() for d in detected_dates[:3]])
                if dates_suggestion:
                    st.caption(f"Detected dates (first 3): {dates_suggestion}")
                start_str = st.text_input("Start date (YYYY-MM-DD)", value="", key=f"start_{idx}")
                end_str = st.text_input("End date (YYYY-MM-DD)", value="", key=f"end_{idx}")
                fields["start_date"] = to_date(start_str)
                fields["end_date"] = to_date(end_str)

            protocols.append(fields)

# -------------------------
# Pairwise comparison
# -------------------------

def build_matrix(protocols):
    rows = []
    for i in range(len(protocols)):
        for j in range(i+1, len(protocols)):
            p1, p2 = protocols[i], protocols[j]
            score, breakdown = score_similarity(p1, p2, weights)
            # overlap ratio on execution timelines
            ovr = compute_overlap_ratio(p1.get("start_date"), p1.get("end_date"),
                                        p2.get("start_date"), p2.get("end_date"))
            endpoint_match = degree_match(p1.get("primary_endpoint"), p2.get("primary_endpoint")) >= 0.75
            moa_match = degree_match(p1.get("drug_class"), p2.get("drug_class")) >= 0.75
            label = classify_pair(score, ovr, endpoint_match, moa_match)

            rows.append({
                "Protocol A": p1.get("file_name"),
                "Protocol B": p2.get("file_name"),
                "Similarity": score,
                "Overlap ratio": round(ovr, 2),
                "Endpoint aligned": endpoint_match,
                "MoA aligned": moa_match,
                "Classification": label,
                "Breakdown": breakdown,
                "A_start": p1.get("start_date").isoformat() if p1.get("start_date") else "",
                "A_end": p1.get("end_date").isoformat() if p1.get("end_date") else "",
                "B_start": p2.get("start_date").isoformat() if p2.get("start_date") else "",
                "B_end": p2.get("end_date").isoformat() if p2.get("end_date") else "",
            })
    return pd.DataFrame(rows)

if len(protocols) >= 2:
    st.markdown("---")
    st.header("Pairwise results")
    df = build_matrix(protocols)
    st.dataframe(df.drop(columns=["Breakdown"]), use_container_width=True)

    # Visualization: Similarity vs Execution Timeline (line segments per protocol)
    st.subheader("Similarity vs Execution Timelines")
    st.caption("Each protocol is plotted as a horizontal line at a y-position equal to its mean similarity across its pairings.")

    # Compute mean similarity per protocol
    prot_names = [p["file_name"] for p in protocols]
    mean_sim = {name: [] for name in prot_names}
    for _, row in df.iterrows():
        mean_sim[row["Protocol A"]].append(row["Similarity"])
        mean_sim[row["Protocol B"]].append(row["Similarity"])
    mean_sim = {k: (sum(v)/len(v) if v else 0) for k,v in mean_sim.items()}

    fig = go.Figure()
    for p in protocols:
        start = p.get("start_date")
        end = p.get("end_date")
        if not (start and end):  # skip if dates missing
            continue
        y = mean_sim.get(p["file_name"], 0)
        fig.add_trace(go.Scatter(x=[start, end], y=[y, y],
                                 mode="lines+markers",
                                 name=p["file_name"],
                                 hovertemplate=f"{p['file_name']}<br>Similarity(y): %{{y:.1f}}<br>Start: {start}<br>End: {end}<extra></extra>"))
    fig.update_layout(xaxis_title="Execution timeline (date)",
                      yaxis_title="Similarity score (mean across pairs)",
                      height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Download
    st.download_button("Download results as CSV", data=df.to_csv(index=False), file_name="pairwise_results.csv", mime="text/csv")

else:
    st.info("Upload at least two protocols to compute similarity and visualize timelines.")

st.markdown("---")
st.caption("Tip: edit fields under each protocol to refine the scoring and classification. All heuristics are transparent in the code.")
