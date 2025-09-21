import os
import re
import io
import json
import hashlib
import numpy as np
import streamlit as st
import PyPDF2 as pdf
import google.genai as genai
from google.genai import types

# Import database functions from the other file
import db

# -------------------- Config --------------------
DEFAULT_GENAI_MODEL_EXTRACT = "gemini-2.5-flash"  # parsing/AI scoring model
DEFAULT_GENAI_MODEL_EMBED = "text-embedding-004"  # embeddings model via client.models.embed_content

# -------------------- Helpers: Robust JSON cleanup & validation --------------------
_PREFIX_RE = re.compile(r'^\s*(?:```)', flags=re.IGNORECASE)

def _extract_outer_json_block(s: str) -> str:
    if not s:
        return s
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("` \n")
    if not (t.startswith("{") and t.endswith("}")):
        t = _PREFIX_RE.sub("", t).strip()
    first = t.find("{")
    last = t.rfind("}")
    return t[first:last+1] if first != -1 and last != -1 and first < last else t

# -------------------- JD utilities: multi-role detection & normalization --------------------
ROLE_SPLIT_RE = re.compile(r'(?:^|\n)\s*\d+\.\s+([^\n]+)\s*(?=\n)', flags=re.IGNORECASE)  # numbered headings
ALIASES = {
    "pyspark": "spark",
    "tableau": "tableau",
    "power bi": "power bi",
    "eda": "exploratory data analysis",
    "c++": "c++",
    "nlp": "nlp",
    "cv": "computer vision",
}

def normalize_tokens(skills):
    out = set()
    for s in skills or []:
        k = (s or "").strip().lower()
        if not k:
            continue
        k = ALIASES.get(k, k)
        out.add(k)
    return sorted(out)

def parse_numeric_constraints(text: str):
    t = (text or "").lower()
    stipend = None
    import re as _re
    m = _re.search(r'(?:₹|inr|\brs\.?)\s*([\d,]+)\s*(?:per\s*)?(month|mo|monthly)?', t)
    if m:
        amt = int(m.group(1).replace(',', ''))
        per = 'month' if (m.group(2) or '').startswith('mo') or (m.group(2) or '') == 'month' else None
        stipend = {"amount": amt, "currency": "INR", "period": per or "month"}
    internship_months = None
    m = _re.search(r'(\d+)\s*month', t)
    if m:
        internship_months = int(m.group(1))
    bond_months = None
    m = _re.search(r'(\d+(?:\.\d+)?)\s*year', t)
    if m:
        try:
            bond_months = int(round(float(m.group(1)) * 12))
        except:
            pass
    m2 = _re.search(r'(\d+)\s*month', t)
    if m2 and bond_months is None:
        bond_months = int(m2.group(1))
    batch_year_max = None
    m = _re.search(r'(20\d{2}).*?(earlier|and earlier|or earlier)', t)
    if m:
        batch_year_max = int(m.group(1))
    return stipend, internship_months, bond_months, batch_year_max

def split_roles_heuristic(jd_text: str):
    titles = ROLE_SPLIT_RE.findall(jd_text or "")
    if not titles:
        return [{"title": None, "text": jd_text or ""}]
    parts = re.split(r'(?:^|\n)\s*\d+\.\s+[^\n]+\s*(?=\n)', jd_text)
    roles = []
    for title, seg in zip(titles, parts[1:]):
        roles.append({"title": title.strip(), "text": seg.strip()})
    return roles

# -------------------- sqlite row helpers --------------------
def _row_to_dict(row):
    try:
        return dict(row)
    except Exception:
        try:
            return {k: row[k] for k in row.keys()}
        except Exception:
            return row

def _rows_to_dicts(rows):
    return [ _row_to_dict(r) for r in rows or [] ]

# -------------------- Gemini client --------------------
def _make_genai_client(api_key: str | None) -> genai.Client:
    if api_key and api_key.strip():
        return genai.Client(api_key=api_key.strip())
    env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not env_key:
        raise RuntimeError("Google API key not set. Provide it in the sidebar or via GOOGLE_API_KEY/GEMINI_API_KEY.")
    return genai.Client()

# -------------------- Gemini wrappers (Client-based) --------------------
def _genai_generate_text(client: genai.Client, model: str, prompt: str) -> str:
    resp = client.models.generate_content(model=model, contents=prompt)
    return getattr(resp, "text", None) or ""

def _genai_embed_text(client: genai.Client, model: str, text: str) -> list[float]:
    eresp = client.models.embed_content(model=model, contents=text)
    if hasattr(eresp, "embeddings") and eresp.embeddings:
        emb0 = eresp.embeddings[0]
        vec = getattr(emb0, "values", None) or getattr(emb0, "embedding", None)
        if vec is not None:
            return list(vec)
    try:
        return list(eresp["embeddings"][0]["values"])
    except Exception:
        pass
    raise RuntimeError("Unexpected embeddings response shape from Gemini API.")

# -------------------- LLM and Prompt Engineering for Structured Data Extraction --------------------
def extract_structured_data(client: genai.Client, model_name: str, text: str, data_type: str):
    if data_type == "resume":
        prompt = f"""
Return RFC 8259–compliant JSON only (no commentary, no markdown).
Resume: --- {text} ---
Schema: {{"name": "string|null", "experience": [{{"company": "string|null", "role": "string|null"}}], "education": [{{"institute_name": "string|null", "degree": "string|null", "specialization": "string|null"}}], "skills": ["string"], "predicted_role": "string|null"}}
Output: JSON only. First char '{{', last char '}}'.
""".strip()
    elif data_type == "jd":
        prompt = f"""
Return RFC 8259–compliant JSON ONLY for one or more roles in this JD.
Rules:
- Detect multiple roles; output as roles array.
- Normalize skills to lowercase and deduplicate.
- Parse numeric constraints: stipend (amount,currency,period), internship_months, bond_months, batch_year_max.
- If intern/fresher, set required_experience_years = 0 else parse minimal years.
- Map education to "bachelor's","master's","phd" or null when unclear.
- No prose, no code fences. First char '{{' last char '}}'.

Input JD:
---
{text}
---

Schema:
{{
  "roles": [
    {{
      "job_title": "string|null",
      "location": "string|null",
      "employment_type": ["string"],
      "internship_months": "integer|null",
      "bond_months": "integer|null",
      "stipend": {{"amount":"integer|null","currency":"string|null","period":"string|null"}},
      "schedule": "string|null",
      "required_skills": ["string"],
      "responsibilities": ["string"],
      "required_experience_years": "integer|null",
      "batch_year_max": "integer|null",
      "required_education": ["string"]|"string|null"
    }}
  ]
}}
""".strip()
    else:
        return None

    try:
        raw = _genai_generate_text(client, model_name, prompt)
        candidate = _extract_outer_json_block(raw)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            retry_prompt = prompt + "\nReminder: JSON ONLY. Start with '{' and end with '}'. No prose."
            raw2 = _genai_generate_text(client, model_name, retry_prompt)
            candidate2 = _extract_outer_json_block(raw2)
            data = json.loads(candidate2)
        if data_type == "jd":
            roles = data.get("roles") if isinstance(data, dict) else None
            if not roles:
                blocks = split_roles_heuristic(text)
                roles = []
                for b in blocks:
                    stipend, intern_m, bond_m, batch_max = parse_numeric_constraints(b["text"])
                    roles.append({
                        "job_title": b["title"] or None,
                        "location": None,
                        "employment_type": [],
                        "internship_months": intern_m,
                        "bond_months": bond_m,
                        "stipend": stipend,
                        "schedule": None,
                        "required_skills": [],
                        "responsibilities": [],
                        "required_experience_years": 0 if (b["title"] or "").lower().find("intern") != -1 else None,
                        "batch_year_max": batch_max,
                        "required_education": None,
                    })
                data = {"roles": roles}
            for r in data.get("roles", []):
                r["required_skills"] = normalize_tokens(r.get("required_skills") or [])
        return data
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Error processing document with Gemini: {e}")
        return None

# -------------------- Scoring (deterministic parts) --------------------
def cosine_to_pct(c: float) -> float:
    from math import isnan
    if c is None or isnan(c):
        return 0.0
    return max(0.0, min(1.0, (c - 0.2) / 0.6)) * 100.0

def extract_years_experience(text: str) -> float:
    t = (text or "").lower()
    yrs = 0.0
    m = re.search(r'(\d+(?:\.\d+)?)\s*[\+\-]?\s*year', t)
    if m:
        try:
            yrs = float(m.group(1))
        except:
            pass
    return yrs

def normalize_list(xs):
    return sorted({(x or "").strip().lower() for x in xs or [] if (x or "").strip()})

def hard_match_score(jd_role: dict, resume_data: dict) -> float:
    req = normalize_list(jd_role.get("required_skills"))
    have = normalize_list(resume_data.get("skills"))
    if not req:
        return 0.0
    resp = " ".join(jd_role.get("responsibilities") or []).lower()
    weights = {}
    for s in req:
        w = 1.0
        if s and s in resp:
            w *= 1.25
        weights[s] = w
    num = sum(weights[s] for s in req if s in have)
    den = sum(weights.values())
    return 100.0 * (num / den) if den > 0 else 0.0

def other_factors_score(jd_role: dict, resume_data: dict, resume_text: str) -> float:
    score = 0.0
    edu_req = jd_role.get("required_education")
    edu_req_set = set()
    if isinstance(edu_req, list):
        edu_req_set = {e.lower() for e in edu_req if isinstance(e, str)}
    elif isinstance(edu_req, str) and edu_req:
        edu_req_set = {edu_req.lower()}
    edu_have = [(e.get("degree") or "").lower() for e in (resume_data.get("education") or [])]
    if edu_req_set:
        if any(any(req in deg for req in edu_req_set) for deg in edu_have):
            score += 8
        elif edu_have:
            score += 4
    req_yrs = jd_role.get("required_experience_years")
    have_yrs = extract_years_experience(resume_text)
    if req_yrs is not None:
        if req_yrs <= 1:
            if have_yrs <= 1:
                score += 6
            elif have_yrs <= 2:
                score += 3
        else:
            if have_yrs >= req_yrs:
                score += 6
            elif have_yrs >= max(0, req_yrs - 1):
                score += 3
    text = (resume_text or "").lower()
    loc = (jd_role.get("location") or "").lower()
    if loc and (loc in text or "relocation" in text or "remote" in text):
        score += 3
    sched = (jd_role.get("schedule") or "").lower()
    if sched and ("day shift" in sched or "monday to friday" in sched):
        if "immediately available" in text or "join immediately" in text:
            score += 3
        else:
            score += 1
    bond_m = jd_role.get("bond_months")
    if bond_m:
        if "bond" in text or "service agreement" in text or "commitment" in text:
            score += 2
    return min(20.0, score)

def compute_semantic_match_pct(client, embed_model, jd_text, resume_text) -> float:
    try:
        jd_vec = np.array(_genai_embed_text(client, embed_model, jd_text), dtype=np.float32)
        rs_vec = np.array(_genai_embed_text(client, embed_model, resume_text), dtype=np.float32)
        denom = np.linalg.norm(jd_vec) * np.linalg.norm(rs_vec)
        if denom == 0:
            return 0.0
        c = float(np.dot(jd_vec, rs_vec) / denom)
        return cosine_to_pct(c)
    except Exception:
        return 0.0

# -------------------- AI rubric scoring --------------------
def _cache_key(jd_role: dict, resume_text: str) -> str:
    blob = json.dumps(jd_role, sort_keys=True) + "||" + (resume_text or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def ai_rubric_score(client, model, jd_role: dict, resume_text: str) -> dict:
    schema = {
      "ai_score": "integer (0-100)",
      "subscores": {
        "skills": "integer (0-40)",
        "experience": "integer (0-20)",
        "education": "integer (0-15)",
        "projects": "integer (0-15)",
        "constraints": "integer (0-10)"
      },
      "explanations": {
        "skills_missing": ["string"],
        "strong_matches": ["string"]
      }
    }
    prompt = f"""
Act as a meticulous HR evaluator for the specific role below.
Return RFC8259 JSON ONLY. First char '{{' last char '}}'. No markdown, no comments.

Role JSON:
{json.dumps(jd_role, ensure_ascii=False)}

Resume Text:
---
{resume_text}
---

Scoring rubric (total 100):
- Skills coverage (0-40): award points for JD required_skills present with evidence from resume.
- Experience fit (0-20): compare resume years to required_experience_years or intern expectations.
- Education fit (0-15): match degree(s) to required_education; related fields get partial credit.
- Project/impact relevance (0-15): projects/work matching responsibilities/tools.
- Constraints (0-10): location alignment, batch_year_max, internship/bond acceptance hints.

Output JSON exactly:
{json.dumps(schema, ensure_ascii=False)}
"""
    raw = _genai_generate_text(client, model, prompt)
    candidate = _extract_outer_json_block(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        raw2 = _genai_generate_text(client, model, prompt + "\nReminder: JSON ONLY.")
        candidate2 = _extract_outer_json_block(raw2)
        data = json.loads(candidate2)
    ai = int(max(0, min(100, int(data.get("ai_score", 0)))))
    subs = data.get("subscores", {}) or {}
    return {"ai_score": ai, "subscores": subs}

def compute_final_score_with_ai(client, embed_model, llm_model, jd_role, resume_text, resume_data, cache: dict) -> tuple[float, dict]:
    hard = hard_match_score(jd_role, resume_data)
    jd_for_sem = " ".join([
        jd_role.get("job_title") or "",
        " ".join(jd_role.get("responsibilities") or []),
        " ".join(jd_role.get("required_skills") or [])
    ])
    sem = compute_semantic_match_pct(client, embed_model, jd_for_sem, resume_text)
    other = other_factors_score(jd_role, resume_data, resume_text)
    key = _cache_key(jd_role, resume_text)
    if key in cache:
        ai_dict = cache[key]
    else:
        ai_dict = ai_rubric_score(client, llm_model, jd_role, resume_text)
        cache[key] = ai_dict
    ai = ai_dict["ai_score"]
    final = 0.35*hard + 0.15*sem + 0.10*other + 0.40*ai
    return round(final, 2), {"hard": hard, "semantic": sem, "other": other, "ai": ai, "ai_subscores": ai_dict.get("subscores", {})}

# -------------------- PDF Extraction --------------------
def extract_text_from_pdf(pdf_file_object):
    try:
        pdf_reader = pdf.PdfReader(pdf_file_object)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error reading a PDF file: {e}")
        return None

# -------------------- THEME: Futuristic Neon Glass --------------------
THEME_CSS = """
<style>
:root {
  --bg: #0b0f17;
  --panel: rgba(17,25,40,0.7);
  --panel-strong: rgba(17,25,40,0.9);
  --text: #e5f1ff;
  --muted: #9bb0c8;
  --primary: #7c5cff;
  --primary-2: #00e1ff;
  --accent: #22d3ee;
  --danger: #ff5577;
  --success: #59f28f;
  --warning: #ffd166;
  --glow: 0 0 24px rgba(124,92,255,0.35), 0 0 48px rgba(0,225,255,0.15);
  --radius: 16px;
  --radius-sm: 12px;
  --radius-lg: 20px;
}
html, body, .stApp {
  background: radial-gradient(1200px 700px at 10% 10%, rgba(124,92,255,0.08), transparent 50%),
              radial-gradient(1000px 500px at 90% 30%, rgba(0,225,255,0.08), transparent 55%),
              linear-gradient(180deg, #0b0f17 0%, #080b12 100%) !important;
  color: var(--text);
}
.block-container { padding-top: 2rem !important; }
.neon-panel {
  background: var(--panel);
  border: 1px solid rgba(124,92,255,0.18);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), var(--glow);
  backdrop-filter: blur(14px) saturate(120%);
  border-radius: var(--radius);
  padding: 1.2rem 1.2rem;
  margin : 1.2rem;
}
.neon-strong { background: var(--panel-strong); }
.hero {
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.5rem;
  background: linear-gradient(135deg, rgba(124,92,255,0.22), rgba(0,225,255,0.18));
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 24px 60px rgba(0,0,0,0.35), 0 0 50px rgba(124,92,255,0.25) inset;
  position: relative; overflow: hidden;
}
.hero:before {
  content: ""; position: absolute; top: -40%; left: -10%;
  width: 120%; height: 180%;
  background: radial-gradient(closest-side, rgba(255,255,255,0.08), transparent 60%);
  filter: blur(40px); transform: rotate(15deg);
  animation: drift 14s linear infinite;
}
@keyframes drift { 0% { transform: translateX(0) rotate(15deg);} 100% { transform: translateX(-12%) rotate(15deg);} }
h1, h2, h3 { color: var(--text); letter-spacing: 0.3px; }
.stButton>button {
  background: linear-gradient(135deg, var(--primary), var(--primary-2));
  color: white; border: 0; border-radius: 12px;
  padding: 0.6rem 1rem; box-shadow: var(--glow);
  transition: transform .08s ease, box-shadow .2s ease;
}
.stButton>button:hover { transform: translateY(-1px) scale(1.01); box-shadow: 0 0 24px rgba(0,225,255,0.4); }
.stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div, .stFileUploader>div, .stNumberInput input {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(124,92,255,0.22) !important;
  color: var(--text) !important; border-radius: 12px !important;
}
.stTabs [role="tab"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(124,92,255,0.18);
  border-radius: 12px; color: var(--muted);
  padding : 1.2rem;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(124,92,255,0.22), rgba(0,225,255,0.18));
  color: var(--text) !important;
}
.metric {
  border-radius: 14px; padding: 0.8rem 1rem;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(124,92,255,0.2);
  display: flex; justify-content: space-between; align-items: center;
}
.chip {
  display: inline-block; padding: 0.25rem 0.6rem;
  border-radius: 999px; background: rgba(124,92,255,0.2);
  border: 1px solid rgba(124,92,255,0.35); color: var(--text);
  margin-right: 6px; margin-bottom: 6px; font-size: 0.8rem;
}
.stProgress > div > div {
  background: linear-gradient(90deg, var(--primary), var(--primary-2)) !important;
}
*::-webkit-scrollbar { height: 8px; width: 8px; }
*::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--primary), var(--primary-2)); border-radius: 10px; }
section[data-testid="stSidebar"] {
  background: rgba(8, 12, 20, 0.75);
  backdrop-filter: blur(12px);
  border-right: 1px solid rgba(124,92,255,0.25);
}
</style>
"""
# -------------------- Main App Logic --------------------
st.set_page_config(page_title="EasyApply", page_icon="📄")
st.markdown(THEME_CSS, unsafe_allow_html=True)

# Hero banner
st.markdown("""
<div class="hero" style="margin:1.2rem">
  <h1>Easy-Apply <span style="opacity:.85">— AI Job Matching</span></h1>
  <p style="color:#b7c7dd;margin:0.4rem 0 0">Neon-precise parsing, smart scoring, and HR-grade evaluation.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state keys
for key in ['logged_in', 'username', 'role', 'user_id']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'logged_in' else False

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Google API Key", type="password", help="Needed for analysis.")
    models = {
        'extract': st.text_input("Extraction/AI Model", DEFAULT_GENAI_MODEL_EXTRACT),
        'embed': st.text_input("Embedding Model", DEFAULT_GENAI_MODEL_EMBED)
    }
    st.markdown("---")
    if st.session_state.logged_in:
        st.write(f"Logged in as **{st.session_state.username}** ({st.session_state.role})")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        st.header("User Access")
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        with login_tab:
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                user = _row_to_dict(db.get_user(login_user))
                if user and db.check_password(user['password_hash'], login_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = user['username']
                    st.session_state.role = user['role']
                    st.session_state.user_id = user['id']
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with signup_tab:
            signup_user = st.text_input("Username", key="signup_user")
            signup_pass = st.text_input("Password", type="password", key="signup_pass")
            signup_role = st.selectbox("Role", ["student", "tnp"], key="signup_role")
            if st.button("Sign Up"):
                if db.add_user(signup_user, signup_pass, signup_role):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already taken.")

# --- Views ---
def tnp_view(client, models):
    st.markdown('<div class="neon-panel neon-strong">', unsafe_allow_html=True)
    st.subheader("Post a New Job")
    with st.form("new_job_form", clear_on_submit=True):
        jd_file = st.file_uploader("Upload Job Description PDF", type="pdf")
        submitted = st.form_submit_button("Post Job")
        if submitted and jd_file:
            with st.spinner("Processing JD..."):
                jd_text = extract_text_from_pdf(jd_file)
                if jd_text:
                    jd_data = extract_structured_data(client, models['extract'], jd_text, 'jd')
                    if jd_data and isinstance(jd_data, dict) and jd_data.get("roles"):
                        posted = 0
                        for role in jd_data["roles"]:
                            title = role.get("job_title") or "Untitled Role"
                            db.add_job(st.session_state.user_id, title, jd_text, role)
                            posted += 1
                        st.success(f"Successfully posted {posted} role(s) from the JD.")
                    else:
                        st.error("Could not extract required details or roles from the JD.")
                else:
                    st.error("Could not read text from the uploaded JD PDF.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics
    jobs = _rows_to_dicts(db.get_tnp_jobs(st.session_state.user_id))
    total_apps = sum(len(_rows_to_dicts(db.get_applications_for_job(j["id"]))) for j in jobs)
    all_scores = []
    for j in jobs:
        all_scores.extend([a["relevance_score"] for a in _rows_to_dicts(db.get_applications_for_job(j["id"]))])
    avg_score = float(np.mean(all_scores)) if all_scores else 0.0

    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f'<div class="metric"><span>Jobs Posted</span><strong>{len(jobs)}</strong></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric"><span>Total Applications</span><strong>{total_apps}</strong></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric"><span>Avg Score</span><strong>{avg_score:.1f}%</strong></div>', unsafe_allow_html=True)

    st.subheader("Your Posted Jobs")
    for job in jobs:
        with st.expander(f"**{job['job_title']}**"):
            st.markdown('<div class="neon-panel">', unsafe_allow_html=True)
            applications = _rows_to_dicts(db.get_applications_for_job(job['id']))
            if not applications:
                st.write("No applications received yet.")
            else:
                for app in applications:
                    resume_data = json.loads(app['resume_data_json']) if isinstance(app.get('resume_data_json'), str) else app.get('resume_data_json', {})
                    st.markdown(f"**Applicant:** {app['username']} | **Score:** `{app['relevance_score']:.2f}%`")
                    st.progress(int(app['relevance_score']))
                    st.text(f"Predicted Role: {resume_data.get('predicted_role', 'N/A')}")
                    st.text(f"Skills: {', '.join(resume_data.get('skills', []))}")
                    st.markdown('<div class="chip">Application ID: {}</div>'.format(app.get("id","-")), unsafe_allow_html=True)
                    st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)

def student_view(client, models):
    st.subheader("Student Dashboard")
    st.markdown('<div class="neon-panel neon-strong">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Available Jobs", "My Applications"])
    with tab1:
        st.subheader("Apply for a Job")
        all_jobs = _rows_to_dicts(db.get_all_jobs())
        if not all_jobs:
            st.info("No jobs are available at the moment.")
        else:
            job_options = {f"{job['job_title']} (ID: {job['id']})": job['id'] for job in all_jobs}
            selected_job_str = st.selectbox("Select a Job", options=list(job_options.keys()))
            with st.form("application_form", clear_on_submit=True):
                resume_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
                if st.form_submit_button("Apply") and resume_file and selected_job_str:
                    with st.spinner("Analyzing your resume and submitting application..."):
                        job_id = job_options[selected_job_str]
                        job_details = next((j for j in all_jobs if j['id'] == job_id), None)
                        resume_text = extract_text_from_pdf(resume_file)
                        if resume_text and job_details:
                            resume_data = extract_structured_data(client, models['extract'], resume_text, 'resume')
                            jd_role = None
                            try:
                                jd_role = job_details.get("jd_data_json") or job_details.get("jd_structured") or job_details.get("jd_data") or None
                                if isinstance(jd_role, str):
                                    jd_role = json.loads(jd_role)
                                if isinstance(jd_role, dict) and "roles" in jd_role and isinstance(jd_role["roles"], list) and jd_role["roles"]:
                                    jd_role = jd_role["roles"][0]
                            except Exception:
                                jd_role = None
                            if not isinstance(jd_role, dict):
                                jd_role = {
                                    "job_title": job_details.get("job_title"),
                                    "required_skills": [],
                                    "responsibilities": [],
                                    "required_experience_years": None,
                                    "location": None,
                                    "schedule": None,
                                    "bond_months": None,
                                }
                            cache = st.session_state.setdefault("ai_score_cache", {})
                            final_score, parts = compute_final_score_with_ai(
                                client, models['embed'], models['extract'], jd_role, resume_text, resume_data or {}, cache
                            )
                            if resume_data:
                                if db.add_application(st.session_state.user_id, job_id, resume_text, resume_data, final_score):
                                    st.success(f"Application submitted successfully! Score: {final_score:.2f}")
                                    with st.expander("Score breakdown"):
                                        st.markdown('<div class="chip">Hard: {:.0f}</div>'.format(parts['hard']), unsafe_allow_html=True)
                                        st.markdown('<div class="chip">Semantic: {:.0f}</div>'.format(parts['semantic']), unsafe_allow_html=True)
                                        st.markdown('<div class="chip">Other: {:.0f}</div>'.format(parts['other']), unsafe_allow_html=True)
                                        st.markdown('<div class="chip">AI: {:.0f}</div>'.format(parts['ai']), unsafe_allow_html=True)
                                        if parts.get("ai_subscores"):
                                            st.json(parts["ai_subscores"])
                                else:
                                    st.warning("You have already applied for this job.")
                            else:
                                st.error("Could not extract structured data from your resume.")
                        else:
                            st.error("Failed to process resume or find job details.")
    with tab2:
        st.subheader("Your Submitted Applications")
        student_apps = _rows_to_dicts(db.get_student_applications(st.session_state.user_id))
        if not student_apps:
            st.info("You haven't applied for any jobs yet.")
        else:
            for app in student_apps:
                st.markdown(f"- **{app['job_title']}**: Your relevance score was calculated as `{app['relevance_score']:.2f}%`")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Page View Logic ---
if st.session_state.logged_in:
    try:
        client = _make_genai_client(api_key_input)
    except Exception as e:
        st.error(f"Gemini initialization error: {e}")
        st.stop()
    if st.session_state.role == 'tnp':
        tnp_view(client, models)
    else:
        student_view(client, models)
else:
    st.info("Please login or sign up using the sidebar to continue.")
