import os
import re
import io
import json
import numpy as np
import streamlit as st
import PyPDF2 as pdf
from google import genai
from google.genai import types

# Import database functions from the other file
import db

# -------------------- Config --------------------
DEFAULT_GENAI_MODEL_EXTRACT = "gemini-2.5-flash"  # text generation/parsing model [web:1][web:3]
DEFAULT_GENAI_MODEL_EMBED = "text-embedding-004"  # embeddings model via client.models.embed_content [web:1][web:3]

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
ROLE_SPLIT_RE = re.compile(r'(?:^|\n)\s*\d+\.\s+([^\n]+)\s*(?=\n)', flags=re.IGNORECASE)  # numbered headings [file:20]
# common aliases for normalization (extensible)
ALIASES = {
    "pyspark": "spark",
    "tableau": "tableau",
    "power bi": "power bi",
    "eda": "exploratory data analysis",
    "c++": "c++",
    "nlp": "nlp",
    "cv": "computer vision",
}  # skill mapping for dedupe [file:20]

def normalize_tokens(skills):
    out = set()
    for s in skills or []:
        k = (s or "").strip().lower()
        if not k:
            continue
        k = ALIASES.get(k, k)
        out.add(k)
    return sorted(out)  # canonical list for scoring [file:20]

def parse_numeric_constraints(text: str):
    t = (text or "").lower()
    stipend = None
    import re as _re
    # stipend like ₹5,000 per month
    m = _re.search(r'(?:₹|inr|\brs\.?)\s*([\d,]+)\s*(?:per\s*)?(month|mo|monthly)?', t)
    if m:
        amt = int(m.group(1).replace(',', ''))
        per = 'month' if (m.group(2) or '').startswith('mo') or (m.group(2) or '') == 'month' else None
        stipend = {"amount": amt, "currency": "INR", "period": per or "month"}
    # internship duration like 6 months
    internship_months = None
    m = _re.search(r'(\d+)\s*month', t)
    if m:
        internship_months = int(m.group(1))
    # bond months like 2.6 years or 30 months
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
    # batch year max like 2023 and earlier
    batch_year_max = None
    m = _re.search(r'(20\d{2}).*?(earlier|and earlier|or earlier)', t)
    if m:
        batch_year_max = int(m.group(1))
    return stipend, internship_months, bond_months, batch_year_max  # numeric fields to enrich LLM output [file:20]

def split_roles_heuristic(jd_text: str):
    # If the doc uses numbered sections, extract titles and segments
    titles = ROLE_SPLIT_RE.findall(jd_text or "")
    if not titles:
        return [{"title": None, "text": jd_text or ""}]
    # Split on the same boundaries
    parts = re.split(r'(?:^|\n)\s*\d+\.\s+[^\n]+\s*(?=\n)', jd_text)
    # parts[0] is preamble; align titles with subsequent parts
    roles = []
    for title, seg in zip(titles, parts[1:]):
        roles.append({"title": title.strip(), "text": seg.strip()})
    return roles  # list of role blocks [file:20]

# -------------------- Gemini client --------------------
def _make_genai_client(api_key: str | None) -> genai.Client:
    if api_key and api_key.strip():
        return genai.Client(api_key=api_key.strip())  # explicit key path [web:7]
    env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not env_key:
        raise RuntimeError("Google API key not set. Provide it in the sidebar or via GOOGLE_API_KEY/GEMINI_API_KEY.")  # key guidance [web:7]
    return genai.Client()  # env autodetection [web:7]

# -------------------- Gemini wrappers (Client-based) --------------------
def _genai_generate_text(client: genai.Client, model: str, prompt: str) -> str:
    resp = client.models.generate_content(model=model, contents=prompt)  # modern call [web:1][web:3]
    return getattr(resp, "text", None) or ""  # aggregated text field [web:1][web:3]

def _genai_embed_text(client: genai.Client, model: str, text: str) -> list[float]:
    eresp = client.models.embed_content(model=model, contents=text)  # embeddings endpoint [web:1][web:3]
    if hasattr(eresp, "embeddings") and eresp.embeddings:
        emb0 = eresp.embeddings[0]
        vec = getattr(emb0, "values", None) or getattr(emb0, "embedding", None)
        if vec is not None:
            return list(vec)
    try:
        return list(eresp["embeddings"][0]["values"])
    except Exception:
        pass
    raise RuntimeError("Unexpected embeddings response shape from Gemini API.")  # response shape guard [web:1][web:3]

# -------------------- LLM and Prompt Engineering for Structured Data Extraction --------------------
def extract_structured_data(client: genai.Client, model_name: str, text: str, data_type: str):
    if data_type == "resume":
        prompt = f"""
Return RFC 8259–compliant JSON only (no commentary, no markdown).
Resume: --- {text} ---
Schema: {{"name": "string|null", "experience": [{{"company": "string|null", "role": "string|null"}}], "education": [{{"institute_name": "string|null", "degree": "string|null", "specialization": "string|null"}}], "skills": ["string"], "predicted_role": "string|null"}}
Output: JSON only. First char '{{', last char '}}'.
""".strip()  # strict JSON policy [web:1][web:3]
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
""".strip()  # multi-role schema for bullet-heavy JDs [file:20]
    else:
        return None  # unsupported data_type [web:1]

    try:
        raw = _genai_generate_text(client, model_name, prompt)  # generation call [web:1][web:3]
        candidate = _extract_outer_json_block(raw)  # fence cleanup [web:1]
        try:
            data = json.loads(candidate)  # strict JSON load [web:1]
        except json.JSONDecodeError:
            retry_prompt = prompt + "\nReminder: JSON ONLY. Start with '{' and end with '}'. No prose."  # terse retry [web:1]
            raw2 = _genai_generate_text(client, model_name, retry_prompt)  # second attempt [web:1]
            candidate2 = _extract_outer_json_block(raw2)  # cleanup [web:1]
            data = json.loads(candidate2)  # parse [web:1]
        # Post-process: normalize skills and enrich numeric constraints if missing
        if data_type == "jd":
            roles = data.get("roles") if isinstance(data, dict) else None  # roles array [file:20]
            if not roles:
                # heuristic fallback split if model returned single body
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
                data = {"roles": roles}  # heuristic packaging [file:20]
            # normalize skills tokens
            for r in data.get("roles", []):
                r["required_skills"] = normalize_tokens(r.get("required_skills") or [])  # canonical skills [file:20]
        return data  # structured output [web:1]
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Error processing document with Gemini: {e}")  # user feedback [web:1]
        return None

# -------------------- Semantic Similarity Calculation & PDF Extraction --------------------
def calculate_semantic_similarity(client: genai.Client, embedding_model_name: str, jd_text: str, resume_text: str) -> float:
    try:
        jd_vec = np.array(_genai_embed_text(client, embedding_model_name, jd_text), dtype=np.float32)  # embed JD [web:1][web:3]
        resume_vec = np.array(_genai_embed_text(client, embedding_model_name, resume_text), dtype=np.float32)  # embed resume [web:1][web:3]
        denom = (np.linalg.norm(jd_vec) * np.linalg.norm(resume_vec))
        if denom == 0:
            return 0.0  # zero-vector guard [web:1]
        cos_sim = float(np.dot(jd_vec, resume_vec) / denom)  # cosine [web:1]
        return max(0.0, cos_sim) * 100.0  # 0-100% [web:1]
    except Exception as e:
        st.error(f"Could not calculate semantic similarity: {e}")  # error path [web:1]
        return 0.0

def extract_text_from_pdf(pdf_file_object):
    try:
        pdf_reader = pdf.PdfReader(pdf_file_object)  # PyPDF2 reader [file:20]
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)  # concatenate pages [file:20]
    except Exception as e:
        st.error(f"Error reading a PDF file: {e}")  # PDF error [file:20]
        return None

# -------------------- UI Views / Pages --------------------
def tnp_view(client, models):
    st.header("TnP Dashboard")  # section header [file:20]
    st.subheader("Post a New Job")  # JD posting [file:20]
    with st.form("new_job_form", clear_on_submit=True):
        jd_file = st.file_uploader("Upload Job Description PDF", type="pdf")  # JD upload [file:20]
        submitted = st.form_submit_button("Post Job")  # submit [file:20]
        if submitted and jd_file:
            with st.spinner("Processing JD..."):
                jd_text = extract_text_from_pdf(jd_file)  # extract text [file:20]
                if jd_text:
                    jd_data = extract_structured_data(client, models['extract'], jd_text, 'jd')  # multi-role extraction [web:1]
                    if jd_data and isinstance(jd_data, dict) and jd_data.get("roles"):
                        # Optional: let TnP select roles to post; here we post all roles
                        posted = 0
                        for role in jd_data["roles"]:
                            title = role.get("job_title") or "Untitled Role"  # default title [file:20]
                            db.add_job(st.session_state.user_id, title, jd_text, role)  # store each role [file:20]
                            posted += 1
                        st.success(f"Successfully posted {posted} role(s) from the JD.")  # feedback [file:20]
                    else:
                        st.error("Could not extract required details or roles from the JD.")  # failure [file:20]
                else:
                    st.error("Could not read text from the uploaded JD PDF.")  # PDF failure [file:20]

    st.subheader("Your Posted Jobs")  # listing [file:20]
    for job in db.get_tnp_jobs(st.session_state.user_id):
        with st.expander(f"**{job['job_title']}**"):
            st.markdown("---")  # divider [file:20]
            applications = db.get_applications_for_job(job['id'])  # fetch apps [file:20]
            if not applications:
                st.write("No applications received yet.")  # empty state [file:20]
                continue
            for app in applications:
                resume_data = json.loads(app['resume_data_json'])  # parsed resume [file:20]
                st.markdown(f"**Applicant:** {app['username']} | **Score:** `{app['relevance_score']:.2f}%`")  # summary [file:20]
                st.progress(int(app['relevance_score']))  # progress bar [file:20]
                st.text(f"Predicted Role: {resume_data.get('predicted_role', 'N/A')}")  # role hint [file:20]
                st.text(f"Skills: {', '.join(resume_data.get('skills', []))}")  # skills list [file:20]
                st.markdown("---")  # separator [file:20]

def student_view(client, models):
    st.header("Student Dashboard")  # header [file:20]
    tab1, tab2 = st.tabs(["Available Jobs", "My Applications"])  # tabs [file:20]
    with tab1:
        st.subheader("Apply for a Job")  # apply section [file:20]
        all_jobs = db.get_all_jobs()  # pull jobs [file:20]
        if not all_jobs:
            st.info("No jobs are available at the moment.")  # empty state [file:20]
            return
        job_options = {f"{job['job_title']} (ID: {job['id']})": job['id'] for job in all_jobs}  # selector [file:20]
        selected_job_str = st.selectbox("Select a Job", options=job_options.keys())  # dropdown [file:20]
        with st.form("application_form", clear_on_submit=True):
            resume_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")  # resume upload [file:20]
            if st.form_submit_button("Apply") and resume_file and selected_job_str:
                with st.spinner("Analyzing your resume and submitting application..."):
                    job_id = job_options[selected_job_str]  # selected id [file:20]
                    job_details = next((j for j in all_jobs if j['id'] == job_id), None)  # job fetch [file:20]
                    resume_text = extract_text_from_pdf(resume_file)  # resume text [file:20]
                    if resume_text and job_details:
                        resume_data = extract_structured_data(client, models['extract'], resume_text, 'resume')  # resume JSON [web:1]
                        # Optional hybrid score: semantic + skill overlap if JD role has skills
                        score = calculate_semantic_similarity(client, models['embed'], job_details['jd_text'], resume_text)  # base score [web:1]
                        if resume_data:
                            if db.add_application(st.session_state.user_id, job_id, resume_text, resume_data, score):
                                st.success("Application submitted successfully!")  # success [file:20]
                            else:
                                st.warning("You have already applied for this job.")  # duplicate [file:20]
                        else:
                            st.error("Could not extract structured data from your resume.")  # parse fail [file:20]
                    else:
                        st.error("Failed to process resume or find job details.")  # generic error [file:20]
    with tab2:
        st.subheader("Your Submitted Applications")  # history [file:20]
        student_apps = db.get_student_applications(st.session_state.user_id)  # fetch [file:20]
        if not student_apps:
            st.info("You haven't applied for any jobs yet.")  # empty [file:20]
        else:
            for app in student_apps:
                st.markdown(f"- **{app['job_title']}**: Your relevance score was calculated as `{app['relevance_score']:.2f}%`")  # item [file:20]

# -------------------- Main App Logic --------------------
st.set_page_config(page_title="Job Portal", page_icon="📄")  # page meta [file:20]
st.title("📄 AI-Powered Job Portal")  # title [file:20]

# Initialize session state keys
for key in ['logged_in', 'username', 'role', 'user_id']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'logged_in' else False  # session init [file:20]

with st.sidebar:
    st.header("Configuration")  # sidebar [file:20]
    api_key_input = st.text_input("Google API Key", type="password", help="Needed for analysis.")  # key input [web:7]
    models = {
        'extract': st.text_input("Extraction Model", DEFAULT_GENAI_MODEL_EXTRACT),  # model control [web:1]
        'embed': st.text_input("Embedding Model", DEFAULT_GENAI_MODEL_EMBED)  # embedding control [web:1]
    }
    st.markdown("---")  # divider [file:20]
    if st.session_state.logged_in:
        st.write(f"Logged in as **{st.session_state.username}** ({st.session_state.role})")  # status [file:20]
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]  # clear session [file:20]
            st.rerun()  # reload [file:20]
    else:
        st.header("User Access")  # auth [file:20]
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])  # tabs [file:20]
        with login_tab:
            login_user = st.text_input("Username", key="login_user")  # user [file:20]
            login_pass = st.text_input("Password", type="password", key="login_pass")  # pass [file:20]
            if st.button("Login"):
                user = db.get_user(login_user)  # lookup [file:20]
                if user and db.check_password(user['password_hash'], login_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = user['username']
                    st.session_state.role = user['role']
                    st.session_state.user_id = user['id']
                    st.rerun()  # redirect [file:20]
                else:
                    st.error("Invalid username or password.")  # failure [file:20]
        with signup_tab:
            signup_user = st.text_input("Username", key="signup_user")  # uname [file:20]
            signup_pass = st.text_input("Password", type="password", key="signup_pass")  # pwd [file:20]
            signup_role = st.selectbox("Role", ["student", "tnp"], key="signup_role")  # role [file:20]
            if st.button("Sign Up"):
                if db.add_user(signup_user, signup_pass, signup_role):
                    st.success("Account created! Please login.")  # success [file:20]
                else:
                    st.error("Username already taken.")  # duplicate [file:20]

# --- Main Page View Logic ---
if st.session_state.logged_in:
    try:
        client = _make_genai_client(api_key_input)  # client init [web:7]
        if st.session_state.role == 'tnp':
            tnp_view(client, models)  # TnP view [file:20]
        else:
            student_view(client, models)  # student view [file:20]
    except Exception as e:
        st.error(f"Initialization Error: {e}. Please check your API key in the sidebar.")  # error [web:7]
else:
    st.info("Please login or sign up using the sidebar to continue.")  # call-to-action [file:20]
