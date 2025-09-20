import os
import re
import io
import json
import numpy as np
import streamlit as st
import PyPDF2 as pdf
from google import generativeai as genai

# Import database functions from the other file
import db

# -------------------- Config --------------------
DEFAULT_GENAI_MODEL_EXTRACT = "gemini-2.5-flash"
DEFAULT_GENAI_MODEL_EMBED = "text-embedding-004"


# -------------------- Helpers: Robust JSON cleanup & validation --------------------
_PREFIX_RE = re.compile(r'^\s*(?:```)', flags=re.IGNORECASE)

def _extract_outer_json_block(s: str) -> str:
    """
    Best-effort cleanup:
      - strip whitespace and code fences
      - remove common prose prefixes ("Here is the JSON:")
      - keep substring from first '{' to last '}'
    Returns a candidate JSON string (may still be invalid).
    """
    if not s: return s
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("` \n")
    if not (t.startswith("{") and t.endswith("}")):
        t = _PREFIX_RE.sub("", t).strip()
    first = t.find("{")
    last = t.rfind("}")
    return t[first:last+1] if first != -1 and last != -1 and first < last else t


# -------------------- Gemini wrappers --------------------
def _make_genai_client(api_key: str | None):
    """Initializes the Gemini client with the provided API key."""
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Google API key not set. Provide GOOGLE_API_KEY env var or the sidebar input.")
    genai.configure(api_key=api_key)


def _genai_generate_text(model: str, prompt: str) -> str:
    """Calls Gemini to generate text. Returns the plain text string."""
    model_instance = genai.GenerativeModel(model)
    resp = model_instance.generate_content(prompt)
    return getattr(resp, "text", None) or ""


def _genai_embed_text(model: str, text: str) -> list[float]:
    """Calls Gemini embeddings model; returns a vector list[float]."""
    eresp = genai.embed_content(model=model, content=text)
    return list(eresp['embedding'])


# -------------------- LLM and Prompt Engineering for Structured Data Extraction --------------------
def extract_structured_data(model_name: str, text: str, data_type: str):
    """
    Sends text to Gemini to extract structured data (JSON).
    Includes strict JSON cleanup/validation and a single retry with a terse reminder.
    """
    if data_type == "resume":
        prompt = f"""
Act as an expert resume parser. Return RFC 8259–compliant JSON only (no commentary, no markdown).
Resume: --- {text} ---
Schema: {{"name": "string|null", "experience": [{{"company": "string|null", "role": "string|null"}}], "education": [{{"institute_name": "string|null", "degree": "string|null", "specialization": "string|null"}}], "skills": ["string"], "predicted_role": "string|null"}}
Output: JSON only. First char '{{', last char '}}'.
"""
    elif data_type == "jd":
        prompt = f"""
You are a deterministic JSON generator.
Output policy:
- Emit RFC 8259–compliant JSON only. No prose, no code fences, no comments.
- The very first character MUST be '{{' and the very last MUST be '}}'.

Input JD:
---
{text}
---

Extraction rules:
- Infer missing fields when strongly implied; else use null for scalars and [] for arrays.
- Normalize required_skills to lowercase and deduplicate.
- Parse years of experience from ranges/phrases (e.g., "3-5 years" -> 3).
- Map education to "bachelor's", "master's", "phd", or null when unclear.

Schema to follow exactly:
{{
  "job_title": "string|null",
  "required_skills": ["string"],
  "required_experience_years": "integer|null",
  "required_education": "bachelor's"|"master's"|"phd"|"string|null"
}}
"""
    else:
        return None

    try:
        # Attempt 1
        raw = _genai_generate_text(model_name, prompt)
        candidate = _extract_outer_json_block(raw)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Attempt 2 with terse reminder
            retry_prompt = prompt + "\nReminder: JSON ONLY. Start with '{' and end with '}'. No prose."
            raw2 = _genai_generate_text(model_name, retry_prompt)
            candidate2 = _extract_outer_json_block(raw2)
            return json.loads(candidate2)
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Error processing document with Gemini: {e}")
        return None

# -------------------- Semantic Similarity Calculation & PDF Extraction --------------------
def calculate_semantic_similarity(embedding_model_name: str, jd_text: str, resume_text: str) -> float:
    """Calculates cosine similarity between two texts using Gemini embeddings."""
    try:
        jd_vec = np.array(_genai_embed_text(embedding_model_name, jd_text), dtype=np.float32)
        resume_vec = np.array(_genai_embed_text(embedding_model_name, resume_text), dtype=np.float32)
        cos_sim = float(np.dot(jd_vec, resume_vec) / (np.linalg.norm(jd_vec) * np.linalg.norm(resume_vec)))
        return max(0.0, cos_sim) * 100.0
    except Exception as e:
        st.error(f"Could not calculate semantic similarity: {e}")
        return 0.0

def extract_text_from_pdf(pdf_file_object):
    """Reads an uploaded PDF file object and extracts the text from it."""
    try:
        pdf_reader = pdf.PdfReader(pdf_file_object)
        return "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except Exception as e:
        st.error(f"Error reading a PDF file: {e}")
        return None

# -------------------- UI Views / Pages --------------------
def tnp_view(models):
    """Renders the UI for the Training and Placement Officer role."""
    st.header("TnP Dashboard")
    st.subheader("Post a New Job")
    with st.form("new_job_form", clear_on_submit=True):
        jd_file = st.file_uploader("Upload Job Description PDF", type="pdf")
        submitted = st.form_submit_button("Post Job")
        if submitted and jd_file:
            with st.spinner("Processing JD..."):
                jd_text = extract_text_from_pdf(jd_file)
                if jd_text:
                    jd_data = extract_structured_data(models['extract'], jd_text, 'jd')
                    if jd_data and jd_data.get("job_title"):
                        db.add_job(st.session_state.user_id, jd_data["job_title"], jd_text, jd_data)
                        st.success(f"Successfully posted job: {jd_data['job_title']}")
                    else: st.error("Could not extract required details from the JD.")
                else: st.error("Could not read text from the uploaded JD PDF.")

    st.subheader("Your Posted Jobs")
    for job in db.get_tnp_jobs(st.session_state.user_id):
        with st.expander(f"**{job['job_title']}**"):
            st.markdown("---")
            applications = db.get_applications_for_job(job['id'])
            if not applications:
                st.write("No applications received yet.")
                continue
            for app in applications:
                resume_data = json.loads(app['resume_data_json'])
                st.markdown(f"**Applicant:** {app['username']} | **Score:** `{app['relevance_score']:.2f}%`")
                st.progress(int(app['relevance_score']))
                st.text(f"Predicted Role: {resume_data.get('predicted_role', 'N/A')}")
                st.text(f"Skills: {', '.join(resume_data.get('skills', []))}")
                st.markdown("---")

def student_view(models):
    """Renders the UI for the Student role."""
    st.header("Student Dashboard")
    tab1, tab2 = st.tabs(["Available Jobs", "My Applications"])
    with tab1:
        st.subheader("Apply for a Job")
        all_jobs = db.get_all_jobs()
        if not all_jobs:
            st.info("No jobs are available at the moment.")
            return
        job_options = {f"{job['job_title']} (ID: {job['id']})": job['id'] for job in all_jobs}
        selected_job_str = st.selectbox("Select a Job", options=job_options.keys())
        with st.form("application_form", clear_on_submit=True):
            resume_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
            if st.form_submit_button("Apply") and resume_file and selected_job_str:
                with st.spinner("Analyzing your resume and submitting application..."):
                    job_id = job_options[selected_job_str]
                    job_details = next((j for j in all_jobs if j['id'] == job_id), None)
                    resume_text = extract_text_from_pdf(resume_file)
                    if resume_text and job_details:
                        resume_data = extract_structured_data(models['extract'], resume_text, 'resume')
                        score = calculate_semantic_similarity(models['embed'], job_details['jd_text'], resume_text)
                        if resume_data:
                            if db.add_application(st.session_state.user_id, job_id, resume_text, resume_data, score):
                                st.success("Application submitted successfully!")
                            else: st.warning("You have already applied for this job.")
                        else: st.error("Could not extract structured data from your resume.")
                    else: st.error("Failed to process resume or find job details.")
    with tab2:
        st.subheader("Your Submitted Applications")
        student_apps = db.get_student_applications(st.session_state.user_id)
        if not student_apps:
            st.info("You haven't applied for any jobs yet.")
        else:
            for app in student_apps:
                st.markdown(f"- **{app['job_title']}**: Your relevance score was calculated as `{app['relevance_score']:.2f}%`")

# -------------------- Main App Logic --------------------
st.set_page_config(page_title="Job Portal", page_icon="📄")
st.title("📄 AI-Powered Job Portal")

# Initialize session state keys
for key in ['logged_in', 'username', 'role', 'user_id']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'logged_in' else False

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Google API Key", type="password", help="Needed for analysis.")
    models = {
        'extract': st.text_input("Extraction Model", DEFAULT_GENAI_MODEL_EXTRACT),
        'embed': st.text_input("Embedding Model", DEFAULT_GENAI_MODEL_EMBED)
    }
    st.markdown("---")
    if st.session_state.logged_in:
        st.write(f"Logged in as **{st.session_state.username}** ({st.session_state.role})")
        if st.button("Logout"):
            # Clear all session state on logout
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    else:
        st.header("User Access")
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        with login_tab:
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                user = db.get_user(login_user)
                if user and db.check_password(user['password_hash'], login_pass):
                    st.session_state.logged_in = True
                    st.session_state.username = user['username']
                    st.session_state.role = user['role']
                    st.session_state.user_id = user['id']
                    st.rerun()
                else: st.error("Invalid username or password.")
        with signup_tab:
            signup_user = st.text_input("Username", key="signup_user")
            signup_pass = st.text_input("Password", type="password", key="signup_pass")
            signup_role = st.selectbox("Role", ["student", "tnp"], key="signup_role")
            if st.button("Sign Up"):
                if db.add_user(signup_user, signup_pass, signup_role):
                    st.success("Account created! Please login.")
                else: st.error("Username already taken.")

# --- Main Page View Logic ---
if st.session_state.logged_in:
    try:
        _make_genai_client(api_key_input)
        if st.session_state.role == 'tnp':
            tnp_view(models)
        else:
            student_view(models)
    except Exception as e:
        st.error(f"Initialization Error: {e}. Please check your API key in the sidebar.")
else:
    st.info("Please login or sign up using the sidebar to continue.")

