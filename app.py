import os
import re
import io
import json
import time
import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import PyPDF2 as pdf
from datetime import datetime
from google import genai
from google.genai import types

# -------- DB functions module --------
import db

# -------------------- Config --------------------
DEFAULT_GENAI_MODEL_EXTRACT = "gemini-2.5-flash"
DEFAULT_GENAI_MODEL_EMBED = "text-embedding-004"

# -------------------- JSON cleanup --------------------
_PREFIX_RE = re.compile(r'^\s*(?:```)', flags=re.IGNORECASE)

def _extract_outer_json_block(s: str) -> str:
    if not s: return s
    t = s.strip()
    if t.startswith("```"): t = t.strip("` \n")
    if not (t.startswith("{") and t.endswith("}")): t = _PREFIX_RE.sub("", t).strip()
    first = t.find("{"); last = t.rfind("}")
    return t[first:last+1] if first != -1 and last != -1 and first < last else t

# -------------------- JD utilities --------------------
ROLE_SPLIT_RE = re.compile(r'(?:^|\n)\s*\d+\.\s+([^\n]+)\s*(?=\n)', flags=re.IGNORECASE)
ALIASES = {"pyspark":"spark","tableau":"tableau","power bi":"power bi","eda":"exploratory data analysis","c++":"c++","nlp":"nlp","cv":"computer vision"}
def normalize_tokens(skills):
    out=set()
    for s in skills or []:
        k=(s or "").strip().lower()
        if not k: continue
        k=ALIASES.get(k,k); out.add(k)
    return sorted(out)
def parse_numeric_constraints(text:str):
    t=(text or "").lower(); stipend=None
    import re as _re
    m=_re.search(r'(?:₹|inr|\brs\.?)\s*([\d,]+)\s*(?:per\s*)?(month|mo|monthly)?',t)
    if m:
        amt=int(m.group(1).replace(',','')); per='month' if (m.group(2) or '').startswith('mo') or (m.group(2) or '')=='month' else None
        stipend={"amount":amt,"currency":"INR","period":per or "month"}
    internship_months=None; m=_re.search(r'(\d+)\s*month',t)
    if m: internship_months=int(m.group(1))
    bond_months=None; m=_re.search(r'(\d+(?:\.\d+)?)\s*year',t)
    if m:
        try: bond_months=int(round(float(m.group(1))*12))
        except: pass
    m2=_re.search(r'(\d+)\s*month',t)
    if m2 and bond_months is None: bond_months=int(m2.group(1))
    batch_year_max=None; m=_re.search(r'(20\d{2}).*?(earlier|and earlier|or earlier)',t)
    if m: batch_year_max=int(m.group(1))
    return stipend, internship_months, bond_months, batch_year_max
def split_roles_heuristic(jd_text:str):
    titles=ROLE_SPLIT_RE.findall(jd_text or "")
    if not titles: return [{"title":None,"text":jd_text or ""}]
    parts=re.split(r'(?:^|\n)\s*\d+\.\s+[^\n]+\s*(?=\n)', jd_text)
    return [{"title":t.strip(),"text":seg.strip()} for t,seg in zip(titles, parts[1:])]

# -------------------- sqlite row helpers --------------------
def _row_to_dict(row):
    try: return dict(row)
    except Exception:
        try: return {k:row[k] for k in row.keys()}
        except Exception: return row
def _rows_to_dicts(rows): return [_row_to_dict(r) for r in rows or []]

# -------------------- Gemini client --------------------
def _make_genai_client(api_key: str | None) -> genai.Client:
    if api_key and api_key.strip(): return genai.Client(api_key=api_key.strip())
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise RuntimeError("Google API key not set. Provide GOOGLE_API_KEY/GEMINI_API_KEY or sidebar input.")
    return genai.Client()

# -------------------- Gemini wrappers --------------------
def _genai_generate_text(client: genai.Client, model: str, prompt: str) -> str:
    resp=client.models.generate_content(model=model, contents=prompt); return getattr(resp,"text",None) or ""
def _genai_embed_text(client: genai.Client, model: str, text: str) -> list[float]:
    eresp=client.models.embed_content(model=model, contents=text)
    if hasattr(eresp,"embeddings") and eresp.embeddings:
        emb0=eresp.embeddings[0]; vec=getattr(emb0,"values",None) or getattr(emb0,"embedding",None)
        if vec is not None: return list(vec)
    try: return list(eresp["embeddings"][0]["values"])
    except Exception: pass
    raise RuntimeError("Unexpected embeddings response shape.")

# -------------------- Extraction --------------------
def extract_structured_data(client: genai.Client, model_name: str, text: str, data_type: str):
    if data_type=="resume":
        prompt=f"""
Return RFC 8259–compliant JSON only (no commentary, no markdown).
Resume: --- {text} ---
Schema: {{"name":"string|null","experience":[{{"company":"string|null","role":"string|null"}}],"education":[{{"institute_name":"string|null","degree":"string|null","specialization":"string|null"}}],"skills":["string"],"predicted_role":"string|null"}}
Output: JSON only. First char '{{', last char '}}'.
""".strip()
    elif data_type=="jd":
        prompt=f"""
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
      "job_title":"string|null",
      "location":"string|null",
      "employment_type":["string"],
      "internship_months":"integer|null",
      "bond_months":"integer|null",
      "stipend":{{"amount":"integer|null","currency":"string|null","period":"string|null"}},
      "schedule":"string|null",
      "required_skills":["string"],
      "responsibilities":["string"],
      "required_experience_years":"integer|null",
      "batch_year_max":"integer|null",
      "required_education":["string"]|"string|null"
    }}
  ]
}}
""".strip()
    else:
        return None
    try:
        raw=_genai_generate_text(client, model_name, prompt); candidate=_extract_outer_json_block(raw)
        try: data=json.loads(candidate)
        except json.JSONDecodeError:
            raw2=_genai_generate_text(client, model_name, prompt+"\nReminder: JSON ONLY. Start with '{' and end with '}'. No prose.")
            candidate2=_extract_outer_json_block(raw2); data=json.loads(candidate2)
        if data_type=="jd":
            roles=data.get("roles") if isinstance(data,dict) else None
            if not roles:
                blocks=split_roles_heuristic(text); roles=[]
                for b in blocks:
                    stipend,intern_m,bond_m,batch_max=parse_numeric_constraints(b["text"])
                    roles.append({"job_title":b["title"] or None,"location":None,"employment_type":[],
                                  "internship_months":intern_m,"bond_months":bond_m,"stipend":stipend,"schedule":None,
                                  "required_skills":[],"responsibilities":[],"required_experience_years":0 if "intern" in (b["title"] or "").lower() else None,
                                  "batch_year_max":batch_max,"required_education":None})
                data={"roles":roles}
            for r in data.get("roles",[]): r["required_skills"]=normalize_tokens(r.get("required_skills") or [])
        return data
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Error processing document with Gemini: {e}"); return None

# -------------------- Scoring --------------------
def cosine_to_pct(c: float) -> float:
    from math import isnan
    if c is None or isnan(c): return 0.0
    return max(0.0, min(1.0, (c-0.2)/0.6))*100.0
def extract_years_experience(text:str)->float:
    t=(text or "").lower(); yrs=0.0; m=re.search(r'(\d+(?:\.\d+)?)\s*[\+\-]?\s*year',t)
    if m:
        try: yrs=float(m.group(1))
        except: pass
    return yrs
def normalize_list(xs): return sorted({(x or "").strip().lower() for x in xs or [] if (x or "").strip()})
def hard_match_score(jd_role:dict,resume_data:dict)->float:
    req=normalize_list(jd_role.get("required_skills")); have=normalize_list(resume_data.get("skills"))
    if not req: return 0.0
    resp=" ".join(jd_role.get("responsibilities") or []).lower()
    weights={s:(1.25 if s and s in resp else 1.0) for s in req}
    num=sum(weights[s] for s in req if s in have); den=sum(weights.values()); return 100.0*(num/den) if den>0 else 0.0
def other_factors_score(jd_role:dict,resume_data:dict,resume_text:str)->float:
    score=0.0; edu_req=jd_role.get("required_education"); edu_req_set=set()
    if isinstance(edu_req,list): edu_req_set={e.lower() for e in edu_req if isinstance(e,str)}
    elif isinstance(edu_req,str) and edu_req: edu_req_set={edu_req.lower()}
    edu_have=[(e.get("degree") or "").lower() for e in (resume_data.get("education") or [])]
    if edu_req_set:
        if any(any(req in deg for req in edu_req_set) for deg in edu_have): score+=8
        elif edu_have: score+=4
    req_yrs=jd_role.get("required_experience_years"); have_yrs=extract_years_experience(resume_text)
    if req_yrs is not None:
        if req_yrs<=1:
            if have_yrs<=1: score+=6
            elif have_yrs<=2: score+=3
        else:
            if have_yrs>=req_yrs: score+=6
            elif have_yrs>=max(0,req_yrs-1): score+=3
    text=(resume_text or "").lower(); loc=(jd_role.get("location") or "").lower()
    if loc and (loc in text or "relocation" in text or "remote" in text): score+=3
    sched=(jd_role.get("schedule") or "").lower()
    if sched and ("day shift" in sched or "monday to friday" in sched):
        score+=3 if ("immediately available" in text or "join immediately" in text) else 1
    if jd_role.get("bond_months") and ("bond" in text or "service agreement" in text or "commitment" in text): score+=2
    return min(20.0, score)
def compute_semantic_match_pct(client, embed_model, jd_text, resume_text)->float:
    try:
        jd_vec=np.array(_genai_embed_text(client, embed_model, jd_text),dtype=np.float32)
        rs_vec=np.array(_genai_embed_text(client, embed_model, resume_text),dtype=np.float32)
        denom=np.linalg.norm(jd_vec)*np.linalg.norm(rs_vec)
        if denom==0: return 0.0
        c=float(np.dot(jd_vec, rs_vec)/denom); return cosine_to_pct(c)
    except Exception: return 0.0

# -------------------- AI rubric scoring --------------------
def _cache_key(jd_role:dict,resume_text:str)->str:
    blob=json.dumps(jd_role,sort_keys=True)+"||"+(resume_text or ""); return hashlib.sha256(blob.encode("utf-8")).hexdigest()
def ai_rubric_score(client, model, jd_role:dict, resume_text:str)->dict:
    schema={"ai_score":"integer (0-100)","subscores":{"skills":"integer (0-40)","experience":"integer (0-20)","education":"integer (0-15)","projects":"integer (0-15)","constraints":"integer (0-10)"},"explanations":{"skills_missing":["string"],"strong_matches":["string"]}}
    prompt=f"""
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
    raw=_genai_generate_text(client, model, prompt); candidate=_extract_outer_json_block(raw)
    try: data=json.loads(candidate)
    except json.JSONDecodeError:
        raw2=_genai_generate_text(client, model, prompt+"\nReminder: JSON ONLY."); candidate2=_extract_outer_json_block(raw2); data=json.loads(candidate2)
    ai=int(max(0,min(100,int(data.get("ai_score",0))))); subs=data.get("subscores",{}) or {}
    return {"ai_score":ai,"subscores":subs}
def compute_final_score_with_ai(client, embed_model, llm_model, jd_role, resume_text, resume_data, cache:dict):
    hard=hard_match_score(jd_role,resume_data)
    jd_for_sem=" ".join([jd_role.get("job_title") or "", " ".join(jd_role.get("responsibilities") or []), " ".join(jd_role.get("required_skills") or [])])
    sem=compute_semantic_match_pct(client, embed_model, jd_for_sem, resume_text)
    other=other_factors_score(jd_role,resume_data,resume_text)
    key=_cache_key(jd_role,resume_text)
    ai_dict=cache.get(key) or ai_rubric_score(client, llm_model, jd_role, resume_text); cache[key]=ai_dict
    ai=ai_dict["ai_score"]
    final=0.35*hard + 0.15*sem + 0.10*other + 0.40*ai
    return round(final,2), {"hard":hard,"semantic":sem,"other":other,"ai":ai,"ai_subscores":ai_dict.get("subscores",{})}

# -------------------- PDF --------------------
def extract_text_from_pdf(pdf_file_object):
    try:
        pdf_reader=pdf.PdfReader(pdf_file_object); return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error reading a PDF file: {e}"); return None

# -------------------- THEME --------------------
THEME_CSS = """
<style>
:root { --bg:#0b0f17; --panel:rgba(17,25,40,0.7); --panel-strong:rgba(17,25,40,0.9); --text:#e5f1ff; --muted:#9bb0c8; --primary:#7c5cff; --primary-2:#00e1ff; --accent:#22d3ee; --glow:0 0 24px rgba(124,92,255,0.35), 0 0 48px rgba(0,225,255,0.15); --radius:16px; }
html, body, .stApp { background: radial-gradient(1200px 700px at 10% 10%, rgba(124,92,255,0.08), transparent 50%), radial-gradient(1000px 500px at 90% 30%, rgba(0,225,255,0.08), transparent 55%), linear-gradient(180deg, #0b0f17 0%, #080b12 100%) !important; color: var(--text); }
.block-container { padding-top: 2rem !important; }
.neon-panel { background: var(--panel); border: 1px solid rgba(124,92,255,0.18); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03), var(--glow); backdrop-filter: blur(14px) saturate(120%); border-radius: var(--radius); padding: 1.2rem 1.2rem; margin:1.2rem; }
.neon-strong { background: var(--panel-strong); }
.hero { border-radius: 20px; padding: 1.25rem 1.5rem; background: linear-gradient(135deg, rgba(124,92,255,0.22), rgba(0,225,255,0.18)); border: 1px solid rgba(255,255,255,0.12); box-shadow: 0 24px 60px rgba(0,0,0,0.35), 0 0 50px rgba(124,92,255,0.25) inset; position: relative; overflow: hidden; }
.hero:before { content:""; position:absolute; top:-40%; left:-10%; width:120%; height:180%; background: radial-gradient(closest-side, rgba(255,255,255,0.08), transparent 60%); filter: blur(40px); transform: rotate(15deg); animation: drift 14s linear infinite; }
.stButton>button { background: linear-gradient(135deg, var(--primary), var(--primary-2)); color: white; border: 0; border-radius: 12px; padding: 0.6rem 1rem; box-shadow: var(--glow); transition: transform .08s ease, box-shadow .2s ease; }
.stButton>button:hover { transform: translateY(-1px) scale(1.01); box-shadow: 0 0 24px rgba(0,225,255,0.4); }
.stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div, .stFileUploader>div, .stNumberInput input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(124,92,255,0.22) !important; color: var(--text) !important; border-radius: 12px !important; }
.stTabs [role="tab"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(124,92,255,0.18); border-radius: 12px; color: #b7c7dd; padding:1.6rem; margin:0.4rem;}
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(124,92,255,0.22), rgba(0,225,255,0.18)); color: var(--text) !important; }
.metric { border-radius: 14px; padding: 0.8rem 1rem; background: rgba(255,255,255,0.05); border: 1px solid rgba(124,92,255,0.2); display: flex; justify-content: space-between; align-items: center; }
.chip { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 999px; background: rgba(124,92,255,0.2); border: 1px solid rgba(124,92,255,0.35); color: var(--text); margin-right: 6px; margin-bottom: 6px; font-size: 0.8rem; }
.stProgress > div > div { background: linear-gradient(90deg, var(--primary), var(--primary-2)) !important; }
section[data-testid="stSidebar"] { background: rgba(8, 12, 20, 0.75); backdrop-filter: blur(12px); border-right: 1px solid rgba(124,92,255,0.25); }
</style>
"""

# -------------------- App Shell --------------------
st.set_page_config(page_title="EasyApply", page_icon="📄")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown("""
<div class="hero" style="margin:1.2rem">
  <h1> Easy Apply <span style="opacity:.85">— AI Job Matching</span></h1>
  <p style="color:#b7c7dd;margin:0.4rem 0 0">Neon-precise parsing, smart scoring, and HR-grade evaluation.</p>
</div>
""", unsafe_allow_html=True)

# Session init
for key in ['logged_in','username','role','user_id']:
    if key not in st.session_state: st.session_state[key]=None if key!='logged_in' else False

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Google API Key", type="password", help="Needed for analysis.")
    models = {'extract': st.text_input("Extraction/AI Model", DEFAULT_GENAI_MODEL_EXTRACT),
              'embed': st.text_input("Embedding Model", DEFAULT_GENAI_MODEL_EMBED)}
    st.markdown("---")
    if st.session_state.logged_in:
        st.write(f"Logged in as **{st.session_state.username}** ({st.session_state.role})")
        if st.button("Logout"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
    else:
        st.header("User Access")
        login_tab, signup_tab = st.tabs(["Login","Sign Up"])
        with login_tab:
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                user=_row_to_dict(db.get_user(login_user))
                if user and db.check_password(user['password_hash'], login_pass):
                    st.session_state.logged_in=True; st.session_state.username=user['username']; st.session_state.role=user['role']; st.session_state.user_id=user['id']; st.rerun()
                else: st.error("Invalid username or password.")
        with signup_tab:
            signup_user=st.text_input("Username", key="signup_user")
            signup_pass=st.text_input("Password", type="password", key="signup_pass")
            signup_role=st.selectbox("Role", ["student","tnp"], key="signup_role")
            if st.button("Sign Up"):
                if db.add_user(signup_user, signup_pass, signup_role): st.success("Account created! Please login.")
                else: st.error("Username already taken.")

#--------------------- CLEANING IT ----------------------------------
def safe_get(d, key, default="Not Specified"):
    """Return a cleaned value from dict d, or default if missing/None/'None'."""
    val = d.get(key) if isinstance(d, dict) else None
    if val is None:
        return default
    if isinstance(val, str) and val.strip().lower() == "none":
        return default
    if isinstance(val, list) and not val:  # empty list
        return default
    return val

# -------------------- Student view (fixed tabs) --------------------
def render_job_detail_panel(job_dict):
    role_struct=None
    try:
        role_struct=job_dict.get("jd_data_json") or job_dict.get("jd_structured") or job_dict.get("jd_data")
        if isinstance(role_struct,str): role_struct=json.loads(role_struct)
        if isinstance(role_struct,dict) and "roles" in role_struct and isinstance(role_struct["roles"],list) and role_struct["roles"]:
            role_struct=role_struct["roles"][0]
    except Exception: role_struct=None
    with st.expander("📄 Job details", expanded=False):
        if isinstance(role_struct, dict):
            stipend = role_struct.get("stipend") or {}

            st.markdown(f"**Role:** {safe_get(role_struct, 'job_title')}")
            st.markdown(f"**Location:** {safe_get(role_struct, 'location')}")
            st.markdown(f"**Schedule:** {safe_get(role_struct, 'schedule')}")

            st.markdown(
                f"**Stipend:** {safe_get(stipend, 'amount', '-')} "
                f"{safe_get(stipend, 'currency', '')} per "
                f"{safe_get(stipend, 'period', 'month')}"
            )

            st.markdown(f"**Internship:** {safe_get(role_struct, 'internship_months', '-')} months")
            st.markdown(f"**Bond:** {safe_get(role_struct, 'bond_months', '-')} months")

            # Education field
            edu = safe_get(role_struct, "required_education", "N/A")
            if isinstance(edu, list):
                edu_str = ", ".join(edu) if edu else "N/A"
            elif isinstance(edu, str):
                edu_str = edu
            else:
                edu_str = "N/A"
            st.markdown(f"**Education:** {edu_str}")

            # Skills
            skills = role_struct.get("required_skills")
            if isinstance(skills, list) and skills:
                skills_str = ", ".join(skills)
            else:
                skills_str = "N/A"
            st.markdown("**Required skills:** " + skills_str)

            # Responsibilities
            responsibilities = role_struct.get("responsibilities")
            if isinstance(responsibilities, list) and responsibilities:
                st.markdown("**Responsibilities:**")
                for r in responsibilities[:12]:
                    st.markdown(f"- {r}")
        else:
            st.write("Structured details unavailable; showing original JD text:")
            st.code(job_dict.get("jd_text","N/A")[:3000])

def student_view(client, models):
    st.subheader("Student Dashboard")
    tabs = st.tabs(["Available Jobs","My Applications","Help"])
    with tabs[0]:
        st.markdown('<div class="neon-panel neon-strong">', unsafe_allow_html=True)
        st.subheader("Apply for a Job")
        all_jobs=_rows_to_dicts(db.get_all_jobs())
        if not all_jobs:
            st.info("No jobs are available at the moment.")
        else:
            job_options={f"{job['job_title']} (ID: {job['id']})": job['id'] for job in all_jobs}
            selected_job_str=st.selectbox("Select a Job", options=list(job_options.keys()))
            selected_job_id=job_options[selected_job_str]
            sel_job=next((j for j in all_jobs if j['id']==selected_job_id), None)
            if sel_job: render_job_detail_panel(sel_job)
            with st.form("application_form", clear_on_submit=True):
                resume_file=st.file_uploader("Upload Your Resume (PDF)", type="pdf")
                if st.form_submit_button("Apply") and resume_file and sel_job:
                    with st.spinner("Analyzing your resume and submitting application..."):
                        resume_text=extract_text_from_pdf(resume_file)
                        if resume_text and sel_job:
                            resume_data=extract_structured_data(client, models['extract'], resume_text, 'resume')
                            jd_role=None
                            try:
                                jd_role=sel_job.get("jd_data_json") or sel_job.get("jd_structured") or sel_job.get("jd_data")
                                if isinstance(jd_role,str): jd_role=json.loads(jd_role)
                                if isinstance(jd_role,dict) and "roles" in jd_role and isinstance(jd_role["roles"],list) and jd_role["roles"]:
                                    jd_role=jd_role["roles"][0]
                            except Exception: jd_role=None
                            if not isinstance(jd_role,dict):
                                jd_role={"job_title":sel_job.get("job_title"),"required_skills":[],"responsibilities":[],"required_experience_years":None,"location":None,"schedule":None,"bond_months":None}
                            cache=st.session_state.setdefault("ai_score_cache",{})
                            final_score, parts = compute_final_score_with_ai(client, models['embed'], models['extract'], jd_role, resume_text, resume_data or {}, cache)
                            if resume_data:
                                if db.add_application(st.session_state.user_id, sel_job['id'], resume_text, resume_data, final_score):
                                    st.success(f"Application submitted successfully! Score: {final_score:.2f}")
                                    with st.expander("Score breakdown"):
                                        st.markdown(f'<div class="chip">Hard: {parts["hard"]:.0f}</div>', unsafe_allow_html=True)
                                        st.markdown(f'<div class="chip">Semantic: {parts["semantic"]:.0f}</div>', unsafe_allow_html=True)
                                        st.markdown(f'<div class="chip">Other: {parts["other"]:.0f}</div>', unsafe_allow_html=True)
                                        st.markdown(f'<div class="chip">AI: {parts["ai"]:.0f}</div>', unsafe_allow_html=True)
                                        if parts.get("ai_subscores"): st.json(parts["ai_subscores"])
                                else: st.warning("You have already applied for this job.")
                            else: st.error("Could not extract structured data from your resume.")
                        else: st.error("Failed to process resume or find job details.")
        st.markdown('</div>', unsafe_allow_html=True)
    with tabs[1]:
        st.subheader("My Applications")
        st.markdown('<div class="neon-panel">', unsafe_allow_html=True)
        student_apps=_rows_to_dicts(db.get_student_applications(st.session_state.user_id))
        if not student_apps: st.info("You haven't applied for any jobs yet.")
        else:
            for app in student_apps:
                st.markdown(f"- **{app['job_title']}**: Your relevance score was `{app['relevance_score']:.2f}%`")
        st.markdown('</div>', unsafe_allow_html=True)
    with tabs[2]:
        st.subheader("Help")
        help_page("student")

# -------------------- TnP Views --------------------
def tnp_view(client, models):
    st.subheader("TnP Dashboard")
    st.markdown('<div class="neon-panel neon-strong">', unsafe_allow_html=True)
    st.subheader("Post a New Job")
    with st.form("new_job_form", clear_on_submit=True):
        jd_file=st.file_uploader("Upload Job Description PDF", type="pdf")
        submitted=st.form_submit_button("Post Job")
        if submitted and jd_file:
            with st.spinner("Processing JD..."):
                jd_text=extract_text_from_pdf(jd_file)
                if jd_text:
                    jd_data=extract_structured_data(client, models['extract'], jd_text, 'jd')
                    if jd_data and isinstance(jd_data,dict) and jd_data.get("roles"):
                        posted=0
                        for role in jd_data["roles"]:
                            title=role.get("job_title") or "Untitled Role"
                            db.add_job(st.session_state.user_id, title, jd_text, role); posted+=1
                        st.success(f"Successfully posted {posted} role(s) from the JD.")
                    else: st.error("Could not extract required details or roles from the JD.")
                else: st.error("Could not read text from the uploaded JD PDF.")
    st.markdown('</div>', unsafe_allow_html=True)

    jobs=_rows_to_dicts(db.get_tnp_jobs(st.session_state.user_id))
    total_apps=sum(len(_rows_to_dicts(db.get_applications_for_job(j["id"]))) for j in jobs)
    all_scores=[a["relevance_score"] for j in jobs for a in _rows_to_dicts(db.get_applications_for_job(j["id"]))]
    avg_score=float(np.mean(all_scores)) if all_scores else 0.0
    col1,col2,col3=st.columns(3)
    with col1: st.markdown(f'<div class="metric"><span>Jobs Posted</span><strong>{len(jobs)}</strong></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric"><span>Total Applications</span><strong>{total_apps}</strong></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric"><span>Avg Score</span><strong>{avg_score:.1f}%</strong></div>', unsafe_allow_html=True)

    st.subheader("Your Posted Jobs")
    for job in jobs:
        with st.expander(f"**{job['job_title']}**"):
            st.markdown('<div class="neon-panel">', unsafe_allow_html=True)
            render_job_detail_panel(job)
            st.markdown("---")
            applications=_rows_to_dicts(db.get_applications_for_job(job['id']))
            if not applications:
                st.write("No applications received yet.")
            else:
                for app in applications:
                    resume_data = json.loads(app['resume_data_json']) if isinstance(app.get('resume_data_json'), str) else app.get('resume_data_json', {})
                    st.markdown(f"**Applicant:** {app['username']} | **Score:** `{app['relevance_score']:.2f}%`")
                    st.progress(int(app['relevance_score']))
                    st.text(f"Predicted Role: {resume_data.get('predicted_role', 'N/A')}")
                    st.text(f"Skills: {', '.join(resume_data.get('skills', []))}")
                    with st.expander("👤 View applicant details"):
                        st.markdown(f"**Name:** {resume_data.get('name','N/A')}")
                        st.markdown(f"**Predicted role:** {resume_data.get('predicted_role','N/A')}")
                        edu = resume_data.get("education") or []
                        if edu:
                            st.markdown("**Education:**")
                            for e in edu[:5]:
                                deg=e.get("degree") or "-"
                                inst=e.get("institute_name") or "-"
                                spec=e.get("specialization") or "-"
                                st.markdown(f"- {deg} in {spec} @ {inst}")
                        exp = resume_data.get("experience") or []
                        if exp:
                            st.markdown("**Experience:**")
                            for x in exp[:8]:
                                st.markdown(f"- {x.get('role','-')} @ {x.get('company','-')}")
                        if app.get("resume_text"):
                            st.markdown("**Resume snippet:**")
                            txt=app["resume_text"]
                            st.code((txt[:1500] + ("..." if len(txt)>1500 else "")))
                    st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)

def tnp_analytics_page():
    st.subheader("Analytics")
    mode = st.radio("Mode", ["Overview","Per Applicant","Per Job","Compare Applicants"], horizontal=True)

    # Pull data
    jobs=_rows_to_dicts(db.get_tnp_jobs(st.session_state.user_id))
    apps_df_rows=[]
    for j in jobs:
        for a in _rows_to_dicts(db.get_applications_for_job(j["id"])):
            apps_df_rows.append({
                "job_id": j["id"],
                "job_title": j["job_title"],
                "applicant": a["username"],
                "score": a["relevance_score"],
                "created_at": a.get("created_at") or "",
            })
    df = pd.DataFrame(apps_df_rows) if apps_df_rows else pd.DataFrame(columns=["job_id","job_title","applicant","score","created_at"])
    if "created_at" in df.columns and not df.empty:
        try:
            df["created_at"]=pd.to_datetime(df["created_at"])
        except Exception:
            df["created_at"]=pd.NaT

    if mode=="Overview":
        st.markdown('<div class="neon-panel">', unsafe_allow_html=True)
        col1,col2,col3,col4=st.columns(4)
        with col1: st.markdown(f'<div class="metric"><span>Roles</span><strong>{len(jobs)}</strong></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric"><span>Applications</span><strong>{len(df)}</strong></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric"><span>Avg Score</span><strong>{(df["score"].mean() if not df.empty else 0):.1f}%</strong></div>', unsafe_allow_html=True)
        with col4:
            hit = (df["score"]>=70).mean()*100 if not df.empty else 0
            st.markdown(f'<div class="metric"><span>Hit Rate ≥70%</span><strong>{hit:.1f}%</strong></div>', unsafe_allow_html=True)

        # Time series
        if not df.empty and df["created_at"].notna().any():
            ts = df.set_index("created_at").resample("D").size().rename("applications").to_frame()
            st.line_chart(ts, height=220)
        # Score distribution
        if not df.empty:
            st.bar_chart(df["score"].clip(0,100).round(), height=220)
        # Job-wise average
        if not df.empty:
            job_avg = df.groupby("job_title")["score"].mean().sort_values(ascending=False).head(15)
            st.bar_chart(job_avg, height=260)
        st.markdown('</div>', unsafe_allow_html=True)

    elif mode=="Per Applicant":
        st.markdown('<div class="neon-panel">', unsafe_allow_html=True)
        users = sorted(df["applicant"].dropna().unique().tolist()) if not df.empty else []
        sel = st.selectbox("Select applicant", users) if users else None
        if sel:
            d = df[df["applicant"]==sel].copy()
            st.markdown(f"**Applications by {sel}: {len(d)} | Avg: {d['score'].mean():.1f}%**")
            if d["created_at"].notna().any():
                st.line_chart(d.set_index("created_at")["score"], height=220)
            st.dataframe(d[["job_title","score","created_at"]].sort_values(by="created_at", ascending=False), use_container_width=True)
        else:
            st.info("No applicants yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif mode=="Per Job":
        st.markdown('<div class="neon-panel">', unsafe_allow_html=True)
        jobs_opts = [j["job_title"] for j in jobs]
        sel = st.selectbox("Select job", jobs_opts) if jobs_opts else None
        if sel:
            d = df[df["job_title"]==sel].copy()
            st.markdown(f"**Applications for {sel}: {len(d)} | Avg: {d['score'].mean():.1f}%**")
            if not d.empty:
                st.bar_chart(d["score"], height=220)
                st.dataframe(d[["applicant","score","created_at"]].sort_values(by="score", ascending=False), use_container_width=True)
            else:
                st.info("No applications yet for this role.")
        else:
            st.info("No roles yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:  # Compare Applicants
        st.markdown('<div class="neon-panel">', unsafe_allow_html=True)
        users = sorted(df["applicant"].dropna().unique().tolist()) if not df.empty else []
        sel_multi = st.multiselect("Choose applicants to compare", users, default=users[:2] if len(users)>=2 else users)
        if sel_multi:
            comp = df[df["applicant"].isin(sel_multi)]
            pivot = comp.pivot_table(index="job_title", columns="applicant", values="score", aggfunc="mean")
            st.dataframe(pivot.round(1), use_container_width=True)
            # Radar-like representation is non-trivial; show grouped bars via melt
            melted = pivot.reset_index().melt(id_vars="job_title", var_name="applicant", value_name="score").dropna()
            if not melted.empty:
                # Simple grouped bar by using pivot again to bars
                for applicant in sel_multi:
                    series = pivot[applicant].dropna().sort_values(ascending=False).head(10) if applicant in pivot.columns else pd.Series(dtype=float)
                    st.markdown(f"**Top roles for {applicant}**")
                    if not series.empty:
                        st.bar_chart(series, height=220)
                    else:
                        st.info("No data for this applicant.")
        else:
            st.info("Select at least one applicant.")
        st.markdown('</div>', unsafe_allow_html=True)

def results_history_page(role: str):
    st.subheader("Results History")
    if role=="tnp":
        jobs=_rows_to_dicts(db.get_tnp_jobs(st.session_state.user_id))
        rows=[]
        for j in jobs:
            for a in _rows_to_dicts(db.get_applications_for_job(j["id"])):
                rows.append({"job_id":j["id"],"job_title":j["job_title"],"applicant":a["username"],"score":a["relevance_score"],"created_at":a.get("created_at","")})
    else:
        rws=_rows_to_dicts(db.get_student_applications(st.session_state.user_id))
        rows=[{"job_id":r["job_id"],"job_title":r["job_title"],"score":r["relevance_score"],"created_at":r.get("created_at","")} for r in rws]
    st.markdown('<div class="neon-panel">', unsafe_allow_html=True)
    df=pd.DataFrame(rows) if rows else pd.DataFrame(columns=["job_id","job_title","applicant","score","created_at"])
    if not df.empty:
        q=st.text_input("Search by job title/applicant")
        min_s,max_s=st.slider("Score range",0,100,(0,100))
        if "created_at" in df.columns:
            try: df["created_at"]=pd.to_datetime(df["created_at"])
            except: pass
        if q:
            mask=df.apply(lambda r: q.lower() in " ".join([str(v).lower() for v in r.values]), axis=1)
            df=df[mask]
        df=df[(df["score"]>=min_s) & (df["score"]<=max_s)]
        st.dataframe(df.sort_values(by="created_at", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No records yet.")
    st.markdown('</div>', unsafe_allow_html=True)

def help_page(role: str):
    if role == "tnp":
        st.header("Help & Documentation — TnP")

        # API Key Setup
        with st.expander("🔑 API Key Setup (Required for AI Features)", expanded=True):
            st.markdown("""
            ### Setting up Gemini API Key:
            
            1. Get API Key: Open Google AI Studio and create a Gemini API key for free           
            2. Create secrets.toml in your project:
               ```
               # .streamlit/secrets.toml
               GOOGLE_API_KEY or GEMINI_API_KEY= "your_api_key_here"
               ```
            3. File structure:
               ```
               your_project/
               ├── .streamlit/
               │   └── secrets.toml
               ├── db.py
               └── app.py
            
               ```
            4. Restart Streamlit after adding secrets

            Alternative (local environment variables):
            - macOS/Linux: export GEMINI_API_KEY ="your_key"
            - Windows (PowerShell): setx GEMINI_API_KEY "your_key"

            5. Otherwise after step 1 you can just skip to Visit the webpage and enter your API_KEY in the sidebar[It's secured]
            """)

        # Quick Start Guide (TnP)
        with st.expander("Quick Start Guide (TnP)", expanded=True):
            st.markdown("""
            ### Getting Started in 4 Steps:
            
            1. Setup API: Add Gemini API key to secrets.toml or environment.
            2. Post Job(s): Go to Dashboard → Post a New Job and upload a JD (PDF). Multi-role JDs auto-split.
            3. Verify Details: Expand Job details to confirm skills, responsibilities, stipend, and constraints.
            4. Review Applicants: Open Your Posted Jobs, check scores and structured profiles; explore Analytics and Results.
            """)

        # Shortlisting Logic
        with st.expander("Enhanced Shortlisting Logic", expanded=True):
            st.markdown("""
            ### How Candidates are Scored and Shortlisted:
            
            Final score = 0.35×Hard + 0.15×Semantic + 0.10×Other + 0.40×AI

            - Hard: Overlap with required_skills (weighted by responsibilities presence).
            - Semantic: Embedding similarity with calibrated scaling to avoid mid-band saturation.
            - Other: Education fit, experience alignment, location/availability/bond signals.
            - AI: HR-style rubric score with auditable subscores and explanations.

            Automatic shortlisting guidance:
            - 75%+: Excellent — shortlist
            - 60–74%: Good — shortlist
            - 50–59% with 3+ key skills: Qualified — consider
            - 45–49%: Fair — track
            - <45%: Poor/No match
            """)

        # Using Analytics
        with st.expander("Using Analytics", expanded=False):
            st.markdown("""
            ### Modes:
            - Overview: Totals, trends, score distribution, job-wise average.
            - Per Applicant: Time-series of scores and application list per person.
            - Per Job: Distribution and list of applicants for a role.
            - Compare Applicants: Side-by-side performance across roles.

            ### Tips:
            - Use date filters, pick specific applicants/jobs to drill down.
            - Compare 2–4 candidates to understand relative strengths quickly.
            - Calibrate thresholds after collecting 100–200 applications.
            """)

        # Installation Guide
        with st.expander("Installation & Setup", expanded=False):
            st.markdown("""
            ### Required packages:
            ```
            pip install -r requirements.txt
            ```
            Run:
            ```
            streamlit run app.py
            ```
            """)

        # Feedback System
        with st.expander("AI Feedback System", expanded=False):
            st.markdown("""
            ### What the AI Provides:
            - AI score (0–100) + rubric subscores (skills, experience, education, projects, constraints).
            - JSON-only outputs for auditability and easy review.
            - Use subscores to justify shortlisting decisions and give specific feedback.
            """)

        st.subheader("Need Help?")
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            Common Issues:
            1. Secrets file or environment variable not set
            2. Inappropriate JD or unclear JD with no proper definations.
            3. Not entering the API key in Sidebar
            4. Password-protected files or file format other than pdf
            """)
        with col2:
            st.success("""
            Best Practices:
            1. Keep JD responsibilities and skills explicit
            2. Include stipend, location, schedule, and bond clearly
            3. Use Analytics to tune thresholds per cohort
            4. Compare top applicants to validate the rubric’s decisions
            """)

    else:
        st.header("Help & Documentation — Students")

        # API Key Info (contextual)
        with st.expander("How the App Uses AI", expanded=True):
            st.markdown("""
            - The platform uses an AI model to parse job descriptions and evaluate resumes.
            - An API key can be configured by the system admin for all so that students do not need to set it.
            """)

        # Quick Start Guide (Students)
        with st.expander("Quick Start Guide", expanded=True):
            st.markdown("""
            ### Apply in 3 Steps:
            
            1. Open Available Jobs → select a role
            2. Check “Job details” to understand skills, responsibilities, stipend, and constraints
            3. Upload resume (PDF) and click Apply — view the score and its breakdown
            """)

        # How scoring affects applications
        with st.expander("How Your Score is Calculated", expanded=True):
            st.markdown("""
            Final score = 0.35×Hard + 0.15×Semantic + 0.10×Other + 0.40×Screening
            
            - Hard: Do listed required_skills appear in the resume?
            - Semantic: Does the resume content align with the role text?
            - Other: Education fit, experience years, availability signals (e.g., relocation)
            - Screeing: HR evaluation with rubric-based subscores
            """)

        # Improve score
        with st.expander("Tips to Improve Your Score", expanded=False):
            st.markdown("""
            - Mirror the exact skill names from Job details if applicable.
            - Add concise bullets under projects/experience that show tool use and impact.
            - Mention location flexibility or immediate availability if relevant.
            - Keep sections well-structured (skills, education, projects, experience).
            """)

        # Installation/Access context
        with st.expander("Access & Files", expanded=False):
            st.markdown("""
            - Upload resumes as PDF's.
            - Avoid password-protected files.
            - If a resume is updated, reapply or contact TnP to refresh the application.
            """)

        # Feedback details
        with st.expander("Understanding Feedback", expanded=False):
            st.markdown("""
            After applying, expand “Score breakdown” to see:
            - Hard/Semantic/Other/AI components
            - Missing key skills and strength highlights (if AI is enabled)
            - Use these to tailor improvements before reapplying to other roles
            """)

        st.subheader("Need Help?")
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            Common Issues:
            1. PDF has no selectable text
            2. File is password-protected
            3. Resume content lacks explicit skill keywords
            """)
        with col2:
            st.success("""
            Best Practices:
            1. Keep the resume clean and keyword-rich.
            2. Highlight role-relevant projects and tools
            3. Check Job details to align wording before applying
            """)


def tnp_router(client, models):
    tabs = st.tabs(["Dashboard","Analytics","Results History","Help"])
    with tabs[0]: tnp_view(client, models)
    with tabs[1]: tnp_analytics_page()
    with tabs[2]: results_history_page("tnp")
    with tabs[3]: help_page("tnp")

# -------------------- Main --------------------
if st.session_state.logged_in:
    try:
        client=_make_genai_client(api_key_input)
    except Exception as e:
        st.error(f"Gemini initialization error: {e}"); st.stop()
    if st.session_state.role=='tnp': tnp_router(client, models)
    else: student_view(client, models)
else:
    st.info("Please login or sign up using the sidebar to continue.")
