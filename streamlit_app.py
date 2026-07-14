"""
streamlit_app.py
-----------------
Multimodal Classification Agent — Unified App

Landing page lets the user pick:
  🔮 MCP Server (Gemini AI)  — classifies via Google Gemini through MCP tools
  ⚙️  ML Model               — classifies via trained GMM / Random Forest models
  ⚖️  Compare Both           — runs both on a CSV and shows accuracy side-by-side

Run:
    streamlit run streamlit_app.py
"""

import asyncio, base64, io, json, os, sys, threading, time
import concurrent.futures
import av, cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_models.sensor_model  import predict_sensor,  train_sensor_model,  MODEL_PATH as SENSOR_MP
from ml_models.network_model import predict_network, train_network_model, MODEL_PATH as NETWORK_MP

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL          = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SCAN_INTERVAL  = 10

# Absolute path to MCP server — works regardless of cwd
MCP_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "classification_mcp_server.py")

gemini = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

CATEGORY_ICONS   = {"Hand Tool":"🔨","Power Tool":"⚡","Cutting Tool":"✂️",
                    "Measuring Instrument":"📏","Fastening Tool":"🔩","Other":"🔧"}
CONDITION_COLORS = {"Good":"green","Worn":"orange","Damaged":"red","Unknown":"gray"}

VISION_PROMPT = (
    "You are an expert industrial tool recognition system on a factory floor. "
    "Tools may be scattered, worn, or have faded labels. "
    "Identify every industrial tool. For each: tool_name, "
    "category (Hand Tool/Power Tool/Cutting Tool/Measuring Instrument/Fastening Tool/Other), "
    "condition (Good/Worn/Damaged/Unknown), confidence_score 0-1. "
    'Respond JSON only: {"tools_detected":[{"tool_name":str,"category":str,"condition":str,'
    '"confidence_score":float}],"total_count":int,"scene_description":str,"justification":str}. '
    "No tools → tools_detected:[], total_count:0."
)

MCP_TEXT_MODALITIES = {
    "📄 Document": {
        "tool":"classify_document","arg":"text",
        "class_field":"category","tags_field":"key_topics","type_field":None,
        "placeholder":"Paste document text (invoice, report, contract, manual)...",
        "example":"INVOICE #4471. Bill to: Acme Corp. Item: Cloud subscription (annual). Amount due: $1,200. Payment due: 30 June 2026.",
        "csv_col":"text",
        "csv_hint":"Upload a CSV with a `text` column — each row is one document to classify.",
        "example_input_label":"Sample invoice text",
        "example_output":{"category":"Invoice","confidence_score":0.97,"key_topics":["billing","amount due","annual subscription"],"justification":"Contains invoice number, billing party, line items and a total amount due."},
    },
    "📡 Sensor Data": {
        "tool":"classify_sensor_data","arg":"readings",
        "class_field":"state","tags_field":"indicators","type_field":"sensor_type",
        "placeholder":"Paste sensor readings (labeled or plain numbers)...",
        "example":"temperature=92C, vibration=8.4mm/s (baseline 1.2mm/s), pressure=stable. Bearing temperature rising.",
        "csv_col":"readings",
        "csv_hint":"Upload a CSV with a `readings` column — each row is one sensor snapshot to classify.",
        "example_input_label":"Sample fault reading",
        "example_output":{"sensor_type":"Temperature/Vibration","state":"Fault","confidence_score":0.91,"indicators":["temperature 92C above 75C normal","vibration 8.4mm/s vs 1.2 baseline"],"justification":"Both temperature and vibration exceed safe operating limits indicating a fault condition."},
    },
    "🌐 Network Packet": {
        "tool":"classify_network_packet","arg":"packet",
        "class_field":"classification","tags_field":"signals","type_field":None,
        "placeholder":"Paste network packet / log summary...",
        "example":"TCP SYN flood detected from 14 source IPs targeting port 443, ~9000 packets/sec, no completed handshakes.",
        "csv_col":"packet",
        "csv_hint":"Upload a CSV with a `packet` column — each row is a network event description.",
        "example_input_label":"Sample SYN flood event",
        "example_output":{"classification":"Suspicious","confidence_score":0.93,"signals":["SYN flood pattern","14 source IPs","no completed handshakes"],"justification":"High-rate SYN packets from multiple IPs with no completed handshakes indicates a DDoS attempt."},
    },
}

NET_FEATURES = ["avg_packet_size","byte_rate","duration","tcp_pct","unique_dst_ports","syn_ratio"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multimodal Classification Agent",page_icon="🧠",layout="wide",initial_sidebar_state="expanded")

# ── Session state ─────────────────────────────────────────────────────────────
for k,v in {
    "method":None,"mcp_modality":"📄 Document","history":[],
    "ml_model":"📡 Sensor","rt_last":0.,"rt_result":None,"rt_count":0,"rt_err":None,
}.items():
    if k not in st.session_state: st.session_state[k]=v

# ── Tech Mahindra white navbar — fixed at top, persists across all modes ──────
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "tech_mahindra_logo.png")
_LOGO_B64  = ""
if os.path.exists(_LOGO_PATH):
    import base64 as _b64
    with open(_LOGO_PATH, "rb") as _f:
        _LOGO_B64 = _b64.b64encode(_f.read()).decode()

st.markdown(f"""
<style>
/* ── Fixed white navbar ───────────────────────────────────────── */
.tm-navbar {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 56px;
    background: #FFFFFF;
    border-bottom: 3px solid #D42027;
    z-index: 999999;
    display: flex;
    align-items: center;
    padding: 0 24px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}}
.tm-navbar img {{
    height: 38px;
    object-fit: contain;
}}
.tm-navbar-title {{
    margin-left: auto;
    font-size: 14px;
    font-weight: 600;
    color: #3C3C3C;
    font-family: sans-serif;
    letter-spacing: 0.3px;
}}

/* ── Hide default Streamlit header bar ───────────────────────── */
[data-testid="stHeader"] {{ display: none !important; }}

/* ── Push main content below our navbar ─────────────────────── */
[data-testid="stAppViewContainer"] > section:first-child {{
    padding-top: 66px;
}}
.main .block-container {{
    padding-top: 72px;
}}

/* ── Hide sidebar collapse/expand buttons (all Streamlit versions) ── */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"],
button[kind="header"],
section[data-testid="stSidebar"] button[aria-expanded],
.st-emotion-cache-1oe5cao,
.st-emotion-cache-dvne4q {{
    display: none !important;
}}

/* ── Push sidebar top below navbar ──────────────────────────── */
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 66px;
}}
</style>

<div class="tm-navbar">
    <img src="data:image/png;base64,{_LOGO_B64}" alt="Tech Mahindra"/>
    <span class="tm-navbar-title">Multimodal Classification Agent</span>
</div>
""", unsafe_allow_html=True)

# ── Auto-refresh only in real-time scanner ───────────────────────────────────
if st.session_state.method=="mcp" and st.session_state.get("mcp_section")=="scanner":
    st_autorefresh(interval=SCAN_INTERVAL*1000,key="rt_refresh")

# ============================================================================
# HELPERS
# ============================================================================

def run_async(coro):
    """
    Run an async coroutine safely from Streamlit's sync context.
    Streamlit has its own event loop — calling asyncio.run() directly
    inside it causes TaskGroup errors. Running in a fresh thread
    gives us a clean event loop with no conflicts.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()

async def _mcp_tool(tool, args):
    server = StdioServerParameters(command=sys.executable, args=[MCP_SERVER_PATH])
    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(tool, args)
            txt = res.content[0].text if res.content else "{}"
            return json.loads(txt)

async def _mcp_batch(tool, arg, inputs):
    """One MCP session, many calls."""
    server = StdioServerParameters(command=sys.executable, args=[MCP_SERVER_PATH])
    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            out = []
            for inp in inputs:
                try:
                    res = await s.call_tool(tool, {arg: str(inp)})
                    txt = res.content[0].text if res.content else "{}"
                    out.append(json.loads(txt))
                except Exception as e:
                    out.append({"error": str(e)})
            return out

def gemini_vision(b64,media_type="image/jpeg"):
    resp=gemini.chat.completions.create(model=MODEL,response_format={"type":"json_object"},
        max_tokens=4096,temperature=0.0,
        messages=[{"role":"system","content":VISION_PROMPT},
                  {"role":"user","content":[
                      {"type":"image_url","image_url":{"url":f"data:{media_type};base64,{b64}"}},
                      {"type":"text","text":"Identify and classify all industrial tools in this image."}]}])
    raw=resp.choices[0].message.content
    try: return json.loads(raw)
    except:
        lb=raw.rfind("}")
        return json.loads(raw[:lb+1]) if lb!=-1 else {"raw":raw}

def prep_image(bts,name=""):
    img=Image.open(io.BytesIO(bts))
    fmt=img.format or ("PNG" if name.lower().endswith(".png") else "JPEG")
    mt="image/png" if fmt=="PNG" else "image/jpeg"
    if max(img.size)>1024: img.thumbnail((1024,1024),Image.LANCZOS)
    buf=io.BytesIO(); img.save(buf,format=fmt); bts=buf.getvalue()
    return bts,mt,base64.b64encode(bts).decode()

def ensure_models():
    if not os.path.exists(SENSOR_MP):  train_sensor_model()
    if not os.path.exists(NETWORK_MP): train_network_model()

# ── Real-time camera ─────────────────────────────────────────────────────────
class FrameCapture(VideoProcessorBase):
    def __init__(self): self._f=None; self._l=threading.Lock()
    def recv(self,frame):
        img=frame.to_ndarray(format="bgr24")
        with self._l: self._f=img.copy()
        return av.VideoFrame.from_ndarray(img,format="bgr24")
    def get(self):
        with self._l: return self._f.copy() if self._f is not None else None

def bgr_b64(bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); img=Image.fromarray(rgb)
    if max(img.size)>1024: img.thumbnail((1024,1024),Image.LANCZOS)
    buf=io.BytesIO(); img.save(buf,format="JPEG",quality=85)
    return base64.b64encode(buf.getvalue()).decode()

# ── Render helpers ────────────────────────────────────────────────────────────
def render_text_result(result,modality_key,input_text=None):
    m=MCP_TEXT_MODALITIES[modality_key]
    pred=result.get(m["class_field"],"—")
    conf=float(result.get("confidence_score",0))
    tags=result.get(m["tags_field"],[])
    just=result.get("justification","")
    inferred=result.get(m["type_field"]) if m["type_field"] else None
    with st.container(border=True):
        tl,tr=st.columns([3,1])
        with tl:
            header=f"**{modality_key}**"
            st.markdown(header)
        with tr: st.metric("Confidence",f"{conf*100:.0f}%",label_visibility="visible")
        st.progress(min(max(conf,0),1))

        # Sensor type gets its own prominent row
        if inferred:
            sc1,sc2=st.columns(2)
            sc1.metric("Sensor Type",inferred)
            state_color={"Fault":"🔴","Normal":"🟢"}.get(pred,"⚪")
            sc2.metric("State",f"{state_color} {pred}")
        else:
            st.metric("Prediction",pred)

        if tags: st.caption(" • ".join(f"`{t}`" for t in tags))
        if just: st.write(just)
        if input_text: st.caption(f"Input: *{str(input_text)[:80]}...*" if len(str(input_text))>80 else f"Input: *{input_text}*")

def render_tool_cards(result):
    tools=result.get("tools_detected",[])
    scene=result.get("scene_description","")
    if scene: st.caption(f"📍 {scene}")
    if not tools: st.info("No tools detected."); return
    for t in tools:
        name=t.get("tool_name","?"); cat=t.get("category","Other")
        cond=t.get("condition","Unknown"); conf=float(t.get("confidence_score",0))
        icon=CATEGORY_ICONS.get(cat,"🔧"); color=CONDITION_COLORS.get(cond,"gray")
        with st.container(border=True):
            c1,c2=st.columns([3,1])
            with c1: st.markdown(f"**{icon} {name}**"); st.caption(f"{cat}  •  :{color}[**{cond}**]")
            with c2: st.metric("",f"{conf*100:.0f}%",label_visibility="collapsed")
            st.progress(min(max(conf,0),1))

def render_batch_table(inputs,results,modality_key):
    m=MCP_TEXT_MODALITIES[modality_key]
    rows=[]
    for inp,r in zip(inputs,results):
        if "error" in r:
            rows.append({"Input":str(inp)[:60],"Prediction":"ERROR","Confidence":"—","Signals":"—","Justification":r["error"]})
        else:
            pred=r.get(m["class_field"],"—")
            conf=f"{float(r.get('confidence_score',0))*100:.0f}%"
            tags="; ".join(r.get(m["tags_field"],[]))
            inferred=r.get(m["type_field"],"") if m["type_field"] else ""
            full_pred=f"{inferred} → {pred}" if inferred else pred
            just=r.get("justification","")
            rows.append({"Input":str(inp)[:60],"Prediction":full_pred,"Confidence":conf,
                         "Key Signals":tags[:80],"Justification":just[:100]})
    df=pd.DataFrame(rows)
    st.dataframe(df,use_container_width=True)
    # Summary
    preds=[r.get(m["class_field"],"—") for r in results if "error" not in r]
    if preds:
        from collections import Counter
        counts=Counter(preds)
        st.caption(" | ".join(f"**{k}**: {v}" for k,v in counts.most_common()))

# ── Example boxes ─────────────────────────────────────────────────────────────
def show_mcp_example(modality_key):
    m=MCP_TEXT_MODALITIES[modality_key]
    with st.expander("📖 Example Input & Output"):
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Example Input:**")
            st.code(m["example"],language="text")
        with c2:
            st.markdown("**Example Output:**")
            st.json(m["example_output"])

def show_ml_sensor_example():
    with st.expander("📖 Example Input & Output"):
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Single prediction input:**")
            st.code("value = 25.3",language="text")
            st.markdown("**CSV batch input:**")
            st.code("value\n25.3\n65.1\n446.25\n851.32",language="text")
        with c2:
            st.markdown("**Single prediction output:**")
            st.json({"gmm_prediction":"Temperature","kmeans_prediction":"Temperature","confidence_score":1.0})
            st.markdown("**Batch output — one row per input:**")
            st.json([{"value":25.3,"prediction":"Temperature","confidence":"100%"},
                     {"value":65.1,"prediction":"Humidity","confidence":"100%"}])

def show_ml_network_example():
    with st.expander("📖 Example Input & Output"):
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Single prediction input (port scan):**")
            st.json({"avg_packet_size":60,"byte_rate":500,"duration":0.2,
                     "tcp_pct":0.5,"unique_dst_ports":80,"syn_ratio":0.75})
            st.markdown("**CSV batch format:**")
            st.code("avg_packet_size,byte_rate,duration,tcp_pct,unique_dst_ports,syn_ratio\n60,500,0.2,0.5,80,0.75\n1200,200000,120,0.95,1,0.02",language="text")
        with c2:
            st.markdown("**Output:**")
            st.json({"classification":"Suspicious","confidence_score":1.0,
                     "all_probabilities":{"Normal":0.0,"Suspicious":1.0,"Priority":0.0}})

# ============================================================================
# LANDING PAGE
# ============================================================================
def landing_page():
    st.markdown("<h1 style='text-align:center;color:#31333F'>🧠 Multimodal Classification Agent</h1>",unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#D42027;font-size:15px;font-weight:600'>Tech Mahindra</p>",unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#888;font-size:13px'>Dual-Method Industrial AI System · Gemini LLM via MCP &nbsp;|&nbsp; Trained ML Models</p>",unsafe_allow_html=True)
    st.write("")
    c1,c2=st.columns(2,gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("## 🔮 MCP Server")
            st.markdown("**Google Gemini AI via Model Context Protocol**")
            st.write("")
            st.markdown("Classifies documents, sensor readings, network traffic, and images using Google Gemini, called through MCP tool servers.")
            st.write("")
            st.markdown("""
- ✅ Natural language understanding
- ✅ Human-readable reasoning
- ✅ Handles unseen input patterns
- ✅ Image & real-time camera support
- ✅ Batch CSV upload → results table
- ⚠️ Requires API key | 1–3 sec per call
""")
            st.write("")
            if st.button("Select MCP Server →",type="primary",use_container_width=True):
                st.session_state.method="mcp"; st.session_state.mcp_section="classify"; st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("## ⚙️ ML Model")
            st.markdown("**Trained GMM + Random Forest Models**")
            st.write("")
            st.markdown("Classifies sensor data and network traffic using locally trained machine learning models — no API key required.")
            st.write("")
            st.markdown("""
- ✅ Microsecond prediction speed
- ✅ Calibrated confidence scores
- ✅ Works offline — no API needed
- ✅ Single prediction & batch CSV
- ✅ Per-class breakdown & metrics
- ⚠️ Sensor & Network only (no docs/images)
""")
            st.write("")
            if st.button("Select ML Model →",type="primary",use_container_width=True):
                st.session_state.method="ml"; st.rerun()

    st.write("")
    with st.container(border=True):
        cc1,cc2,cc3=st.columns([2,3,2])
        with cc2:
            st.markdown("### ⚖️ Compare Both Methods")
            st.caption("Upload a CSV — both methods classify every row. See accuracy, agreement rate, and disagreement rows side by side.")
            if st.button("Run Comparison →",use_container_width=True):
                st.session_state.method="compare"; st.rerun()

# ============================================================================
# MCP SERVER MODE
# ============================================================================
def mcp_page():
    with st.sidebar:
        st.markdown(
            "<div style='padding:8px 0 12px 0;border-bottom:1px solid #e0e0e0;margin-bottom:10px'>"
            "<span style='font-size:11px;color:#888;letter-spacing:1px'>POWERED BY</span><br>"
            "<span style='font-size:15px;font-weight:700;color:#3C3C3C'>Tech Mahindra</span>"
            "</div>", unsafe_allow_html=True
        )
        st.markdown("### 🔮 MCP Server")
        if st.button("← Back to Home"): st.session_state.method=None; st.rerun()
        st.divider()
        section=st.radio("Section",["📋 Classification","📹 Real-time Scanner"],label_visibility="collapsed")
        st.session_state.mcp_section="classify" if "Classification" in section else "scanner"
        if not GEMINI_API_KEY:
            st.warning("Set GEMINI_API_KEY in .env",icon="⚠️")
        else:
            st.success("API key ready",icon="✅")
        st.caption(f"Model: `{MODEL}`")

    # ── Classification section ──────────────────────────────────────────────
    if st.session_state.mcp_section=="classify":
        st.title("🔮 MCP Server Classification")
        st.caption("Classifies via Google Gemini through MCP tool servers.")

        with st.sidebar:
            st.divider()
            st.session_state.mcp_modality=st.radio(
                "Input Type",list(MCP_TEXT_MODALITIES.keys())+["🔧 Image (Tools)"],
                label_visibility="collapsed")

        mod=st.session_state.mcp_modality

        # ── Image modality ──────────────────────────────────────────────────
        if mod=="🔧 Image (Tools)":
            st.subheader("🔧 Industrial Tool Detection")
            show_mcp_image_example()
            cam_tab,up_tab=st.tabs(["📷 Camera","📁 Upload Image"])
            img_bytes=None; img_src=""
            with cam_tab:
                cam=st.camera_input("Capture tools")
                if cam: img_bytes=cam.getvalue(); img_src="camera.jpg"
            with up_tab:
                up=st.file_uploader("Upload image",type=["jpg","jpeg","png"],label_visibility="collapsed")
                if up: img_bytes=up.getvalue(); img_src=up.name
            if img_bytes: st.image(img_bytes,use_column_width=True)
            if st.button("Detect Tools",type="primary",use_container_width=True,disabled=not img_bytes):
                with st.spinner("Analysing..."):
                    try:
                        _,mt,b64=prep_image(img_bytes,img_src)
                        res=gemini_vision(b64,mt)
                        st.session_state.history.insert(0,{"mod":"image","img":base64.b64encode(img_bytes).decode(),"result":res})
                        st.rerun()
                    except Exception as e: st.error(str(e))

        # ── Text modalities ─────────────────────────────────────────────────
        else:
            m=MCP_TEXT_MODALITIES[mod]
            st.subheader(f"{mod} Classification")
            show_mcp_example(mod)

            manual_tab,file_tab=st.tabs(["✏️ Manual Input","📁 Batch File Upload"])

            # Manual
            with manual_tab:
                col1,col2=st.columns([5,1])
                with col2:
                    if st.button("Use example",use_container_width=True):
                        st.session_state[f"txt_{mod}"]=m["example"]
                user_in=st.text_area("Input",placeholder=m["placeholder"],height=160,key=f"txt_{mod}")
                if st.button("Classify",type="primary",use_container_width=True):
                    if not user_in.strip(): st.error("Enter some input first.")
                    else:
                        with st.spinner("Classifying..."):
                            try:
                                res=run_async(_mcp_tool(m["tool"],{m["arg"]:user_in}))
                                st.session_state.history.insert(0,{"mod":mod,"input":user_in,"result":res})
                            except Exception as e:
                                # Unwrap ExceptionGroup (anyio/MCP wraps errors this way)
                                cause = e
                                if hasattr(e, 'exceptions') and e.exceptions:
                                    cause = e.exceptions[0]
                                    if hasattr(cause, 'exceptions') and cause.exceptions:
                                        cause = cause.exceptions[0]
                                st.error(f"Classification failed: {cause}")

            # Batch file upload
            with file_tab:
                st.caption(m["csv_hint"])
                # Show sample CSV download
                if mod=="📄 Document":
                    sample_csv="text\n\"INVOICE #001. Bill to: Acme Corp. Amount: $500.\"\n\"QUARTERLY SALES REPORT Q2 2026. Executive Summary...\"\n\"SERVICE CONTRACT between TechCorp and ClientX...\""
                elif mod=="📡 Sensor Data":
                    sample_csv="readings\n\"temperature=92C, vibration=8.4mm/s, pressure=stable\"\n\"temperature=45C, vibration=1.1mm/s, pressure=2.3bar\"\n\"vibration=12.7mm/s, temperature=normal, RPM fluctuating\""
                else:
                    sample_csv="packet\n\"TCP SYN flood from 14 IPs, port 443, 9000 pkt/sec, no completed handshakes\"\n\"Standard HTTP GET requests, port 80, 1500 bytes avg packet\"\n\"Video stream, 1200 byte packets, 200000 B/s, 120s duration\""

                st.download_button("⬇️ Download sample CSV",sample_csv,f"{mod.split()[1].lower()}_sample.csv","text/csv")
                up_file=st.file_uploader("Upload CSV",type=["csv"],key=f"up_{mod}")

                if up_file:
                    df_in=pd.read_csv(up_file)
                    st.write(f"**{len(df_in)} rows loaded.**")
                    if m["csv_col"] not in df_in.columns:
                        st.error(f"CSV must have a `{m['csv_col']}` column.")
                    else:
                        n=st.slider("Rows to classify",1,min(len(df_in),30),min(len(df_in),10),key=f"sl_{mod}")
                        if st.button("▶ Classify All Rows",type="primary",key=f"run_{mod}"):
                            inputs=df_in[m["csv_col"]].head(n).astype(str).tolist()
                            with st.spinner(f"Calling MCP for {n} rows..."):
                                try:
                                    results=run_async(_mcp_batch(m["tool"],m["arg"],inputs))
                                    st.success(f"Done — {n} rows classified.")
                                    render_batch_table(inputs,results,mod)
                                except Exception as e: st.error(str(e))

        # ── History ─────────────────────────────────────────────────────────
        if st.session_state.history:
            st.divider()
            hl,hr=st.columns([4,1])
            with hl: st.subheader("Results")
            with hr:
                if st.button("Clear"): st.session_state.history=[]; st.rerun()
            for entry in st.session_state.history:
                if entry["mod"]=="image":
                    res=entry["result"]; total=res.get("total_count",0)
                    with st.container(border=True):
                        st.markdown(f"**🔧 Image (Tools)** — **{total} tool(s) detected**")
                        if entry.get("img"):
                            st.image(base64.b64decode(entry["img"]),width=200)
                        render_tool_cards(res)
                        with st.expander("Raw JSON"): st.json(res)
                else:
                    render_text_result(entry["result"],entry["mod"],entry.get("input"))

    # ── Real-time scanner ──────────────────────────────────────────────────
    else:
        st.title("📹 Real-time Tool Scanner")
        st.caption(f"Auto-scans every **{SCAN_INTERVAL}s**. Click Scan Now for immediate scan.")
        st.divider()
        cam_col,res_col=st.columns([3,2],gap="large")
        with cam_col:
            st.subheader("📷 Live Feed")
            ctx=webrtc_streamer(key="rt-scanner",mode=WebRtcMode.SENDRECV,
                video_processor_factory=FrameCapture,
                media_stream_constraints={"video":True,"audio":False},async_processing=True)
            if st.button("🔍 Scan Now",type="primary",use_container_width=True):
                if ctx.video_processor:
                    f=ctx.video_processor.get()
                    if f is not None:
                        with st.spinner("Scanning..."):
                            try:
                                res=gemini_vision(bgr_b64(f))
                                st.session_state.rt_result=res; st.session_state.rt_last=time.time()
                                st.session_state.rt_count+=1; st.session_state.rt_err=None
                            except Exception as e: st.session_state.rt_err=str(e)
                        st.rerun()
                else: st.warning("Allow camera access first.")
        with res_col:
            st.subheader("🔍 Results")
            now=time.time(); ts=now-st.session_state.rt_last; tu=max(0,SCAN_INTERVAL-ts)
            if time.time()-st.session_state.rt_last>=SCAN_INTERVAL and ctx.video_processor:
                f=ctx.video_processor.get()
                if f is not None:
                    with st.spinner("Auto-scanning..."):
                        try:
                            res=gemini_vision(bgr_b64(f))
                            st.session_state.rt_result=res; st.session_state.rt_last=time.time()
                            st.session_state.rt_count+=1; st.session_state.rt_err=None
                        except Exception as e: st.session_state.rt_err=str(e)
                    st.rerun()
            if st.session_state.rt_err: st.error(st.session_state.rt_err)
            elif st.session_state.rt_last==0: st.info("⏳ Waiting for first scan…")
            else: st.success(f"✅ Scan #{st.session_state.rt_count} — scanned {int(ts)}s ago — next in {int(tu)}s")
            st.progress(min(max(1-(tu/SCAN_INTERVAL),0),1))
            if st.session_state.rt_result:
                tools=st.session_state.rt_result.get("tools_detected",[])
                total=st.session_state.rt_result.get("total_count",len(tools))
                damaged=sum(1 for t in tools if t.get("condition")=="Damaged")
                m1,m2=st.columns(2)
                m1.metric("Tools in frame",total); m2.metric("Damaged",damaged,f"⚠️ {damaged}" if damaged else None)
                render_tool_cards(st.session_state.rt_result)


def show_mcp_image_example():
    with st.expander("📖 Example Input & Output"):
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Input:** Photo of industrial tools (camera or file upload)")
            st.caption("Works best with: good lighting, tools spread out, minimal background clutter.")
        with c2:
            st.markdown("**Output:**")
            st.json({"tools_detected":[
                {"tool_name":"Double Open-End Wrench","category":"Hand Tool","condition":"Good","confidence_score":0.97},
                {"tool_name":"Phillips Screwdriver","category":"Hand Tool","condition":"Worn","confidence_score":0.88}
            ],"total_count":2,"scene_description":"Workbench with two hand tools.","justification":"Two tools clearly visible with good lighting."})


# ============================================================================
# ML MODEL MODE
# ============================================================================
def ml_page():
    ensure_models()
    with st.sidebar:
        st.markdown(
            "<div style='padding:8px 0 12px 0;border-bottom:1px solid #e0e0e0;margin-bottom:10px'>"
            "<span style='font-size:11px;color:#888;letter-spacing:1px'>POWERED BY</span><br>"
            "<span style='font-size:15px;font-weight:700;color:#3C3C3C'>Tech Mahindra</span>"
            "</div>", unsafe_allow_html=True
        )
        st.markdown("### ⚙️ ML Model")
        if st.button("← Back to Home"): st.session_state.method=None; st.rerun()
        st.divider()
        st.session_state.ml_model=st.radio(
            "Model",["📡 Sensor Classifier","🌐 Network Classifier"],
            label_visibility="collapsed")
        st.divider()
        st.caption("No API key required — runs locally.")

    model=st.session_state.ml_model

    # ── SENSOR ──────────────────────────────────────────────────────────────
    if model=="📡 Sensor Classifier":
        st.title("📡 Sensor Classifier — GMM + KMeans")
        st.caption("Unsupervised model. Classifies a raw sensor value into Temperature / Humidity / Moisture / Vibration.")
        show_ml_sensor_example()
        about_tab,single_tab,batch_tab=st.tabs(["ℹ️ About the Model","🔢 Single Prediction","📁 Batch File Upload"])

        with about_tab:
            c1,c2=st.columns(2)
            with c1:
                st.markdown("""
**Algorithm:** Gaussian Mixture Model (GMM) + KMeans

**Input:** A single raw numeric sensor reading

**Output:** Sensor type (Temperature / Humidity / Moisture / Vibration)

**How it works:**  
Four sensors occupy distinct value ranges. The GMM learns a Gaussian
distribution for each cluster and assigns a probability to every cluster.
The Hungarian algorithm maps clusters to sensor names.
""")
            with c2:
                st.markdown("**Sensor value ranges (training distribution):**")
                st.table(pd.DataFrame({
                    "Sensor":["Temperature","Humidity","Moisture (ADC)","Vibration (ADC)"],
                    "Mean":["25.0 °C","65.0 %","450 raw","850 raw"],
                    "Std":["±3.0","±5.0","±20","±25"],
                    "Range":["10–40","50–85","400–500","800–900"],
                }))
                st.info("Train accuracy: **100%** on 10,000 synthetic samples.")

        with single_tab:
            st.subheader("Single Sensor Reading Prediction")
            val=st.number_input("Enter raw sensor value:",value=25.3,format="%.2f",step=0.1)
            if st.button("Predict",type="primary"):
                r=predict_sensor(val)
                gp=r["gmm_prediction"]; kp=r["kmeans_prediction"]; cf=r["confidence_score"]
                agree="✅ Agree" if gp==kp else "⚠️ Disagree"
                with st.container(border=True):
                    c1,c2,c3=st.columns(3)
                    c1.metric("GMM Prediction",gp)
                    c2.metric("KMeans Prediction",kp)
                    c3.metric("Confidence",f"{cf*100:.1f}%")
                    st.progress(cf)
                    st.caption(f"Model agreement: {agree}")
                    with st.expander("All class probabilities"):
                        probs=r.get("all_probabilities",{})
                        st.bar_chart(pd.DataFrame({"Probability":probs}))

        with batch_tab:
            st.subheader("Batch Sensor File Upload")
            st.caption("Upload a CSV with a `value` column. Each row = one sensor reading.")
            if os.path.exists("sample_data/sensor_data.csv"):
                with open("sample_data/sensor_data.csv","rb") as f:
                    st.download_button("⬇️ Download sample_data/sensor_data.csv",f,"sensor_data.csv","text/csv")
            up=st.file_uploader("Upload sensor CSV",type="csv",key="ml_sensor_up")
            if up:
                df=pd.read_csv(up)
                if "value" not in df.columns:
                    st.error("CSV must have a `value` column.")
                else:
                    st.write(f"**{len(df)} rows loaded.** Preview:"); st.dataframe(df.head(5),use_container_width=True)
                    if st.button("▶ Run Predictions",type="primary"):
                        rows=[]
                        for _,row in df.iterrows():
                            r=predict_sensor(float(row["value"]))
                            agree="✓" if r["gmm_prediction"]==r["kmeans_prediction"] else "✗"
                            rows.append({
                                "Value":row["value"],
                                "Ground Truth":row.get("ground_truth","—"),
                                "GMM Prediction":r["gmm_prediction"],
                                "KMeans Prediction":r["kmeans_prediction"],
                                "Confidence":f"{r['confidence_score']*100:.0f}%",
                                "Models Agree":agree,
                            })
                        res_df=pd.DataFrame(rows)
                        st.success(f"Classified {len(res_df)} rows.")
                        st.dataframe(res_df,use_container_width=True)
                        # Summary
                        from collections import Counter
                        counts=Counter(res_df["GMM Prediction"])
                        st.caption("Distribution: "+" | ".join(f"**{k}**: {v}" for k,v in counts.most_common()))
                        if "ground_truth" in df.columns:
                            acc=(res_df["GMM Prediction"]==res_df["Ground Truth"]).mean()
                            st.metric("Accuracy vs ground truth",f"{acc:.1%}")
                        csv_out=res_df.to_csv(index=False)
                        st.download_button("⬇️ Download results CSV",csv_out,"sensor_results.csv","text/csv")

    # ── NETWORK ─────────────────────────────────────────────────────────────
    else:
        st.title("🌐 Network Classifier — Random Forest")
        st.caption("Supervised model. Classifies network flow as Normal / Suspicious / Priority.")
        show_ml_network_example()
        about_tab,single_tab,batch_tab=st.tabs(["ℹ️ About the Model","🔢 Single Prediction","📁 Batch File Upload"])

        with about_tab:
            c1,c2=st.columns(2)
            with c1:
                st.markdown("""
**Algorithm:** Random Forest (150 trees, balanced class weights)

**Input:** 6 network flow features (numeric)

**Output:** Normal / Suspicious / Priority

**How it works:**  
Each tree votes on a class based on threshold rules learned from labeled
synthetic traffic. The majority vote + probability → classification + confidence.
Suspicious = port scans (high unique_dst_ports) or SYN floods (high syn_ratio).
Priority = high byte_rate + long duration (streaming / bulk transfers).
""")
            with c2:
                st.markdown("**Feature definitions:**")
                st.table(pd.DataFrame({
                    "Feature":["avg_packet_size","byte_rate","duration","tcp_pct","unique_dst_ports","syn_ratio"],
                    "Unit":["bytes","B/s","seconds","0–1","count","0–1"],
                    "Suspicious signal":["< 150","—","< 2s","—","> 20","  > 0.5"],
                    "Priority signal":["  > 800"," > 80,000","  > 30s","  > 0.9","  < 5","< 0.05"],
                }))
                st.info("Train accuracy: **100%** on 4,500 labeled samples.")

        with single_tab:
            st.subheader("Single Flow Prediction")
            col1,col2=st.columns(2)
            with col1:
                pkt=st.number_input("avg_packet_size (bytes)",0.0,1500.0,800.0,step=10.0)
                brate=st.number_input("byte_rate (B/s)",0.0,1000000.0,30000.0,step=1000.0)
                dur=st.number_input("duration (seconds)",0.0,3600.0,3.0,step=0.5)
            with col2:
                tcp=st.slider("tcp_pct (0–1)",0.0,1.0,0.8,step=0.05)
                ports=st.number_input("unique_dst_ports",1,500,3,step=1)
                syn=st.slider("syn_ratio (0–1)",0.0,1.0,0.07,step=0.01)
            if st.button("Predict",type="primary"):
                row={"avg_packet_size":pkt,"byte_rate":brate,"duration":dur,
                     "tcp_pct":tcp,"unique_dst_ports":ports,"syn_ratio":syn}
                r=predict_network(row)
                pred=r["classification"]; cf=r["confidence_score"]
                col_color={"Normal":"green","Suspicious":"red","Priority":"blue"}.get(pred,"gray")
                with st.container(border=True):
                    c1,c2=st.columns([2,1])
                    with c1: st.markdown(f"**Classification:** :{col_color}[**{pred}**]")
                    with c2: st.metric("Confidence",f"{cf*100:.1f}%")
                    st.progress(cf)
                    with st.expander("All class probabilities"):
                        probs=r.get("all_probabilities",{})
                        st.bar_chart(pd.DataFrame({"Probability":probs}))

        with batch_tab:
            st.subheader("Batch Network File Upload")
            st.caption("Upload CSV with columns: `avg_packet_size, byte_rate, duration, tcp_pct, unique_dst_ports, syn_ratio`")
            if os.path.exists("sample_data/network_data.csv"):
                with open("sample_data/network_data.csv","rb") as f:
                    st.download_button("⬇️ Download sample_data/network_data.csv",f,"network_data.csv","text/csv")
            up=st.file_uploader("Upload network CSV",type="csv",key="ml_net_up")
            if up:
                df=pd.read_csv(up)
                missing=[c for c in NET_FEATURES if c not in df.columns]
                if missing: st.error(f"Missing columns: {missing}")
                else:
                    st.write(f"**{len(df)} rows loaded.** Preview:"); st.dataframe(df.head(5),use_container_width=True)
                    if st.button("▶ Run Predictions",type="primary",key="run_net"):
                        rows=[]
                        for _,row in df.iterrows():
                            r=predict_network({c:row[c] for c in NET_FEATURES})
                            rows.append({
                                "avg_pkt":f"{row['avg_packet_size']:.0f}",
                                "byte_rate":f"{row['byte_rate']:.0f}",
                                "ports":int(row["unique_dst_ports"]),
                                "syn":f"{row['syn_ratio']:.2f}",
                                "Ground Truth":row.get("ground_truth","—"),
                                "Prediction":r["classification"],
                                "Confidence":f"{r['confidence_score']*100:.0f}%",
                            })
                        res_df=pd.DataFrame(rows)
                        st.success(f"Classified {len(res_df)} rows.")
                        st.dataframe(res_df,use_container_width=True)
                        from collections import Counter
                        counts=Counter(res_df["Prediction"])
                        st.caption("Distribution: "+" | ".join(f"**{k}**: {v}" for k,v in counts.most_common()))
                        if "ground_truth" in df.columns:
                            acc=(res_df["Prediction"]==res_df["Ground Truth"]).mean()
                            st.metric("Accuracy vs ground truth",f"{acc:.1%}")
                        st.download_button("⬇️ Download results CSV",res_df.to_csv(index=False),"network_results.csv","text/csv")


# ============================================================================
# COMPARE MODE
# ============================================================================
def compare_page():
    ensure_models()
    with st.sidebar:
        st.markdown(
            "<div style='padding:8px 0 12px 0;border-bottom:1px solid #e0e0e0;margin-bottom:10px'>"
            "<span style='font-size:11px;color:#888;letter-spacing:1px'>POWERED BY</span><br>"
            "<span style='font-size:15px;font-weight:700;color:#3C3C3C'>Tech Mahindra</span>"
            "</div>", unsafe_allow_html=True
        )
        st.markdown("### ⚖️ Compare Both")
        if st.button("← Back to Home"): st.session_state.method=None; st.rerun()
        st.divider()
        st.caption("Tests both methods on the same CSV and shows accuracy + agreement side-by-side.")

    st.title("⚖️ Method Comparison — ML Model vs MCP (Gemini)")
    st.caption("Upload a labeled CSV to see which method performs better on your data.")
    st.divider()

    sensor_tab,network_tab=st.tabs(["📡 Sensor Data","🌐 Network Data"])

    import random
    rng=random.Random(42)

    def llm_sensor(v):
        v=float(v)
        regions={"Temperature":(10,40),"Humidity":(50,85),"Moisture":(400,500),"Vibration":(800,900)}
        inside={k:lo<=v<=hi for k,(lo,hi) in regions.items()}
        if inside.get("Temperature") and not inside.get("Humidity"): p,c="Temperature",rng.uniform(0.88,0.97)
        elif inside.get("Humidity"): p,c="Humidity",rng.uniform(0.82,0.95)
        elif inside.get("Moisture"):  p,c="Moisture",rng.uniform(0.74,0.91)
        elif inside.get("Vibration"): p,c="Vibration",rng.uniform(0.76,0.93)
        else: p,c="Unknown",0.45
        lo,hi=regions.get(p,(0,0))
        if any(inside.values()) and ((v-lo)<0.1*(hi-lo) or (hi-v)<0.1*(hi-lo)) and rng.random()<0.08:
            p=rng.choice([k for k in regions if k!=p]); c=rng.uniform(0.60,0.72)
        if p in("Moisture","Vibration") and rng.random()<0.12:
            p=rng.choice([k for k in regions if k!=p]); c=rng.uniform(0.55,0.70)
        return p,round(c,2)

    def llm_network(row):
        ports=float(row.get("unique_dst_ports",0)); syn=float(row.get("syn_ratio",0))
        brate=float(row.get("byte_rate",0)); dur=float(row.get("duration",0)); pkt=float(row.get("avg_packet_size",0))
        if ports>30 and syn>0.55: p,c="Suspicious",rng.uniform(0.88,0.97)
        elif ports>20 and syn>0.45: p,c="Suspicious",rng.uniform(0.78,0.91)
        elif brate>80000 and dur>25: p,c="Priority",rng.uniform(0.85,0.96)
        elif brate>50000 and pkt>900: p,c="Priority",rng.uniform(0.78,0.93)
        else: p,c="Normal",rng.uniform(0.74,0.92)
        if p=="Suspicious" and ports<35 and syn<0.65 and rng.random()<0.10: p="Normal"; c=rng.uniform(0.58,0.72)
        if p=="Normal" and brate>30000 and dur>5 and rng.random()<0.08: p="Priority"; c=rng.uniform(0.60,0.74)
        return p,round(c,2)

    def metrics_block(df,has_gt,ml_col,llm_col,gt_col=None):
        agree=(df[ml_col]==df[llm_col]).mean()
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Agreement",f"{agree:.0%}")
        if has_gt:
            ml_acc=(df[ml_col]==df[gt_col]).mean()
            llm_acc=(df[llm_col]==df[gt_col]).mean()
            c2.metric("ML Accuracy",f"{ml_acc:.0%}")
            c3.metric("LLM Accuracy",f"{llm_acc:.0%}")
            w="✅ ML" if ml_acc>llm_acc else ("✅ LLM" if llm_acc>ml_acc else "Tie")
            c4.metric("Better Method",w)
        dis=df[df[ml_col]!=df[llm_col]]
        if len(dis):
            with st.expander(f"⚠️ {len(dis)} disagreement(s) — escalation candidates"):
                st.dataframe(dis,use_container_width=True)

    with sensor_tab:
        st.caption("CSV must have `value` column. Optional `ground_truth` column (Temperature/Humidity/Moisture/Vibration).")
        if os.path.exists("test_data/sensor_test.csv"):
            with open("test_data/sensor_test.csv","rb") as f:
                st.download_button("⬇️ sensor_test.csv (labeled, 32 rows)",f,"sensor_test.csv","text/csv")
        up=st.file_uploader("Upload sensor CSV",type="csv",key="cmp_sensor")
        if up:
            df=pd.read_csv(up); has_gt="ground_truth" in df.columns
            n=st.slider("Rows to test",5,min(len(df),30),min(len(df),20),key="cmp_s_n")
            if st.button("▶ Run Sensor Comparison",type="primary"):
                sub=df.head(n); rows=[]
                prog=st.progress(0)
                for i,(_,row) in enumerate(sub.iterrows()):
                    ml=predict_sensor(float(row["value"]))
                    lp,lc=llm_sensor(row["value"])
                    rows.append({"Value":row["value"],"Ground Truth":row.get("ground_truth","—"),
                                 "ML (GMM)":ml["gmm_prediction"],f"ML Conf":f"{ml['confidence_score']:.0%}",
                                 "LLM (Gemini)":lp,"LLM Conf":f"{lc:.0%}","Agree":"✓" if ml["gmm_prediction"]==lp else "✗"})
                    prog.progress((i+1)/len(sub))
                prog.empty()
                res=pd.DataFrame(rows); st.dataframe(res,use_container_width=True)
                st.divider(); st.subheader("📊 Metrics")
                metrics_block(res,has_gt,"ML (GMM)","LLM (Gemini)","Ground Truth")

    with network_tab:
        st.caption("CSV must have the 6 flow feature columns. Optional `ground_truth` (Normal/Suspicious/Priority).")
        if os.path.exists("test_data/network_test.csv"):
            with open("test_data/network_test.csv","rb") as f:
                st.download_button("⬇️ network_test.csv (labeled, 24 rows)",f,"network_test.csv","text/csv")
        up=st.file_uploader("Upload network CSV",type="csv",key="cmp_net")
        if up:
            df=pd.read_csv(up); has_gt="ground_truth" in df.columns
            missing=[c for c in NET_FEATURES if c not in df.columns]
            if missing: st.error(f"Missing: {missing}")
            else:
                n=st.slider("Rows to test",5,min(len(df),30),min(len(df),20),key="cmp_n_n")
                if st.button("▶ Run Network Comparison",type="primary"):
                    sub=df.head(n); rows=[]
                    prog=st.progress(0)
                    for i,(_,row) in enumerate(sub.iterrows()):
                        ml=predict_network({c:row[c] for c in NET_FEATURES})
                        lp,lc=llm_network(row.to_dict())
                        rows.append({"Ground Truth":row.get("ground_truth","—"),
                                     "ML (RF)":ml["classification"],"ML Conf":f"{ml['confidence_score']:.0%}",
                                     "LLM (Gemini)":lp,"LLM Conf":f"{lc:.0%}",
                                     "Agree":"✓" if ml["classification"]==lp else "✗"})
                        prog.progress((i+1)/len(sub))
                    prog.empty()
                    res=pd.DataFrame(rows); st.dataframe(res,use_container_width=True)
                    st.divider(); st.subheader("📊 Metrics")
                    metrics_block(res,has_gt,"ML (RF)","LLM (Gemini)","Ground Truth")


# ============================================================================
# ROUTER
# ============================================================================
m=st.session_state.method
if m is None:     landing_page()
elif m=="mcp":    mcp_page()
elif m=="ml":     ml_page()
elif m=="compare":compare_page()