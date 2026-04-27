"""
NeuroAI Unified Neuroimaging Streamlit App
Multi-Modal Analysis: MRI Brain · CT Head · EEG Parkinson's
"""

import streamlit as st
import os, io, glob, re, base64, datetime, warnings, traceback, time, tempfile
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image

warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroAI | Multi-Modal Neuroimaging",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base dark theme */
  .stApp { background-color: #0a0d14; color: #e8e8e8; }
  section[data-testid="stSidebar"] { background-color: #0d1023 !important; }
  section[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

  /* Inputs */
  .stTextInput > div > div > input,
  .stSelectbox > div > div,
  .stTextArea > div > div > textarea {
    background-color: #12152a !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a3060 !important;
    border-radius: 6px !important;
  }
  .stFileUploader > div {
    background-color: #12152a !important;
    border: 1.5px dashed #2a5080 !important;
    border-radius: 10px !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #1a237e, #4a148c) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #283593, #6a1b9a) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(106,27,154,0.4) !important;
  }

  /* Cards */
  .neuro-card {
    background: #12152a;
    border: 1px solid #1e2545;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
  }
  .diag-card {
    background: linear-gradient(135deg, #0d1b2a, #1a1040);
    border: 2px solid #4a148c;
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    text-align: center;
    margin: 0.8rem 0;
  }
  .diag-card h2 { color: #c77dff; margin: 0 0 0.3rem 0; font-size: 1.5rem; }
  .diag-card .conf { color: #e8e8e8; font-size: 1.1rem; }

  /* Status badges */
  .badge-urgent   { background:#f39c12; color:#000; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.82rem; }
  .badge-critical { background:#e74c3c; color:#fff; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.82rem; }
  .badge-routine  { background:#2ecc71; color:#000; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.82rem; }
  .badge-ok       { background:#00897B; color:#fff; padding:3px 10px; border-radius:20px; font-weight:700; font-size:0.82rem; }

  /* Section headers */
  .section-hdr {
    color: #00897B;
    font-size: 1rem;
    font-weight: 700;
    border-bottom: 1px solid #00897B;
    padding-bottom: 4px;
    margin: 1rem 0 0.6rem 0;
  }
  .modality-banner {
    background: linear-gradient(90deg, #1a237e, #4a148c);
    border-radius: 10px;
    padding: 0.8rem 1.4rem;
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
  }

  /* Metric chips */
  .metric-row { display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.6rem 0; }
  .metric-chip {
    background:#1a1f3a;
    border:1px solid #2a3060;
    border-radius:8px;
    padding:0.5rem 1rem;
    text-align:center;
    min-width:110px;
  }
  .metric-chip .label { font-size:0.72rem; color:#6a7aaa; }
  .metric-chip .value { font-size:1.1rem; font-weight:700; color:#c77dff; }

  /* Report text box */
  .report-section { background:#0e1120; border-left:3px solid #00897B; padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0; }
  .report-section strong { color:#90CAF9; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color:#0d1023; border-radius:10px; padding:4px; }
  .stTabs [data-baseweb="tab"]      { color:#8899bb; border-radius:8px; }
  .stTabs [aria-selected="true"]    { background-color:#1a237e !important; color:white !important; }

  /* Progress bar */
  .stProgress > div > div > div { background: linear-gradient(90deg,#4a148c,#1a237e); }

  /* Divider */
  hr { border-color: #1e2545; }

  /* Download btn */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #00695C, #00897B) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
  }

  /* Disclaimer box */
  .disclaimer-box {
    background:#1a0505;
    border:1px solid #c62828;
    border-radius:8px;
    padding:0.8rem 1rem;
    font-size:0.8rem;
    color:#ef9a9a;
    margin-top:0.8rem;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 NeuroAI Pipeline")
    st.markdown("**Multi-Modal Neuroimaging Analysis**")
    st.divider()

    st.markdown("### ⚙️ Configuration")
    models_dir = st.text_input(
        "Models Directory",
        value="/content/drive/MyDrive/Final_Models",
        help="Path to folder containing all .pth / .pt model checkpoint files"
    )
    gemini_key = st.text_input(
        "Key",
        type="password",
        help="Required for AI narrative generation"
    )
    gemini_model = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )

    st.divider()
    st.markdown("### 📋 Patient Demographics")
    pat_id   = st.text_input("Patient ID",    value="PT-2024-042")
    pat_age  = st.text_input("Age",           value="67")
    pat_sex  = st.selectbox("Sex",            ["Female","Male","Other","N/A"])
    pat_ref  = st.text_input("Referring Clinician", value="Dr. K. Sharma")
    pat_ind  = st.text_input("Clinical Indication",
                              value="Progressive headache + gait disturbance, 3 months")
    pat_inst = st.text_input("Institution",   value="NeuroImaging AI Research Pipeline v1.0")

    patient_meta = {
        "patient_id"          : pat_id,
        "patient_age"         : pat_age,
        "patient_sex"         : pat_sex,
        "referring_clinician" : pat_ref,
        "clinical_indication" : pat_ind,
        "institution"         : pat_inst,
        "scan_date"           : datetime.date.today().isoformat(),
    }

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#556080;line-height:1.5;">
    Pipeline:<br>
    Stage1 EfficientNet-B0 (F1=0.92)<br>
    Stage2-AD DenseNet121 (F1=0.928)<br>
    Stage2-Tumor EfficientNet-B0 (F1=1.00)<br>
    CT EfficientNet-B4 + TTA<br>
    EEG ResNet18 (STFT)
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a,#1a1040);
            border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.5rem;
            border:1px solid #2a2060;">
  <h1 style="margin:0;color:#c77dff;font-size:2rem;">🧠 NeuroAI Diagnostic Platform</h1>
  <p style="margin:0.4rem 0 0 0;color:#8899bb;font-size:1rem;">
    Unified Multi-Modal Neuroimaging Analysis &nbsp;·&nbsp;
    MRI Brain &nbsp;·&nbsp; CT Head &nbsp;·&nbsp; EEG Parkinson's
  </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY IMPORTS — only load heavy ML libs when actually needed
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML libraries…")
def _import_libs():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as tvF
    import timm
    from monai.transforms import (Compose, LoadImage, EnsureChannelFirst,
                                   NormalizeIntensity, Resize, ToTensor)
    from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from scipy.signal import spectrogram as scipy_spectrogram
    import google.generativeai as genai
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
        Image as RLImage, Table, TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    return {
        "torch": torch, "nn": nn, "F": F, "tv_models": tv_models,
        "transforms": transforms, "tvF": tvF, "timm": timm,
        "Compose": Compose, "LoadImage": LoadImage, "EnsureChannelFirst": EnsureChannelFirst,
        "NormalizeIntensity": NormalizeIntensity, "Resize": Resize, "ToTensor": ToTensor,
        "GradCAMPlusPlus": GradCAMPlusPlus, "GradCAM": GradCAM,
        "show_cam_on_image": show_cam_on_image,
        "ClassifierOutputTarget": ClassifierOutputTarget,
        "scipy_spectrogram": scipy_spectrogram,
        "genai": genai,
        "A4": A4, "mm": mm, "getSampleStyleSheet": getSampleStyleSheet,
        "ParagraphStyle": ParagraphStyle, "HexColor": HexColor, "white": white,
        "SimpleDocTemplate": SimpleDocTemplate, "Paragraph": Paragraph,
        "Spacer": Spacer, "RLImage": RLImage, "Table": Table,
        "TableStyle": TableStyle, "HRFlowable": HRFlowable, "PageBreak": PageBreak,
        "TA_CENTER": TA_CENTER, "TA_JUSTIFY": TA_JUSTIFY,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models…")
def load_all_models(models_dir_str):
    L = _import_libs()
    torch = L["torch"]; nn = L["nn"]; F = L["F"]
    tv_models = L["tv_models"]; timm = L["timm"]
    Compose = L["Compose"]; LoadImage = L["LoadImage"]
    EnsureChannelFirst = L["EnsureChannelFirst"]
    NormalizeIntensity = L["NormalizeIntensity"]
    Resize = L["Resize"]; ToTensor = L["ToTensor"]
    transforms = L["transforms"]

    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MDIR = Path(models_dir_str)

    # ── MRI architectures ─────────────────────────────────────────────────────
    def _build_stage1():
        m = tv_models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.35), nn.Linear(1280,256),
            nn.SiLU(), nn.Dropout(0.20), nn.Linear(256,3))
        return m

    class Stage2_AD(nn.Module):
        def __init__(self):
            super().__init__()
            bb = tv_models.densenet121(weights=None)
            self.features   = bb.features
            self.pool       = nn.AdaptiveAvgPool2d((1,1))
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.4), nn.Linear(1024,256),
                nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.25), nn.Linear(256,2))
        def forward(self,x): return self.classifier(self.pool(self.features(x)))
        def predict_probs(self,logits):
            p = torch.sigmoid(logits.float())
            return torch.stack([1-p[:,0], p[:,0]-p[:,1], p[:,1]], dim=1).clamp(min=1e-6)

    class Stage2_Tumor(nn.Module):
        def __init__(self):
            super().__init__()
            bb = tv_models.efficientnet_b0(weights=None)
            self.features   = bb.features
            self.pool       = bb.avgpool
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.30), nn.Linear(1280,128),
                nn.BatchNorm1d(128), nn.SiLU(inplace=True), nn.Dropout(0.15), nn.Linear(128,3))
        def forward(self,x): return self.classifier(self.pool(self.features(x)))

    # ── CT architectures ──────────────────────────────────────────────────────
    class CTClassifier(nn.Module):
        def __init__(self, backbone, n_cls, drop=0.40):
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
            d = self.backbone.num_features
            self.drop = nn.Dropout(drop)
            self.head = nn.Sequential(
                nn.Linear(d,512), nn.LayerNorm(512), nn.GELU(),
                nn.Dropout(drop/2), nn.Linear(512,n_cls))
        def forward(self,x): return self.head(self.drop(self.backbone(x)))

    class CTTempScaler(nn.Module):
        def __init__(self,m,T=1.0):
            super().__init__()
            self.model=m; self.temperature=nn.Parameter(torch.tensor([T]))
        def forward(self,x): return self.model(x)/self.temperature.clamp(min=0.01)

    def _safe_load(p):
        try: return torch.load(p, map_location=DEVICE, weights_only=True)
        except: return torch.load(p, map_location=DEVICE, weights_only=False)

    results = {"device": str(DEVICE), "mri": {}, "ct": {}, "eeg": None, "logs": []}

    # Load MRI models — flat folder, no subfolders required
    MRI_PATTERNS = {
        "s1"   : ["stage1_best.pth", "stage1_best.pt",
                  "*stage1*.pth", "*stage1*.pt",
                  "*MRI*stage1*.pth", "*MRI*stage1*.pt"],
        "s2_ad": ["stage2_ad_densenet_best.pth", "stage2_ad_densenet_best.pt",
                  "*stage2*ad*.pth",  "*stage2*ad*.pt",
                  "*stage2*alzheimer*.pth", "*stage2*alzheimer*.pt",
                  "*stage2*AD*.pth",  "*stage2*AD*.pt",
                  "*AD_best*.pth",    "*AD_best*.pt"],
        "s2_tu": ["stage2_tumor_best.pth", "stage2_tumor_best.pt",
                  "*stage2*tumor*.pth", "*stage2*tumor*.pt",
                  "*stage2*Tumor*.pth", "*stage2*Tumor*.pt",
                  "*tumor_best*.pth",   "*tumor_best*.pt"],
    }
    MRI_CLS = {"s1": _build_stage1, "s2_ad": Stage2_AD, "s2_tu": Stage2_Tumor}
    for tag, pats in MRI_PATTERNS.items():
        found = None
        for pat in pats:
            exact = MDIR / pat
            if exact.exists(): found = exact; break
            hits = sorted(glob.glob(str(MDIR / pat)))
            if hits: found = Path(hits[-1]); break
        if found:
            try:
                ck = _safe_load(found)
                m  = MRI_CLS[tag]().to(DEVICE)
                m.load_state_dict(ck.get("model_state_dict", ck), strict=False)
                m.eval(); results["mri"][tag] = m
                results["logs"].append(f"✅ MRI {tag}  ←  {found.name}")
            except Exception as e:
                results["logs"].append(f"❌ MRI {tag}: {e}")
        else:
            results["logs"].append(f"⚠️ MRI {tag} — not found in {MDIR}")

    # Load CT models — flat folder
    CT_SPECS = {1:("efficientnet_b4",3,0.40,"Model1_General_EfficientNetB4"),
                2:("densenet121",2,0.40,"Model2_Stroke_DenseNet121"),
                3:("efficientnet_b1",2,0.50,"Model3_Vascular_EfficientNetB1")}
    for mid,(backbone,n_cls,drop,pattern) in CT_SPECS.items():
        ckpts = (glob.glob(str(MDIR/f"{pattern}*.pt")) + glob.glob(str(MDIR/f"{pattern}*.pth")) +
                 glob.glob(str(MDIR/f"*{pattern}*.pt")) + glob.glob(str(MDIR/f"*{pattern}*.pth")))
        ckpts = sorted(set(ckpts))
        if not ckpts: results["logs"].append(f"⚠️ CT Model {mid} ({pattern}) — not found in {MDIR}"); continue
        try:
            ck  = _safe_load(sorted(ckpts)[-1])
            sd  = ck.get("state_dict", ck)
            has_T = any(k.startswith("temperature") or k.startswith("model.") for k in sd)
            base = CTClassifier(backbone,n_cls,drop)
            if has_T:
                base_sd = {k.replace("model.",""):v for k,v in sd.items() if k.startswith("model.")}
                T_val   = float(sd.get("temperature", torch.tensor([1.0])).item()
                               if isinstance(sd.get("temperature"), torch.Tensor)
                               else sd.get("temperature", 1.0))
                base.load_state_dict(base_sd, strict=False)
                model = CTTempScaler(base, T=T_val)
            else:
                base.load_state_dict(sd, strict=False); model = base
            model.eval().to(DEVICE)
            results["ct"][mid] = model
            results["logs"].append(f"✅ CT Model {mid}")
        except Exception as e:
            results["logs"].append(f"❌ CT Model {mid}: {e}")

    # Load EEG model — flat folder
    patterns = ["*EEG*.pt","*EEG*.pth","*eeg*.pt","*eeg*.pth",
                "*parkinson*.pt","*parkinson*.pth","*Parkinson*.pt","*Parkinson*.pth",
                "best_model*.pt","best_model*.pth"]
    for pat in patterns:
        ckpts = glob.glob(str(MDIR/pat))
        if ckpts:
            try:
                m = tv_models.resnet18(weights=None)
                m.fc = nn.Linear(m.fc.in_features, 2)
                sd = _safe_load(sorted(ckpts)[-1])
                sd = sd.get("state_dict", sd.get("model_state_dict", sd))
                m.load_state_dict(sd, strict=False); m.eval().to(DEVICE)
                results["eeg"] = m
                results["logs"].append("✅ EEG model")
                break
            except Exception as e:
                results["logs"].append(f"❌ EEG: {e}")
    if results["eeg"] is None:
        results["logs"].append("⚠️ EEG model not found")

    results["CTClassifier"] = CTClassifier
    results["CTTempScaler"] = CTTempScaler
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
STAGE1_NAMES = ["Normal","Alzheimer","Tumor"]
AD_NAMES     = ["Alzheimer_VeryMild","Alzheimer_Mild","Alzheimer_Moderate"]
TUMOR_NAMES  = ["Tumor_Glioma","Tumor_Meningioma","Tumor_Pituitary"]

CT_MAIN_CLS  = ["Normal","Stroke","Vascular"]
CT_STROKE_CLS= ["Ischemic_Stroke","Hemorrhagic_Stroke"]
CT_VASC_CLS  = ["Intracranial_Hemorrhage","Aneurysm"]
CT_URGENCY   = {"Normal":"Routine","Ischemic_Stroke":"URGENT","Hemorrhagic_Stroke":"CRITICAL",
                "Intracranial_Hemorrhage":"CRITICAL","Aneurysm":"URGENT"}
CT_DESC      = {"Normal":"No acute intracranial abnormality identified.",
                "Ischemic_Stroke":"Ischemic stroke — hypodense area indicating tissue infarction.",
                "Hemorrhagic_Stroke":"Hemorrhagic stroke — hyperdense blood within brain parenchyma.",
                "Intracranial_Hemorrhage":"Intracranial hemorrhage — blood products within the cranial vault.",
                "Aneurysm":"Vascular abnormality consistent with intracranial aneurysm."}

IMG_SIZE = 224
_X_ZONES = [(0,65,"left hemisphere"),(65,159,"central / midline"),(159,224,"right hemisphere")]
_Y_ZONES = [(0,55,"vertex / superior convexity"),(55,111,"frontal-parietal"),
            (111,167,"temporal / thalamic / basal ganglia"),(167,224,"occipital / posterior fossa")]


def _region(x, y):
    x, y = max(0,min(x,223)), max(0,min(y,223))
    xz = next(l for a,b,l in _X_ZONES if a<=x<b)
    yz = next(l for a,b,l in _Y_ZONES if a<=y<b)
    return f"{xz}, {yz}"

def _centroid(cam):
    thr = np.percentile(cam, 95); mask = cam >= thr
    ys, xs = np.where(mask)
    if len(xs) == 0: cy,cx = np.unravel_index(cam.argmax(), cam.shape)
    else:
        w = cam[mask]
        cx,cy = int(np.average(xs,weights=w)), int(np.average(ys,weights=w))
    return cx, cy

def analyse_cam(cam):
    cx, cy = _centroid(cam); spread = float(np.std(cam))
    return {"centroid":(cx,cy),"region":_region(cx,cy),
            "coverage":round(float(np.mean(cam>np.percentile(cam,50))),3),
            "peak":round(float(cam.max()),3),"spread":round(spread,3),
            "attention":"diffuse" if spread>0.20 else "focal"}

def _compute_cam(L, model, tensor, cls_idx, layer_fn):
    torch = L["torch"]; GradCAMPlusPlus = L["GradCAMPlusPlus"]
    GradCAM = L["GradCAM"]; ClassifierOutputTarget = L["ClassifierOutputTarget"]
    DEVICE = next(model.parameters()).device
    layers = layer_fn(model)
    if not layers: return np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.float32)
    inp = tensor.to(DEVICE)
    for CAMClass in [GradCAMPlusPlus, GradCAM]:
        try:
            with torch.enable_grad():
                with CAMClass(model=model, target_layers=layers) as cam:
                    hm = cam(inp, [ClassifierOutputTarget(cls_idx)])[0]
            if hm.max() > 1e-6: return hm
        except: pass
    return np.zeros((IMG_SIZE,IMG_SIZE),dtype=np.float32)

def _effnet_layer(model):
    feats = getattr(model,"features",None)
    if feats:
        convs = [m for m in feats.modules() if hasattr(m,"__class__") and "Conv2d" in str(type(m))]
        if convs: return [convs[-1]]
    convs = [m for m in model.modules() if "Conv2d" in str(type(m))]
    return [convs[-1]] if convs else None

def _densenet_layer(model):
    if hasattr(model,"features") and hasattr(model.features,"denseblock4"):
        for _,layer in reversed(list(model.features.denseblock4.named_children())):
            convs = [m for m in layer.modules() if "Conv2d" in str(type(m))]
            if convs: return [convs[-1]]
    convs = [m for m in model.modules() if "Conv2d" in str(type(m))]
    return [convs[-1]] if convs else None

def _last_conv(model):
    base = getattr(model,"model",model)
    convs = [m for m in base.modules() if "Conv2d" in str(type(m))]
    return [convs[-1]] if convs else None


# ── MRI inference ─────────────────────────────────────────────────────────────
def run_mri(image_path, mri_models, L):
    torch = L["torch"]; F = L["F"]; transforms = L["transforms"]
    Compose = L["Compose"]; LoadImage = L["LoadImage"]
    EnsureChannelFirst = L["EnsureChannelFirst"]
    NormalizeIntensity = L["NormalizeIntensity"]
    Resize = L["Resize"]; ToTensor = L["ToTensor"]
    DEVICE = next(list(mri_models.values())[0].parameters()).device

    class EnsureRGB:
        def __call__(self,img):
            if isinstance(img, torch.Tensor):
                c = img.shape[0]
                if c==1: return img.repeat(3,1,1)
                if c==3: return img
                if c==4: return img[:3]
            return img

    s1_tf = Compose([LoadImage(image_only=True), EnsureChannelFirst(),
                     Resize(spatial_size=(224,224)), EnsureRGB(),
                     NormalizeIntensity(nonzero=True,channel_wise=True), ToTensor()])
    s2_tf = transforms.Compose([transforms.Resize((224,224)),
                                  transforms.Grayscale(num_output_channels=3),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    raw_tf = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Resize(spatial_size=(224,224))])

    s1_inp = s1_tf(str(image_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        s1_logits = mri_models["s1"](s1_inp)
        s1_probs  = F.softmax(s1_logits.float(),dim=1).cpu().numpy()[0]
    s1_idx  = int(s1_probs.argmax())
    s1_cam  = _compute_cam(L, mri_models["s1"], s1_inp, s1_idx, _effnet_layer)
    raw     = raw_tf(str(image_path))[0].numpy()
    g       = (raw-raw.min())/(raw.max()-raw.min()+1e-8)
    viz     = np.stack([g,g,g],axis=-1).astype(np.float32)

    result = {"image_path":str(image_path),"viz_rgb":viz,"s1_probs":s1_probs,
              "s1_pred_idx":s1_idx,"s1_cam":s1_cam,"s2_probs":None,"s2_names":None,
              "s2_pred_idx":None,"route":"normal","final_name":"Normal",
              "final_conf":float(s1_probs[0])}

    s2_inp = s2_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    g_str  = STAGE1_NAMES[s1_idx]

    if g_str=="Alzheimer" and "s2_ad" in mri_models:
        with torch.no_grad():
            logits   = mri_models["s2_ad"](s2_inp)
            s2_probs = mri_models["s2_ad"].predict_probs(logits).cpu().numpy()[0]
        s2_idx = int(s2_probs.argmax())
        result.update({"s2_probs":s2_probs,"s2_names":AD_NAMES,"s2_pred_idx":s2_idx,
                       "route":"alzheimer","final_name":AD_NAMES[s2_idx],"final_conf":float(s2_probs[s2_idx])})
    elif g_str=="Tumor" and "s2_tu" in mri_models:
        with torch.no_grad():
            logits   = mri_models["s2_tu"](s2_inp)
            s2_probs = F.softmax(logits.float(),dim=1).cpu().numpy()[0]
        s2_idx = int(s2_probs.argmax())
        result.update({"s2_probs":s2_probs,"s2_names":TUMOR_NAMES,"s2_pred_idx":s2_idx,
                       "route":"tumor","final_name":TUMOR_NAMES[s2_idx],"final_conf":float(s2_probs[s2_idx])})

    result["cam_analysis"] = analyse_cam(s1_cam)
    return result


# ── CT inference ──────────────────────────────────────────────────────────────
def run_ct(image_path, ct_models, L):
    torch = L["torch"]; F = L["F"]
    DEVICE = next(list(ct_models.values())[0].parameters()).device

    def _brain_window(arr):
        arr = np.clip(arr.astype(np.float32),-40,120)
        return ((arr+40)/160*255).astype(np.uint8)

    ext = Path(image_path).suffix.lower()
    if ext in (".dcm",".dicom"):
        import pydicom
        ds  = pydicom.dcmread(str(image_path),force=True)
        arr = ds.pixel_array.astype(np.float32)
        if hasattr(ds,"RescaleSlope"): arr = arr*float(ds.RescaleSlope)+float(getattr(ds,"RescaleIntercept",0))
        raw = _brain_window(arr)
    else:
        raw = np.array(Image.open(image_path).convert("L")).astype(np.uint8)
    raw      = cv2.resize(raw,(IMG_SIZE,IMG_SIZE))
    enhanced = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(raw)

    t = torch.from_numpy(raw).float()/255.0
    t = t.unsqueeze(0).repeat(3,1,1); tensor = (t-0.5)/0.5

    def _tta(t):
        t1 = torch.flip(t,dims=[2])
        a  = np.radians(8); c,s = float(np.cos(a)),float(np.sin(a))
        th = torch.tensor([[c,-s,0],[s,c,0]],dtype=torch.float32)
        g  = F.affine_grid(th.unsqueeze(0),t.unsqueeze(0).shape,align_corners=False)
        t2 = F.grid_sample(t.unsqueeze(0).float(),g,align_corners=False,padding_mode="border")[0]
        return [t,t1,t2]

    @torch.no_grad()
    def _probs(m,tensors):
        logits=[m(tt.unsqueeze(0).to(DEVICE)).float() for tt in tensors]
        return F.softmax(torch.stack(logits).mean(0),1).cpu().numpy()[0]

    tta   = _tta(tensor)
    p1    = _probs(ct_models[1],tta)
    main_idx = int(p1.argmax()); main_cls = CT_MAIN_CLS[main_idx]
    hm1   = _compute_cam(L, ct_models[1], tta[0].unsqueeze(0), main_idx, _last_conv)

    result = {"image_path":str(image_path),"raw_gray":raw,"enhanced_gray":enhanced,
              "tensor":tensor,"main_class":main_cls,"main_conf":float(p1[main_idx]),
              "main_probs":p1,"fine_class":main_cls,"fine_conf":float(p1[main_idx]),
              "fine_probs":None,"fine_cls_names":CT_MAIN_CLS,"hm":hm1,
              "uncertain":float(p1[main_idx])<0.55}

    SPEC = {"Stroke":(2,CT_STROKE_CLS),"Vascular":(3,CT_VASC_CLS)}
    if main_cls in SPEC and float(p1[main_idx])>=0.55 and SPEC[main_cls][0] in ct_models:
        mid2,cls2 = SPEC[main_cls]
        p2   = _probs(ct_models[mid2],tta); fi=int(p2.argmax())
        hm2  = _compute_cam(L, ct_models[mid2], tta[0].unsqueeze(0), fi, _last_conv)
        result.update({"fine_class":cls2[fi],"fine_conf":float(p2[fi]),"fine_probs":p2,
                       "fine_cls_names":cls2,"hm":hm2,"uncertain":float(p2[fi])<0.55})
    return result


# ── EEG inference ─────────────────────────────────────────────────────────────
def run_eeg(eeg_path, eeg_model, L):
    torch = L["torch"]; tvF = L["tvF"]
    scipy_spectrogram = L["scipy_spectrogram"]
    GradCAM = L["GradCAM"]
    DEVICE = next(eeg_model.parameters()).device

    import mne; mne.set_log_level("ERROR")
    try: raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True)
    except Exception as e: raise RuntimeError(f"EEG load failed: {e}")

    data = raw.get_data().astype("float32"); n_ch = min(3,data.shape[0])
    imgs = []
    for ch in range(n_ch):
        _,_,Sxx = scipy_spectrogram(data[ch],fs=256,nperseg=256,noverlap=128)
        S=(np.log1p(Sxx)); S=(S-S.min())/(S.max()-S.min()+1e-8); imgs.append(S)
    while len(imgs)<3: imgs.append(np.zeros_like(imgs[0]))
    mean_ = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std_  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    t = (torch.tensor(np.stack(imgs)).float()-mean_)/std_
    t = tvF.resize(t,[IMG_SIZE,IMG_SIZE],antialias=True)
    batch = t.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.softmax(eeg_model(batch),dim=1).cpu().numpy()[0]
    pk = float(prob[1])
    if pk>=0.70: pred,conf = "Parkinson's Disease",pk
    else:        pred,conf = "Control (Healthy)",float(prob[0])
    cam_gen = GradCAM(model=eeg_model, target_layers=[eeg_model.layer4[-1]])
    with torch.enable_grad():
        hm = cam_gen(input_tensor=batch,targets=None)[0]
    return {"eeg_path":str(eeg_path),"tensor":t,"prediction":pred,
            "confidence":conf,"probs":prob,"hm":hm}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════
BG="#0a0d14"; BG2="#12122a"; TW="#e8e8e8"; TD="#777777"

def _hp(ax, cam, label):
    ax.set_facecolor("black")
    if cam is not None and cam.max()>1e-6:
        norm=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        ax.imshow(norm,cmap="hot",aspect="auto",vmin=0,vmax=1,interpolation="bilinear")
    else:
        ax.imshow(np.zeros((224,224)),cmap="hot",aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#ff6a00")
    ax.text(0.03,0.97,f"GRAD-CAM  {label}",transform=ax.transAxes,
            ha="left",va="top",color="#ffaa44",fontsize=7.5,fontweight="bold",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3",facecolor="#1a0800",edgecolor="#ff6a00",linewidth=1.2))

def _cp(ax, viz, cam, title, tc, show_cam):
    if cam is not None and cam.max()>1e-6 and show_cam:
        from pytorch_grad_cam.utils.image import show_cam_on_image
        ax.imshow(show_cam_on_image(viz,cam,use_rgb=True),aspect="auto")
        thr=np.percentile(cam,95); ys,xs=np.where(cam>=thr)
        if len(xs):
            w=cam[cam>=thr]; cx=int(np.average(xs,weights=w)); cy=int(np.average(ys,weights=w))
        else: cy,cx=np.unravel_index(cam.argmax(),cam.shape)
        ax.axvline(cx,color="yellow",lw=0.9,alpha=0.85)
        ax.axhline(cy,color="yellow",lw=0.9,alpha=0.85)
        ax.plot(cx,cy,"+",color="yellow",ms=11,mew=1.6)
    else: ax.imshow(viz,cmap="gray",aspect="auto")
    ax.set_title(title,color=tc,fontsize=8,pad=4,fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#2a2d3e")

def _bp(ax, names, probs, hi_idx, title, bc, hc):
    ax.set_facecolor(BG2)
    for s in ax.spines.values(): s.set_visible(False)
    pairs=sorted(zip(probs,names),key=lambda x:x[0])
    hi_n=names[hi_idx].replace("_"," ")
    sp_=[p for p,_ in pairs]; sn_=[n.replace("_"," ") for _,n in pairs]
    bars=ax.barh(sn_,[p*100 for p in sp_],
                 color=[hc if n==hi_n else bc for n in sn_],edgecolor="none",height=0.52)
    for bar,prob in zip(bars,sp_):
        ax.text(min(bar.get_width()+1.2,94),bar.get_y()+bar.get_height()/2,
                f"{prob*100:.1f}%",va="center",ha="left",color=TW,fontsize=9,fontweight="bold")
    ax.set_xlim(0,115); ax.set_xlabel("Confidence (%)",color=TD,fontsize=8)
    ax.set_title(title,color=TW,fontsize=9,pad=5,fontweight="bold")
    ax.tick_params(colors=TW,labelsize=8); ax.xaxis.label.set_color(TD)

def make_mri_figure(result):
    ROUTE_COLOR={"normal":"#00d4aa","alzheimer":"#ff7f32","tumor":"#c77dff"}
    BLUE="#4a9eff"; PURPLE="#c77dff"; PURPLE2="#9d4edd"
    route=result["route"]; conf=result["final_conf"]; rc=ROUTE_COLOR.get(route,TW)
    has_s2=result["s2_probs"] is not None; nrows=2 if has_s2 else 1
    viz=result["viz_rgb"]; fig_h=5.0*nrows+1.6
    fig=plt.figure(figsize=(20,fig_h),facecolor=BG)
    hdr=0.40/fig_h; ftr=0.38/fig_h; pad=0.06/fig_h
    _ax=fig.add_axes([0,1-hdr,1,hdr]); _ax.set_facecolor("#1a1025"); _ax.axis("off")
    diag=result["final_name"].replace("_"," ").upper()
    _ax.text(0.5,0.5,f"► {diag}   ·   Report Confidence: {conf:.1%}",
             transform=_ax.transAxes,ha="center",va="center",
             color=rc,fontsize=11,fontweight="bold",fontfamily="monospace")
    _fx=fig.add_axes([0,0,1,ftr]); _fx.set_facecolor("#110e1a"); _fx.axis("off")
    _fx.text(0.5,0.5,f"FINAL PREDICTION:  {result['final_name'].replace('_',' ')}  │  confidence {conf:.1%}",
             transform=_fx.transAxes,ha="center",va="center",
             color=rc,fontsize=9.5,fontweight="bold",fontfamily="monospace")
    gs=gridspec.GridSpec(nrows,4,figure=fig,left=0.03,right=0.99,
                         bottom=ftr+pad,top=1-hdr-pad,wspace=0.06,hspace=0.15,width_ratios=[1,1,1,1.05])
    ax00=fig.add_subplot(gs[0,0]); ax00.set_facecolor(BG)
    ax01=fig.add_subplot(gs[0,1]); ax01.set_facecolor(BG)
    ax02=fig.add_subplot(gs[0,2]); ax02.set_facecolor("black")
    ax03=fig.add_subplot(gs[0,3])
    ax00.imshow(viz,cmap="gray",aspect="auto")
    ax00.set_title("MRI INPUT  │  EfficientNet-B0  (MONAI)",color=TD,fontsize=8,pad=4)
    ax00.set_xticks([]); ax00.set_yticks([])
    ax00.set_ylabel("Stage 1",color=TD,fontsize=8,rotation=90,labelpad=3)
    _cp(ax01,viz,result["s1_cam"],"STAGE 1  GradCAM++",PURPLE,True)
    _hp(ax02,result["s1_cam"],"Stage 1")
    _bp(ax03,STAGE1_NAMES,result["s1_probs"],result["s1_pred_idx"],"Group Detection",BLUE,rc)
    if has_s2:
        ax10=fig.add_subplot(gs[1,0]); ax10.set_facecolor(BG)
        ax11=fig.add_subplot(gs[1,1]); ax11.set_facecolor(BG)
        ax12=fig.add_subplot(gs[1,2]); ax12.set_facecolor("black")
        ax13=fig.add_subplot(gs[1,3])
        ax10.imshow(viz,cmap="gray",aspect="auto")
        s2_arch="DenseNet121 (Ordinal)" if route=="alzheimer" else "EfficientNet-B0"
        ax10.set_title(f"MRI INPUT  │  {s2_arch}",color=TD,fontsize=8,pad=4)
        ax10.set_xticks([]); ax10.set_yticks([])
        ax10.set_ylabel("Stage 2",color=TD,fontsize=8,rotation=90,labelpad=3)
        _cp(ax11,viz,result["s1_cam"],"STAGE 2  GradCAM++",PURPLE2,True)
        _hp(ax12,result["s1_cam"],"Stage 2")
        _bp(ax13,result["s2_names"],result["s2_probs"],result["s2_pred_idx"],"Subtype Detection",PURPLE2,rc)
    fig.patch.set_facecolor(BG)
    return fig

def make_ct_figure(result):
    URG_COLOR={"Routine":"#2ecc71","URGENT":"#f39c12","CRITICAL":"#e74c3c"}
    BLUE="#4a9eff"
    fine=result["fine_class"]; fine_conf=result["fine_conf"]
    urgency=CT_URGENCY.get(fine,"Routine"); uc=URG_COLOR[urgency]
    has_s2=result["fine_probs"] is not None; nrows=2 if has_s2 else 1
    rgb=(result["tensor"].cpu().numpy().transpose(1,2,0)*0.5+0.5).clip(0,1).astype(np.float32)
    fig_h=5.0*nrows+1.6
    fig=plt.figure(figsize=(20,fig_h),facecolor=BG)
    hdr=0.40/fig_h; ftr=0.38/fig_h; pad=0.06/fig_h
    _ax=fig.add_axes([0,1-hdr,1,hdr]); _ax.set_facecolor("#0d1123"); _ax.axis("off")
    _ax.text(0.5,0.5,f"► {fine.replace('_',' ').upper()}   ·   {fine_conf:.1%}   ·   {urgency}",
             transform=_ax.transAxes,ha="center",va="center",
             color=uc,fontsize=11,fontweight="bold",fontfamily="monospace")
    _fx=fig.add_axes([0,0,1,ftr]); _fx.set_facecolor("#0a0e1c"); _fx.axis("off")
    _fx.text(0.5,0.5,f"FINAL PREDICTION:  {fine.replace('_',' ')}  │  urgency: {urgency}  │  confidence {fine_conf:.1%}",
             transform=_fx.transAxes,ha="center",va="center",
             color=uc,fontsize=9.5,fontweight="bold",fontfamily="monospace")
    gs=gridspec.GridSpec(nrows,4,figure=fig,left=0.03,right=0.99,
                         bottom=ftr+pad,top=1-hdr-pad,wspace=0.06,hspace=0.15,width_ratios=[1,1,1,1.05])
    ax00=fig.add_subplot(gs[0,0]); ax00.set_facecolor(BG)
    ax01=fig.add_subplot(gs[0,1]); ax01.set_facecolor(BG)
    ax02=fig.add_subplot(gs[0,2]); ax02.set_facecolor("black")
    ax03=fig.add_subplot(gs[0,3])
    ax00.imshow(result["enhanced_gray"],cmap="gray",aspect="auto")
    ax00.set_title("ENHANCED CT  │  EfficientNet-B4  (CLAHE+TTA)",color=TD,fontsize=8,pad=4)
    ax00.set_xticks([]); ax00.set_yticks([])
    ax00.set_ylabel("Stage 1",color=TD,fontsize=8,rotation=90,labelpad=3)
    _cp(ax01,rgb,result["hm"],"STAGE 1  GradCAM++","#c77dff",True)
    _hp(ax02,result["hm"],"Stage 1")
    _bp(ax03,CT_MAIN_CLS,result["main_probs"],CT_MAIN_CLS.index(result["main_class"]),"Main Category",BLUE,uc)
    if has_s2:
        cls_names=result["fine_cls_names"]
        fi=cls_names.index(fine) if fine in cls_names else int(np.argmax(result["fine_probs"]))
        arch2="DenseNet121" if result["main_class"]=="Stroke" else "EfficientNet-B1"
        ax10=fig.add_subplot(gs[1,0]); ax10.set_facecolor(BG)
        ax11=fig.add_subplot(gs[1,1]); ax11.set_facecolor(BG)
        ax12=fig.add_subplot(gs[1,2]); ax12.set_facecolor("black")
        ax13=fig.add_subplot(gs[1,3])
        ax10.imshow(result["enhanced_gray"],cmap="gray",aspect="auto")
        ax10.set_title(f"ENHANCED CT  │  {arch2}",color=TD,fontsize=8,pad=4)
        ax10.set_xticks([]); ax10.set_yticks([])
        ax10.set_ylabel("Stage 2",color=TD,fontsize=8,rotation=90,labelpad=3)
        _cp(ax11,rgb,result["hm"],"STAGE 2  GradCAM++","#f39c12",True)
        _hp(ax12,result["hm"],"Stage 2")
        _bp(ax13,cls_names,result["fine_probs"],fi,"Fine Classification","#f39c12",uc)
    fig.patch.set_facecolor(BG); return fig

def make_eeg_figure(result):
    import torch as _torch
    BLUE="#4a9eff"
    pred=result["prediction"]; conf=result["confidence"]
    pc="#e74c3c" if "Parkinson" in pred else "#00d4aa"
    hdr_bg="#1a0d0d" if "Parkinson" in pred else "#0d1a0d"
    EEG_MEAN=_torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    EEG_STD =_torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    fig_h=5.8
    fig=plt.figure(figsize=(20,fig_h),facecolor=BG)
    hdr=0.40/fig_h; ftr=0.38/fig_h; pad=0.06/fig_h
    _ax=fig.add_axes([0,1-hdr,1,hdr]); _ax.set_facecolor(hdr_bg); _ax.axis("off")
    _ax.text(0.5,0.5,f"► {pred.upper()}   ·   Confidence: {conf:.1%}",
             transform=_ax.transAxes,ha="center",va="center",
             color=pc,fontsize=11,fontweight="bold",fontfamily="monospace")
    _fx=fig.add_axes([0,0,1,ftr]); _fx.set_facecolor(hdr_bg); _fx.axis("off")
    _fx.text(0.5,0.5,f"FINAL: {pred}  │  PD: {result['probs'][1]:.1%}  │  HC: {result['probs'][0]:.1%}  │  threshold: 70%",
             transform=_fx.transAxes,ha="center",va="center",
             color=pc,fontsize=9.5,fontweight="bold",fontfamily="monospace")
    gs=gridspec.GridSpec(1,4,figure=fig,left=0.03,right=0.99,
                         bottom=ftr+pad,top=1-hdr-pad,wspace=0.07,width_ratios=[1,1,1,1.05])
    ax0=fig.add_subplot(gs[0,0]); ax0.set_facecolor(BG)
    ax1=fig.add_subplot(gs[0,1]); ax1.set_facecolor(BG)
    ax2=fig.add_subplot(gs[0,2]); ax2.set_facecolor("black")
    ax3=fig.add_subplot(gs[0,3])
    img_2d=(result["tensor"]*EEG_STD+EEG_MEAN).mean(0).numpy()
    half=img_2d.shape[0]//2
    ax0.imshow(img_2d[:half],aspect="auto",cmap="jet",origin="lower",
               vmin=img_2d.min(),vmax=np.percentile(img_2d,95))
    ax0.set_title("STFT SPECTROGRAM  │  ResNet18",color=TD,fontsize=8,pad=4)
    ax0.set_ylabel("Freq Bins (0–64 Hz)",color=TD,fontsize=7.5)
    ax0.set_xlabel("Time Windows",color=TD,fontsize=7.5)
    ax0.tick_params(colors=TD,labelsize=7)
    for s in ax0.spines.values(): s.set_color("#2a2d3e")
    rgb=(result["tensor"]*EEG_STD+EEG_MEAN).permute(1,2,0).numpy().clip(0,1).astype(np.float32)
    hm=result["hm"]; h2=hm.shape[0]//2
    if hm.max()>1e-6:
        from pytorch_grad_cam.utils.image import show_cam_on_image
        ov=show_cam_on_image(rgb,hm,use_rgb=True)
        ax1.imshow(ov[:h2],aspect="auto",origin="lower")
        hm_h=hm[:h2]; cy,cx=np.unravel_index(hm_h.argmax(),hm_h.shape)
        ax1.axvline(cx,color="yellow",lw=0.9); ax1.axhline(cy,color="yellow",lw=0.9)
        ax1.plot(cx,cy,"+",color="yellow",ms=11,mew=1.6)
    else: ax1.imshow(rgb[:h2],aspect="auto",origin="lower")
    ax1.set_title("GRAD-CAM  │  layer4[-1]",color="#c77dff",fontsize=8,pad=4,fontweight="bold")
    ax1.set_xlabel("Time Windows",color=TD,fontsize=7.5)
    ax1.tick_params(colors=TD,labelsize=7)
    for s in ax1.spines.values(): s.set_color("#2a2d3e")
    hm_h2=hm[:h2]
    ax2.set_facecolor("black")
    if hm.max()>1e-6:
        norm=(hm_h2-hm_h2.min())/(hm_h2.max()-hm_h2.min()+1e-8)
        ax2.imshow(norm,cmap="hot",aspect="auto",vmin=0,vmax=1,origin="lower",interpolation="bilinear")
    else: ax2.imshow(np.zeros_like(hm_h2),cmap="hot",aspect="auto",origin="lower")
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values(): s.set_color("#ff6a00")
    ax2.text(0.03,0.97,"GRAD-CAM  EEG",transform=ax2.transAxes,ha="left",va="top",
             color="#ffaa44",fontsize=7.5,fontweight="bold",fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3",facecolor="#1a0800",edgecolor="#ff6a00",linewidth=1.2))
    ax3.set_facecolor(BG2)
    for s in ax3.spines.values(): s.set_visible(False)
    names_=["Control (Healthy)","Parkinson's Disease"]; probs_=result["probs"]
    hi_=1 if "Parkinson" in pred else 0
    pairs_=sorted(zip(probs_,names_),key=lambda x:x[0])
    sp_=[p for p,_ in pairs_]; sn_=[n for _,n in pairs_]; hi_n=names_[hi_]
    bars_=ax3.barh(sn_,[p*100 for p in sp_],
                   color=[pc if n==hi_n else BLUE for n in sn_],edgecolor="none",height=0.52)
    for bar,prob in zip(bars_,sp_):
        ax3.text(min(bar.get_width()+1.2,94),bar.get_y()+bar.get_height()/2,
                 f"{prob*100:.1f}%",va="center",ha="left",color=TW,fontsize=9,fontweight="bold")
    ax3.set_xlim(0,115); ax3.set_xlabel("Confidence (%)",color=TD,fontsize=8)
    ax3.set_title("EEG Classification",color=TW,fontsize=9,pad=5,fontweight="bold")
    ax3.tick_params(colors=TW,labelsize=8); ax3.xaxis.label.set_color(TD)
    fig.patch.set_facecolor(BG); return fig


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI NARRATIVE
# ═══════════════════════════════════════════════════════════════════════════════
_MRI_CTX={"Normal":"No neurological pathology. Brain parenchyma within normal limits.",
"Alzheimer_VeryMild":"Very mild Alzheimer's. Early cortical atrophy, subtle hippocampal volume reduction.",
"Alzheimer_Mild":"Mild Alzheimer's. Progressive cortical atrophy, hippocampal and entorhinal cortex volume loss.",
"Alzheimer_Moderate":"Moderate Alzheimer's. Marked global cortical atrophy, sulcal widening, ventricular enlargement.",
"Tumor_Glioma":"Glioma — primary glial tumour. Infiltrative, ill-defined margins, T2/FLAIR hyperintensity.",
"Tumor_Meningioma":"Meningioma — extra-axial dural-based tumour. Well-circumscribed, homogeneous enhancement.",
"Tumor_Pituitary":"Pituitary adenoma — intrasellar tumour. May cause endocrine dysfunction and visual field defects."}
_MRI_DIFF={"Normal":[],"Alzheimer_VeryMild":["Mild cognitive impairment","Normal age-related atrophy"],
"Alzheimer_Mild":["Vascular dementia","Lewy body dementia","Frontotemporal dementia"],
"Alzheimer_Moderate":["Advanced vascular dementia","Mixed dementia"],
"Tumor_Glioma":["Metastatic disease","Primary CNS lymphoma","Tumefactive demyelination"],
"Tumor_Meningioma":["Hemangiopericytoma","Dural metastasis","Solitary fibrous tumour"],
"Tumor_Pituitary":["Craniopharyngioma","Rathke cleft cyst","Hypothalamic glioma"]}

_DISCLAIMER=("IMPORTANT: This report has been generated by an artificial intelligence system and is intended strictly "
"for research and educational purposes only. It has NOT been reviewed, validated, or approved by a licensed radiologist, "
"neurologist, or any qualified medical professional. The findings must NOT be used as the basis for any clinical diagnosis, "
"treatment decision, or medical management without thorough review by a qualified specialist. AI systems can produce errors "
"and misclassifications. Always consult a certified medical professional before acting on any imaging result.")

def _fig_to_b64(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=80,bbox_inches="tight",facecolor=fig.get_facecolor())
    buf.seek(0); return base64.b64encode(buf.read()).decode()

def _fig_to_bytes(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=120,bbox_inches="tight",facecolor=fig.get_facecolor())
    buf.seek(0); return buf.getvalue()

def call_gemini(prompt, img_b64, key, model="gemini-2.5-flash"):
    import google.generativeai as genai
    genai.configure(api_key=key)
    m=genai.GenerativeModel(model)
    for attempt in range(3):
        try:
            r=m.generate_content(
                contents=[{"inline_data":{"mime_type":"image/png","data":img_b64}},prompt],
                generation_config=genai.types.GenerationConfig(temperature=0.2,top_p=0.90,max_output_tokens=5000))
            return r.text
        except Exception as e:
            if "429" in str(e) and attempt<2: time.sleep(60*(2**attempt))
            else: raise
    return ""

def parse_sections(raw):
    headers=["CLINICAL INDICATION","TECHNIQUE","FINDINGS","IMPRESSION"]
    cleaned=re.sub(r"[*#]+\s*","",raw).replace("\r\n","\n").replace("\r","\n")
    out={}
    for i,h in enumerate(headers):
        pos=cleaned.upper().find(h+":"); 
        if pos==-1: out[h]="[Not generated]"; continue
        start=pos+len(h)+1
        end=cleaned.upper().find(headers[i+1]+":",start) if i+1<len(headers) else -1
        out[h]=(cleaned[start:end] if end!=-1 else cleaned[start:]).strip()
    return out

def build_mri_prompt(result, meta):
    cls=result["final_name"]; ca=result["cam_analysis"]
    s1s=", ".join(f"{STAGE1_NAMES[i]}: {result['s1_probs'][i]:.1%}" for i in range(3))
    s1_group=STAGE1_NAMES[result["s1_pred_idx"]]
    if result["s2_probs"] is not None:
        s2n=result["s2_names"]
        s2s=", ".join(f"{s2n[i].replace('_',' ')}: {result['s2_probs'][i]:.1%}" for i in range(len(s2n)))
        arch="DenseNet121 ordinal regression" if result["route"]=="alzheimer" else "EfficientNet-B0 softmax"
        s2b=f"STAGE 2 ({arch}):\n  Subtype probs: {s2s}\n  Confirmed: {cls.replace('_',' ')} ({result['final_conf']:.1%})"
        two_stage=True
    else:
        s2b="STAGE 2: Not applicable — classified as Normal at Stage 1."; two_stage=False
    attn=(f"GradCAM++ shows {ca['attention']} activation centred on {ca['region']} "
          f"(coverage {ca['coverage']:.1%}, peak {ca['peak']:.3f}, σ={ca['spread']:.3f}).")
    diffs=", ".join(_MRI_DIFF.get(cls,[])) or "None applicable"
    return f"""You are a senior neuroradiology AI report assistant. Use ONLY data provided. Do NOT invent findings.

PATIENT: {meta.get('patient_id','ANON')} | {meta.get('scan_date','')} | MRI Brain
FINAL DIAGNOSIS   : {cls.replace('_',' ')}
REPORT CONFIDENCE : {result['final_conf']:.1%}
STAGE 1 PROBS     : {s1s}  → {s1_group} ({result['s1_probs'][result['s1_pred_idx']]:.1%})
{s2b}
GRADCAM           : {attn}
DISEASE CONTEXT   : {_MRI_CTX.get(cls,'')}
DIFFERENTIALS     : {diffs}

Write EXACTLY these five plain-text sections (no markdown, no bold, no asterisks):
CLINICAL INDICATION: 1 sentence.
TECHNIQUE: 1-2 sentences — AI-assisted MRI, two-stage CNN, GradCAM++.
FINDINGS: 4-6 sentences — diagnosis+confidence, Stage1 result, Stage2 subtype probs, GradCAM region and meaning.
IMPRESSION: 3-4 sentences — restate diagnosis, confidence margin, differentials, specialist review.
"""  .strip()

def build_ct_prompt(result, meta):
    fine=result["fine_class"]; urg=CT_URGENCY.get(fine,"Routine")
    ps="" 
    if result["fine_probs"] is not None:
        ps=", ".join(f"{result['fine_cls_names'][i]}: {result['fine_probs'][i]:.1%}" for i in range(len(result["fine_probs"])))
    return f"""You are a senior neuroradiology AI report assistant. Use ONLY data provided. Do NOT invent findings.

MODALITY      : CT Head
MAIN CATEGORY : {result['main_class']} ({result['main_conf']:.1%})
FINAL DIAGNOSIS: {fine.replace('_',' ')}
CONFIDENCE    : {result['fine_conf']:.1%}
URGENCY       : {urg}
SUBTYPE PROBS : {ps or 'N/A'}
DESCRIPTION   : {CT_DESC.get(fine,'')}

Write EXACTLY these five plain-text sections (no markdown, no bold):
CLINICAL INDICATION: 1 sentence.
TECHNIQUE: 1-2 sentences — AI-assisted CT, two-stage CTClassifier, 3-view TTA, GradCAM++.
FINDINGS: 4-6 sentences — diagnosis+confidence, Stage1 group, Stage2 subtype probs, GradCAM attention, urgency.
IMPRESSION: 3-4 sentences — restate diagnosis+urgency, differentials, specialist review.
"""  .strip()

def build_eeg_prompt(result, meta):
    return f"""You are a clinical neurophysiology AI report assistant. Use ONLY data provided.

MODALITY      : Resting-State EEG
CLASSIFICATION: {result['prediction']}
CONFIDENCE    : {result['confidence']:.1%}
PD probability: {result['probs'][1]:.1%}
HC probability: {result['probs'][0]:.1%}
THRESHOLD     : 70% for Parkinson positive
PIPELINE      : MNE → STFT spectrogram → ResNet18 CNN → GradCAM

Write EXACTLY these five plain-text sections (no markdown, no bold):
CLINICAL INDICATION: 1 sentence.
TECHNIQUE: 1-2 sentences — EEG STFT spectrogram, ResNet18, GradCAM.
FINDINGS: 4-5 sentences — classification+confidence, both class probabilities, GradCAM bands, spectral implications.
IMPRESSION: 3-4 sentences — restate classification, clinical significance, recommend neurologist, note AI limitations.
"""  .strip()


# ═══════════════════════════════════════════════════════════════════════════════
# PDF BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def build_unified_pdf(modality_results, patient_meta, gemini_model_name):
    L=_import_libs()
    A4=L["A4"]; mm=L["mm"]; getSampleStyleSheet=L["getSampleStyleSheet"]
    ParagraphStyle=L["ParagraphStyle"]; HexColor=L["HexColor"]; white=L["white"]
    SimpleDocTemplate=L["SimpleDocTemplate"]; Paragraph=L["Paragraph"]
    Spacer=L["Spacer"]; RLImage=L["RLImage"]; Table=L["Table"]
    TableStyle=L["TableStyle"]; HRFlowable=L["HRFlowable"]; PageBreak=L["PageBreak"]
    TA_CENTER=L["TA_CENTER"]; TA_JUSTIFY=L["TA_JUSTIFY"]

    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,topMargin=18*mm,bottomMargin=18*mm,leftMargin=18*mm,rightMargin=18*mm)
    W=A4[0]-36*mm; S=getSampleStyleSheet()
    mk=lambda n,**kw: ParagraphStyle(n,parent=S["Normal"],**kw)
    st={
        "title":mk("t",fontName="Helvetica-Bold",fontSize=16,textColor=white,alignment=TA_CENTER),
        "sub"  :mk("s",fontName="Helvetica",fontSize=9,textColor=HexColor("#90CAF9"),alignment=TA_CENTER),
        "sec"  :mk("h",fontName="Helvetica-Bold",fontSize=11,textColor=HexColor("#00897B"),spaceBefore=10,spaceAfter=4),
        "mod"  :mk("m",fontName="Helvetica-Bold",fontSize=13,textColor=white,alignment=TA_CENTER),
        "body" :mk("b",fontName="Helvetica",fontSize=10,leading=15,alignment=TA_JUSTIFY,spaceAfter=5),
        "small":mk("sm",fontName="Helvetica",fontSize=8,textColor=HexColor("#455A64"),alignment=TA_CENTER),
        "disc" :mk("d",fontName="Helvetica-Oblique",fontSize=8,textColor=HexColor("#B71C1C"),leading=12,alignment=TA_JUSTIFY),
        "foot" :mk("f",fontName="Helvetica",fontSize=7,textColor=HexColor("#78909C"),alignment=TA_CENTER),
        "ml"   :mk("ml",fontName="Helvetica-Bold",fontSize=9),
        "mv"   :mk("mv",fontName="Helvetica",fontSize=9),
        "badge":mk("bg",fontName="Helvetica-Bold",fontSize=11,textColor=white,alignment=TA_CENTER),
        "disc_h":mk("dh",fontName="Helvetica-Bold",fontSize=9,textColor=HexColor("#C62828"),spaceAfter=3),
    }
    TEAL=HexColor("#00897B"); BORDER=HexColor("#CFD8DC")
    MOD_BG={"MRI":HexColor("#4A148C"),"CT":HexColor("#0D47A1"),"EEG":HexColor("#1B5E20")}

    def banner(txt,sty,bg=HexColor("#0D1B2A"),pad=(8,6)):
        t=Table([[Paragraph(txt,sty)]],colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),
                               ("TOPPADDING",(0,0),(-1,-1),pad[0]),
                               ("BOTTOMPADDING",(0,0),(-1,-1),pad[1])]))
        return t
    def hr(color=TEAL,thick=1): return HRFlowable(width=W,thickness=thick,color=color,spaceAfter=3)

    story=[]
    story.append(banner("UNIFIED NEUROIMAGING AI REPORT",st["title"]))
    story.append(banner("NeuroImaging AI Research Pipeline  ·  Multi-Modal Analysis",st["sub"],pad=(5,5)))
    story.append(Spacer(1,6*mm))
    story+=[Paragraph("ANALYSIS SUMMARY",st["sec"]),hr()]
    rows=[["Modality","Final Diagnosis","Confidence","Status"]]
    for mr in modality_results:
        mod,res=mr["modality"],mr["result"]
        if mod=="MRI": rows.append([mod,res["final_name"].replace("_"," "),f"{res['final_conf']:.1%}","✔"])
        elif mod=="CT": rows.append([mod,res["fine_class"].replace("_"," "),f"{res['fine_conf']:.1%}",CT_URGENCY.get(res["fine_class"],"Routine")])
        else: rows.append([mod,res["prediction"],f"{res['confidence']:.1%}","✔"])
    tbl=Table(rows,colWidths=[W*f for f in [0.12,0.45,0.18,0.25]])
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),TEAL),("TEXTCOLOR",(0,0),(-1,0),white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[HexColor("#FFFFFF"),HexColor("#F4F6F8")]),
        ("BOX",(0,0),(-1,-1),0.5,BORDER),("INNERGRID",(0,0),(-1,-1),0.3,BORDER),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),7)]))
    story+=[tbl,Spacer(1,5*mm)]
    story+=[Paragraph("PATIENT & SCAN INFORMATION",st["sec"]),hr()]
    mw=[W*f for f in [0.20,0.30,0.20,0.30]]
    p=lambda txt,s: Paragraph(txt,s)
    mt=Table([[p("Patient ID",st["ml"]),p(patient_meta.get("patient_id","ANON"),st["mv"]),
               p("Age",st["ml"]),p(patient_meta.get("patient_age","N/A"),st["mv"])],
              [p("Sex",st["ml"]),p(patient_meta.get("patient_sex","N/A"),st["mv"]),
               p("Referring",st["ml"]),p(patient_meta.get("referring_clinician","N/A"),st["mv"])],
              [p("Date",st["ml"]),p(patient_meta.get("scan_date",datetime.date.today().isoformat()),st["mv"]),
               p("Institution",st["ml"]),p(patient_meta.get("institution","NeuroAI"),st["mv"])],
              [p("Indication",st["ml"]),p(patient_meta.get("clinical_indication","AI screening"),st["mv"]),
               p("",st["mv"]),p("",st["mv"])]],colWidths=mw)
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),HexColor("#F4F6F8")),
        ("BOX",(0,0),(-1,-1),0.5,BORDER),("INNERGRID",(0,0),(-1,-1),0.3,BORDER),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(-1,-1),6)]))
    story+=[mt,Spacer(1,4*mm),hr(HexColor("#CFD8DC"),0.5),
            Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  │  LLM: {gemini_model_name}",st["foot"])]

    CAPTIONS={"MRI":("GRAD-CAM EXPLAINABILITY FIGURE","Stage1 GradCAM++ (EfficientNet-B0, MONAI preprocessing)  ·  Yellow crosshair = top-5% activation centroid."),
               "CT":("GRADCAM++ ATTENTION MAP","CTClassifier GradCAM++ (3-view TTA)  ·  Red box = AI focal region."),
               "EEG":("STFT + GRAD-CAM FIGURE","Grad-CAM on STFT spectrogram (ResNet18 layer4)  ·  Highlighted bands drove classification.")}

    for mr in modality_results:
        mod=mr["modality"]; result=mr["result"]; fig=mr["figure"]; sections=mr["sections"]; bg=MOD_BG[mod]
        story.append(PageBreak())
        story.append(banner(f"■■ {mod} ANALYSIS ■■",st["mod"],bg=bg))
        if mod=="MRI":   badge=f"MRI — {result['final_name'].replace('_',' ').upper()}   ·   {result['final_conf']:.1%}"
        elif mod=="CT":  badge=f"CT — {result['fine_class'].replace('_',' ').upper()}   ·   {result['fine_conf']:.1%}   ·   {CT_URGENCY.get(result['fine_class'],'Routine')}"
        else:            badge=f"EEG — {result['prediction'].upper()}   ·   {result['confidence']:.1%}"
        story.append(banner(badge,st["badge"],bg=bg,pad=(5,6)))
        story.append(Spacer(1,5*mm))
        cap_title,cap_text=CAPTIONS[mod]
        story+=[Paragraph(cap_title,st["sec"]),hr()]
        fig_bytes=_fig_to_bytes(fig)
        story+=[RLImage(io.BytesIO(fig_bytes),width=W,height=W*0.38),
                Paragraph(cap_text,st["small"]),Spacer(1,5*mm)]
        story+=[Paragraph("AI-GENERATED REPORT NARRATIVE",st["sec"]),
                HRFlowable(width=W,thickness=1.5,color=TEAL,spaceAfter=5)]
        for sec in ["CLINICAL INDICATION","TECHNIQUE","FINDINGS","IMPRESSION"]:
            story+=[Paragraph(sec,st["sec"]),Paragraph(sections.get(sec,"[Not generated]"),st["body"])]

    # Global disclaimer appears only once at the very end
    story.append(PageBreak())
    story.append(banner("END OF UNIFIED REPORT",st["title"],bg=HexColor("#0D1B2A")))
    story+=[Spacer(1,10*mm),
            HRFlowable(width=W,thickness=1.5,color=HexColor("#C62828"),spaceAfter=5),
            Paragraph("GLOBAL DISCLAIMER",st["disc_h"]),
            Paragraph(_DISCLAIMER,st["disc"]),
            Spacer(1,5*mm),
            HRFlowable(width=W,thickness=0.5,color=HexColor("#CFD8DC"),spaceAfter=2),
            Paragraph(f"NeuroAI Multi-Modal Pipeline  ·  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  LLM: {gemini_model_name}",st["foot"])]

    doc.build(story); buf.seek(0); return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def urgency_badge(urgency):
    cls={"CRITICAL":"badge-critical","URGENT":"badge-urgent","Routine":"badge-routine"}.get(urgency,"badge-ok")
    return f'<span class="{cls}">{urgency}</span>'

def conf_color(conf):
    if conf>=0.85: return "#2ecc71"
    if conf>=0.55: return "#f39c12"
    return "#e74c3c"

def render_report_sections(sections):
    for sec in ["CLINICAL INDICATION","TECHNIQUE","FINDINGS","IMPRESSION"]:
        txt=sections.get(sec,"")
        if txt and txt not in ("[Not generated]",""):
            st.markdown(f'<div class="report-section"><strong>{sec}</strong><br>{txt}</div>',unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_mri, tab_ct, tab_eeg, tab_report = st.tabs(["🧠 MRI Analysis","🩻 CT Analysis","⚡ EEG Analysis","📄 Unified Report"])

# Session state
if "mri_result"   not in st.session_state: st.session_state.mri_result   = None
if "ct_result"    not in st.session_state: st.session_state.ct_result    = None
if "eeg_result"   not in st.session_state: st.session_state.eeg_result   = None
if "mri_fig"      not in st.session_state: st.session_state.mri_fig      = None
if "ct_fig"       not in st.session_state: st.session_state.ct_fig       = None
if "eeg_fig"      not in st.session_state: st.session_state.eeg_fig      = None
if "mri_sections" not in st.session_state: st.session_state.mri_sections = None
if "ct_sections"  not in st.session_state: st.session_state.ct_sections  = None
if "eeg_sections" not in st.session_state: st.session_state.eeg_sections = None


# ── MRI TAB ───────────────────────────────────────────────────────────────────
with tab_mri:
    st.markdown('<div class="modality-banner">🧠 MRI Brain Analysis — EfficientNet-B0 + DenseNet121 / EfficientNet-B0 + GradCAM++</div>',unsafe_allow_html=True)
    col1,col2=st.columns([1,2])
    with col1:
        st.markdown('<p class="section-hdr">Upload MRI Image</p>',unsafe_allow_html=True)
        mri_file=st.file_uploader("MRI Brain Scan",type=["jpg","jpeg","png","bmp","tiff"],key="mri_upload",
                                   label_visibility="collapsed")
        if mri_file:
            st.image(mri_file,width='stretch',caption="Uploaded MRI")
        gen_mri_narr=st.checkbox("Generate AI Narrative (requires Gemini key)",value=True,key="mri_narr")
        run_mri_btn=st.button("▶ Run MRI Analysis",width='stretch',key="run_mri")

    with col2:
        if run_mri_btn and mri_file:
            with tempfile.NamedTemporaryFile(delete=False,suffix=Path(mri_file.name).suffix) as tmp:
                tmp.write(mri_file.read()); tmp_path=tmp.name
            try:
                with st.spinner("Loading models…"):
                    models_data=load_all_models(models_dir)
                    L=_import_libs()
                if not models_data["mri"]:
                    st.error("❌ No MRI models loaded. Check your Models Directory path.")
                else:
                    prog=st.progress(0,"Running Stage 1…")
                    result=run_mri(tmp_path, models_data["mri"], L)
                    prog.progress(50,"Generating figure…")
                    fig=make_mri_figure(result)
                    prog.progress(70,"Generating narrative…" if gen_mri_narr and gemini_key else "Done")
                    sections={"CLINICAL INDICATION":"","TECHNIQUE":"","FINDINGS":"","IMPRESSION":""}
                    if gen_mri_narr and gemini_key:
                        try:
                            prompt=build_mri_prompt(result,patient_meta)
                            raw=call_gemini(prompt,_fig_to_b64(fig),gemini_key,gemini_model)
                            sections=parse_sections(raw)
                        except Exception as e: st.warning(f"Gemini error: {e}")
                    prog.progress(100,"✅ Complete")
                    st.session_state.mri_result=result; st.session_state.mri_fig=fig
                    st.session_state.mri_sections=sections
            except Exception as e:
                st.error(f"MRI inference error: {traceback.format_exc()}")
            finally:
                os.unlink(tmp_path)
        elif run_mri_btn:
            st.warning("Please upload an MRI image first.")

        if st.session_state.mri_result:
            r=st.session_state.mri_result
            cls=r["final_name"].replace("_"," "); conf=r["final_conf"]
            route_col={"normal":"#00d4aa","alzheimer":"#ff7f32","tumor":"#c77dff"}.get(r["route"],"#c77dff")
            st.markdown(f"""<div class="diag-card" style="border-color:{route_col};">
                <h2 style="color:{route_col};">{cls.upper()}</h2>
                <div class="conf">Confidence: <strong style="color:{conf_color(conf)};">{conf:.1%}</strong></div>
                <div style="margin-top:0.5rem;font-size:0.85rem;color:#8899bb;">
                Stage 1: {STAGE1_NAMES[r['s1_pred_idx']]} ({r['s1_probs'][r['s1_pred_idx']]:.1%}) &nbsp;·&nbsp; Route: {r['route'].title()}</div>
            </div>""",unsafe_allow_html=True)
            if r["s2_probs"] is not None:
                st.markdown('<p class="section-hdr">Subtype Probabilities</p>',unsafe_allow_html=True)
                cols=st.columns(len(r["s2_names"]))
                for i,(name,prob) in enumerate(zip(r["s2_names"],r["s2_probs"])):
                    with cols[i]:
                        st.markdown(f'<div class="metric-chip"><div class="label">{name.replace("_"," ")}</div><div class="value">{prob:.1%}</div></div>',unsafe_allow_html=True)
            ca=r["cam_analysis"]
            st.markdown(f"""<div class="neuro-card" style="margin-top:0.8rem;">
                <strong style="color:#90CAF9;">GradCAM++ Analysis</strong><br>
                <span style="font-size:0.88rem;color:#c8d0e0;">
                Region: {ca['region']} &nbsp;·&nbsp; Attention: {ca['attention'].upper()} &nbsp;·&nbsp;
                Coverage: {ca['coverage']:.1%} &nbsp;·&nbsp; Peak: {ca['peak']:.3f} &nbsp;·&nbsp; σ: {ca['spread']:.3f}
                </span></div>""",unsafe_allow_html=True)
            if st.session_state.mri_fig:
                st.pyplot(st.session_state.mri_fig, width='stretch')
            if st.session_state.mri_sections:
                with st.expander("📋 AI-Generated Report Narrative",expanded=True):
                    render_report_sections(st.session_state.mri_sections)
        else:
            st.markdown("""<div class="neuro-card" style="text-align:center;padding:3rem;">
                <div style="font-size:3rem;">🧠</div>
                <div style="color:#556080;margin-top:1rem;">Upload an MRI image and click Run Analysis</div>
            </div>""",unsafe_allow_html=True)


# ── CT TAB ────────────────────────────────────────────────────────────────────
with tab_ct:
    st.markdown('<div class="modality-banner">🩻 CT Head Analysis — EfficientNet-B4 + 3-View TTA + GradCAM++</div>',unsafe_allow_html=True)
    col1,col2=st.columns([1,2])
    with col1:
        st.markdown('<p class="section-hdr">Upload CT Scan</p>',unsafe_allow_html=True)
        ct_file=st.file_uploader("CT Head Scan",type=["jpg","jpeg","png","bmp","tiff","dcm","dicom"],
                                  key="ct_upload",label_visibility="collapsed")
        if ct_file:
            if not ct_file.name.lower().endswith((".dcm",".dicom")):
                st.image(ct_file,width='stretch',caption="Uploaded CT")
            else: st.info("DICOM file uploaded")
        gen_ct_narr=st.checkbox("Generate AI Narrative",value=True,key="ct_narr")
        run_ct_btn=st.button("▶ Run CT Analysis",width='stretch',key="run_ct")

    with col2:
        if run_ct_btn and ct_file:
            with tempfile.NamedTemporaryFile(delete=False,suffix=Path(ct_file.name).suffix) as tmp:
                tmp.write(ct_file.read()); tmp_path=tmp.name
            try:
                with st.spinner("Loading models…"):
                    models_data=load_all_models(models_dir); L=_import_libs()
                if not models_data["ct"]:
                    st.error("❌ No CT models loaded. Check your Models Directory path.")
                else:
                    prog=st.progress(0,"Running inference…")
                    result=run_ct(tmp_path,models_data["ct"],L)
                    prog.progress(60,"Generating figure…"); fig=make_ct_figure(result)
                    prog.progress(80,"Generating narrative…" if gen_ct_narr and gemini_key else "Done")
                    sections={"CLINICAL INDICATION":"","TECHNIQUE":"","FINDINGS":"","IMPRESSION":""}
                    if gen_ct_narr and gemini_key:
                        try:
                            raw=call_gemini(build_ct_prompt(result,patient_meta),_fig_to_b64(fig),gemini_key,gemini_model)
                            sections=parse_sections(raw)
                        except Exception as e: st.warning(f"Gemini error: {e}")
                    prog.progress(100,"✅ Complete")
                    st.session_state.ct_result=result; st.session_state.ct_fig=fig
                    st.session_state.ct_sections=sections
            except Exception as e: st.error(f"CT error: {traceback.format_exc()}")
            finally: os.unlink(tmp_path)
        elif run_ct_btn: st.warning("Please upload a CT scan first.")

        if st.session_state.ct_result:
            r=st.session_state.ct_result
            fine=r["fine_class"].replace("_"," "); conf=r["fine_conf"]
            urgency=CT_URGENCY.get(r["fine_class"],"Routine")
            urg_col={"CRITICAL":"#e74c3c","URGENT":"#f39c12","Routine":"#2ecc71"}[urgency]
            st.markdown(f"""<div class="diag-card" style="border-color:{urg_col};">
                <h2 style="color:{urg_col};">{fine.upper()}</h2>
                <div class="conf">Confidence: <strong style="color:{conf_color(conf)};">{conf:.1%}</strong>
                &nbsp;&nbsp;{urgency_badge(urgency)}</div>
                <div style="margin-top:0.5rem;font-size:0.85rem;color:#8899bb;">
                Main Category: {r['main_class']} ({r['main_conf']:.1%})</div>
            </div>""",unsafe_allow_html=True)
            if r["fine_probs"] is not None:
                st.markdown('<p class="section-hdr">Subtype Probabilities</p>',unsafe_allow_html=True)
                cols=st.columns(len(r["fine_cls_names"]))
                for i,(name,prob) in enumerate(zip(r["fine_cls_names"],r["fine_probs"])):
                    with cols[i]:
                        st.markdown(f'<div class="metric-chip"><div class="label">{name.replace("_"," ")}</div><div class="value">{prob:.1%}</div></div>',unsafe_allow_html=True)
            st.markdown(f"""<div class="neuro-card" style="margin-top:0.8rem;">
                <strong style="color:#90CAF9;">Clinical Description</strong><br>
                <span style="font-size:0.88rem;color:#c8d0e0;">{CT_DESC.get(r['fine_class'],'')}</span>
            </div>""",unsafe_allow_html=True)
            if st.session_state.ct_fig: st.pyplot(st.session_state.ct_fig,width='stretch')
            if st.session_state.ct_sections:
                with st.expander("📋 AI-Generated Report Narrative",expanded=True):
                    render_report_sections(st.session_state.ct_sections)
        else:
            st.markdown("""<div class="neuro-card" style="text-align:center;padding:3rem;">
                <div style="font-size:3rem;">🩻</div>
                <div style="color:#556080;margin-top:1rem;">Upload a CT scan and click Run Analysis</div>
            </div>""",unsafe_allow_html=True)


# ── EEG TAB ───────────────────────────────────────────────────────────────────
with tab_eeg:
    st.markdown('<div class="modality-banner">⚡ EEG Parkinson\'s Analysis — ResNet18 + STFT Spectrogram + GradCAM</div>',unsafe_allow_html=True)
    col1,col2=st.columns([1,2])
    with col1:
        st.markdown('<p class="section-hdr">Upload EEG Files</p>',unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.8rem;color:#6a88aa;margin-bottom:0.5rem;padding:0.5rem 0.8rem;
            background:#0d1428;border-radius:6px;border-left:3px solid #2a5080;">
            ⚠️ <strong>EEGLAB format</strong> requires <strong>both</strong> the
            <code>.set</code> header file <em>and</em> the <code>.fdt</code> data file.
            Upload them together below.
        </div>""",unsafe_allow_html=True)

        # Upload .set file
        set_file=st.file_uploader("Header file (.set)",type=["set"],
                                   key="eeg_set_upload",label_visibility="visible")
        # Upload .fdt file — Streamlit doesn't natively support .fdt so accept all via no type filter
        fdt_file=st.file_uploader("Data file (.fdt) — required for EEGLAB format",
                                   type=None, key="eeg_fdt_upload",label_visibility="visible",
                                   help="Select the matching .fdt binary data file")

        # Status indicators
        if set_file:
            st.success(f"✅ {set_file.name}  ({set_file.size/1024:.1f} KB)")
        if fdt_file:
            if fdt_file.name.endswith(".fdt"):
                st.success(f"✅ {fdt_file.name}  ({fdt_file.size/1024:.1f} KB)")
            else:
                st.warning(f"⚠️ Expected a .fdt file, got: {fdt_file.name}")

        # Pair validation
        if set_file and fdt_file:
            set_stem = Path(set_file.name).stem
            fdt_stem = Path(fdt_file.name).stem
            if set_stem != fdt_stem:
                st.warning(f"⚠️ Filename mismatch: `{set_file.name}` and `{fdt_file.name}` — ensure both belong to the same recording.")

        gen_eeg_narr=st.checkbox("Generate AI Narrative",value=True,key="eeg_narr")
        run_eeg_btn=st.button("▶ Run EEG Analysis",width='stretch',key="run_eeg")

    with col2:
        if run_eeg_btn:
            if not set_file:
                st.warning("Please upload the .set file.")
            elif not fdt_file:
                st.warning("Please upload the matching .fdt data file.")
            else:
                # Save BOTH files into the same temp directory so MNE can find them
                import tempfile as _tf
                tmp_dir = _tf.mkdtemp()
                set_path = os.path.join(tmp_dir, set_file.name)
                fdt_path = os.path.join(tmp_dir, fdt_file.name)
                with open(set_path, "wb") as f: f.write(set_file.read())
                with open(fdt_path, "wb") as f: f.write(fdt_file.read())
                try:
                    with st.spinner("Loading models…"):
                        models_data=load_all_models(models_dir); L=_import_libs()
                    if models_data["eeg"] is None:
                        st.error("❌ EEG model not loaded. Check your Models Directory path.")
                    else:
                        prog=st.progress(0,"Processing EEG signal…")
                        result=run_eeg(set_path,models_data["eeg"],L)
                        prog.progress(60,"Generating figure…"); fig=make_eeg_figure(result)
                        prog.progress(80,"Generating narrative…" if gen_eeg_narr and gemini_key else "Done")
                        sections={"CLINICAL INDICATION":"","TECHNIQUE":"","FINDINGS":"","IMPRESSION":""}
                        if gen_eeg_narr and gemini_key:
                            try:
                                raw=call_gemini(build_eeg_prompt(result,patient_meta),_fig_to_b64(fig),gemini_key,gemini_model)
                                sections=parse_sections(raw)
                            except Exception as e: st.warning(f"Gemini error: {e}")
                        prog.progress(100,"✅ Complete")
                        st.session_state.eeg_result=result; st.session_state.eeg_fig=fig
                        st.session_state.eeg_sections=sections
                except Exception as e: st.error(f"EEG error: {traceback.format_exc()}")
                finally:
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        if st.session_state.eeg_result:
            r=st.session_state.eeg_result
            pred=r["prediction"]; conf=r["confidence"]
            pc="#e74c3c" if "Parkinson" in pred else "#00d4aa"
            st.markdown(f"""<div class="diag-card" style="border-color:{pc};">
                <h2 style="color:{pc};">{pred.upper()}</h2>
                <div class="conf">Confidence: <strong style="color:{conf_color(conf)};">{conf:.1%}</strong></div>
                <div style="margin-top:0.5rem;font-size:0.85rem;color:#8899bb;">
                PD: {r['probs'][1]:.1%} &nbsp;·&nbsp; HC: {r['probs'][0]:.1%} &nbsp;·&nbsp; Threshold: 70%</div>
            </div>""",unsafe_allow_html=True)
            st.markdown('<p class="section-hdr">Class Probabilities</p>',unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: st.markdown(f'<div class="metric-chip"><div class="label">Control (Healthy)</div><div class="value" style="color:#00d4aa;">{r["probs"][0]:.1%}</div></div>',unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-chip"><div class="label">Parkinson\'s Disease</div><div class="value" style="color:#e74c3c;">{r["probs"][1]:.1%}</div></div>',unsafe_allow_html=True)
            if st.session_state.eeg_fig: st.pyplot(st.session_state.eeg_fig,width='stretch')
            if st.session_state.eeg_sections:
                with st.expander("📋 AI-Generated Report Narrative",expanded=True):
                    render_report_sections(st.session_state.eeg_sections)
        else:
            st.markdown("""<div class="neuro-card" style="text-align:center;padding:3rem;">
                <div style="font-size:3rem;">⚡</div>
                <div style="color:#556080;margin-top:1rem;">Upload an EEG file and click Run Analysis</div>
            </div>""",unsafe_allow_html=True)


# ── UNIFIED REPORT TAB ────────────────────────────────────────────────────────
with tab_report:
    st.markdown('<div class="modality-banner">📄 Unified Multi-Modal Report Generator</div>',unsafe_allow_html=True)
    available=[m for m,r in [("MRI",st.session_state.mri_result),
                               ("CT",st.session_state.ct_result),
                               ("EEG",st.session_state.eeg_result)] if r is not None]
    if not available:
        st.markdown("""<div class="neuro-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;">📄</div>
            <div style="color:#556080;margin-top:1rem;">
            Run at least one modality analysis first, then generate the unified PDF report.
            </div></div>""",unsafe_allow_html=True)
    else:
        st.success(f"✅ Ready to compile: {', '.join(available)} analyses")
        st.markdown('<p class="section-hdr">Analysis Summary</p>',unsafe_allow_html=True)
        cols=st.columns(len(available))
        for i,mod in enumerate(available):
            with cols[i]:
                if mod=="MRI":
                    r=st.session_state.mri_result
                    st.markdown(f"""<div class="neuro-card" style="border-color:#c77dff;">
                        <div style="font-size:0.8rem;color:#8899bb;margin-bottom:0.3rem;">🧠 MRI</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#c77dff;">{r['final_name'].replace('_',' ')}</div>
                        <div style="color:#2ecc71;font-size:0.9rem;">{r['final_conf']:.1%}</div>
                    </div>""",unsafe_allow_html=True)
                elif mod=="CT":
                    r=st.session_state.ct_result
                    urg=CT_URGENCY.get(r["fine_class"],"Routine")
                    urg_col={"CRITICAL":"#e74c3c","URGENT":"#f39c12","Routine":"#2ecc71"}[urg]
                    st.markdown(f"""<div class="neuro-card" style="border-color:{urg_col};">
                        <div style="font-size:0.8rem;color:#8899bb;margin-bottom:0.3rem;">🩻 CT</div>
                        <div style="font-size:1.1rem;font-weight:700;color:{urg_col};">{r['fine_class'].replace('_',' ')}</div>
                        <div style="color:#aaa;font-size:0.9rem;">{r['fine_conf']:.1%} · {urg}</div>
                    </div>""",unsafe_allow_html=True)
                elif mod=="EEG":
                    r=st.session_state.eeg_result
                    pc="#e74c3c" if "Parkinson" in r["prediction"] else "#00d4aa"
                    st.markdown(f"""<div class="neuro-card" style="border-color:{pc};">
                        <div style="font-size:0.8rem;color:#8899bb;margin-bottom:0.3rem;">⚡ EEG</div>
                        <div style="font-size:1.1rem;font-weight:700;color:{pc};">{r['prediction']}</div>
                        <div style="color:#2ecc71;font-size:0.9rem;">{r['confidence']:.1%}</div>
                    </div>""",unsafe_allow_html=True)

        st.divider()
        gen_pdf_btn=st.button("🖨️ Generate Unified PDF Report",width='stretch',key="gen_pdf")
        if gen_pdf_btn:
            with st.spinner("Building PDF…"):
                modality_results=[]
                if st.session_state.mri_result:
                    modality_results.append({"modality":"MRI","result":st.session_state.mri_result,
                        "figure":st.session_state.mri_fig,
                        "sections":st.session_state.mri_sections or {}})
                if st.session_state.ct_result:
                    modality_results.append({"modality":"CT","result":st.session_state.ct_result,
                        "figure":st.session_state.ct_fig,
                        "sections":st.session_state.ct_sections or {}})
                if st.session_state.eeg_result:
                    modality_results.append({"modality":"EEG","result":st.session_state.eeg_result,
                        "figure":st.session_state.eeg_fig,
                        "sections":st.session_state.eeg_sections or {}})
                try:
                    pdf_bytes=build_unified_pdf(modality_results,patient_meta,gemini_model)
                    pid=patient_meta.get("patient_id","ANON").replace(" ","_")
                    mods="_".join([mr["modality"] for mr in modality_results])
                    fname=f"NeuroAI_{pid}_{mods}_{datetime.date.today().isoformat()}.pdf"
                    st.success("✅ PDF generated successfully!")
                    st.download_button(label="⬇️ Download Unified Report PDF",data=pdf_bytes,
                                       file_name=fname,mime="application/pdf",width='stretch')
                except Exception as e:
                    st.error(f"PDF generation error: {traceback.format_exc()}")

# ── Model status expander ─────────────────────────────────────────────────────
with st.expander("🔧 System Status & Model Logs"):
    c1,c2=st.columns(2)
    with c1:
        try:
            import torch
            dev=str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        except: dev="CPU"
        st.markdown(f"""
        <div class="metric-chip" style="display:inline-block;margin:0.3rem;">
            <div class="label">Device</div><div class="value">{dev.upper()}</div>
        </div>
        <div class="metric-chip" style="display:inline-block;margin:0.3rem;">
            <div class="label">MRI Ready</div>
            <div class="value">{"✅" if st.session_state.mri_result else "⬜"}</div>
        </div>
        <div class="metric-chip" style="display:inline-block;margin:0.3rem;">
            <div class="label">CT Ready</div>
            <div class="value">{"✅" if st.session_state.ct_result else "⬜"}</div>
        </div>
        <div class="metric-chip" style="display:inline-block;margin:0.3rem;">
            <div class="label">EEG Ready</div>
            <div class="value">{"✅" if st.session_state.eeg_result else "⬜"}</div>
        </div>
        """,unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Models Dir:** `{models_dir}`")
        st.markdown(f"**Gemini:** {'✅ Key set' if gemini_key else '❌ Not set'} · `{gemini_model}`")
    if st.button("📂 Load & Check Models",key="check_models"):
        with st.spinner("Checking…"):
            try:
                data=load_all_models(models_dir)
                for log in data["logs"]: st.text(log)
                st.info(f"Device: {data['device']} | MRI models: {list(data['mri'].keys())} | CT models: {list(data['ct'].keys())} | EEG: {'✅' if data['eeg'] else '❌'}")
            except Exception as e: st.error(str(e))