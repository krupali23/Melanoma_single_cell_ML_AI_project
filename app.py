# app.py — Immunotherapy Response Explorer (single file)
import os, json, joblib, base64, warnings, re, io, datetime, glob
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)

# ================== BASIC CONFIG ==================
st.set_page_config(page_title="Immunotherapy Response Explorer", layout="wide")

# ---- constants / paths
REPO_ROOT = os.path.dirname(__file__)
DATA_DIR = (
    os.environ.get("DATA_DIR")
    or st.secrets.get("DATA_DIR", os.path.join(REPO_ROOT, "data"))
)

# prefer data/models if it exists, else top-level models
models_in_data = os.path.join(DATA_DIR, "models")
fallback_models = os.path.join(REPO_ROOT, "models")
MODELS_DIR_DEFAULT = models_in_data if os.path.isdir(models_in_data) else fallback_models
MODELS_DIR = (
    os.environ.get("MODELS_DIR")
    or st.secrets.get("MODELS_DIR", MODELS_DIR_DEFAULT)
)

SC_ANNOT = os.path.join(DATA_DIR, "sc_annot.csv")        # UMAP + metadata
SC_EXPR_PARQUET = os.path.join(DATA_DIR, "sc_expr.parquet")
SC_EXPR_CSV     = os.path.join(DATA_DIR, "sc_expr.csv")

# Marker lists
MARKERS_DIR         = os.path.join(DATA_DIR, "markers")
MARKERS_TOP50_DIR   = os.path.join(MARKERS_DIR, "per_group_top50")  # << your folder

# ================== HERO + TOP NAV ==================
st.markdown(
    """
    <style>
    /* Make the top radio look like a bold tab bar */
    div[data-testid="stRadio"] > label p {font-size: 0; margin: 0;}
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        font-size:1.05rem; font-weight:800; padding:8px 12px; margin:4px 8px 20px 0;
        border-radius:10px; border:1px solid #d9e2ec; background:#fafcfe; cursor:pointer;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {background:#f3f7ff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='margin-bottom:0.2rem'>🧬 Immunotherapy Response Explorer</h1>"
    "<p style='font-size:1.05rem; color:#2f4f4f; margin-top:0'>"
    "Predict responders, visualize single cells, and explore marker genes — all in one place."
    "</p>",
    unsafe_allow_html=True,
)

nav_labels = ["Performance","Cell Map","Gene Explorer","Comparison","Explainability","Chat","Background"]
page = st.radio("Navigation", nav_labels, index=0, horizontal=True, label_visibility="collapsed")

# ================== LANGUAGE (top of sidebar) + i18n for upload text ==================
LANGS = {
    "en":{"name":"English",
          "upload_header":"Score custom CSV",
          "upload_caption":"CSV indexed by patient_id with same feature columns. Add `true_label` (R/NR) to see metrics.",
          "upload_btn":"Upload features CSV"},
    "de":{"name":"Deutsch",
          "upload_header":"Eigenes CSV auswerten",
          "upload_caption":"CSV mit patient_id als Index und identischen Merkmalen. `true_label` (R/NR) hinzufügen für Metriken.",
          "upload_btn":"CSV mit Merkmalen hochladen"},
    "hi":{"name":"हिन्दी",
          "upload_header":"कस्टम CSV स्कोर करें",
          "upload_caption":"patient_id इंडेक्स वाला CSV, वही फीचर कॉलम. मेट्रिक्स के लिए `true_label` (R/NR) जोड़ें।",
          "upload_btn":"फ़ीचर CSV अपलोड करें"},
    "es":{"name":"Español",
          "upload_header":"Evaluar CSV propio",
          "upload_caption":"CSV indexado por patient_id con las mismas columnas. Añade `true_label` (R/NR) para ver métricas.",
          "upload_btn":"Subir CSV de características"}
}
def L(key, lang): return LANGS.get(lang, LANGS["en"]).get(key, LANGS["en"][key])

st.sidebar.header("Language")
lang_code = st.sidebar.selectbox("Language", ["en","de","hi","es"], index=0,
                                 format_func=lambda c: LANGS[c]["name"])

# ================== UPLOAD (immediately under Language) ==================
st.sidebar.header(L("upload_header", lang_code))
st.sidebar.caption(L("upload_caption", lang_code))
upload = st.sidebar.file_uploader(L("upload_btn", lang_code), type=["csv"])
X_custom = None
if upload is not None:
    try:
        X_custom = pd.read_csv(upload, index_col=0)
        st.sidebar.success(f"Uploaded shape: {X_custom.shape}")
    except Exception as e:
        st.sidebar.error("Failed to read uploaded CSV.")
        st.sidebar.exception(e)

# ================== HELPERS ==================
def listdir_safe(p):
    try: return sorted(os.listdir(p))
    except Exception as e: return [f"ERR: {e}"]

@st.cache_data(show_spinner=False)
def load_features():
    p=os.path.join(DATA_DIR,"patient_features.csv")
    if not os.path.exists(p): raise FileNotFoundError(f"Missing features file: {p}")
    return pd.read_csv(p, index_col=0).fillna(0.0)

@st.cache_data(show_spinner=False)
def load_labels():
    for name in ["patient_response_binary.csv","patient_response_cleaned_with_mixed.csv",
                 "patient_response_binary.xlsx","patient_response_cleaned_with_mixed.xlsx"]:
        p=os.path.join(DATA_DIR,name)
        if not os.path.exists(p): continue
        try:
            df = pd.read_excel(p) if p.lower().endswith(".xlsx") else pd.read_csv(p)
            cols=[c.strip().lower() for c in df.columns]; df.columns=cols
            pid=next((c for c in cols if ("patient" in c) or ("sample" in c)), None)
            resp=next((c for c in cols if ("response" in c) or ("label" in c) or ("responder" in c)), None)
            if pid is None or resp is None: continue
            out = df[[pid,resp]].rename(columns={pid: "patient_id", resp: "response"})
            if out["response"].dtype==object:
                m={"responder":1,"r":1,"cr":1,"pr":1,"non-responder":0,"nr":0,"sd":0,"pd":0}
                out["response"]=out["response"].astype(str).str.strip().str.lower().map(m)
            out=out.dropna(subset=["patient_id","response"]).copy()
            # safe strip for patient_id
            out["patient_id"]=out["patient_id"].astype(str).str.strip()
            out["response"]=out["response"].astype(int)
            return out.groupby("patient_id")["response"].max().to_frame()
        except Exception:
            continue
    return None

def parse_model_bundle(obj):
    if isinstance(obj,dict):
        model=obj.get("model",obj); feat=obj.get("feature_names",None); thr=obj.get("final_threshold",obj.get("threshold",0.5))
    else:
        model,feat,thr=obj,None,0.5
    try: thr=float(thr)
    except Exception: thr=0.5
    return model,feat,thr

@st.cache_resource(show_spinner=False)
def load_model(path):
    obj=joblib.load(path); return parse_model_bundle(obj)

def probas(est, X):
    if hasattr(est,"predict_proba"): return est.predict_proba(X)[:,1]
    if hasattr(est,"decision_function"): z=est.decision_function(X); return 1/(1+np.exp(-z))
    p=est.predict(X).astype(float); return np.clip(p,0.0,1.0)

def align(X, feat_names): return X if feat_names is None else X.reindex(columns=feat_names).fillna(0.0)

def feature_importance_series(est, feat_names):
    if hasattr(est,"feature_importances_"):
        return pd.Series(est.feature_importances_, index=feat_names).sort_values(ascending=False)
    coef=getattr(est,"coef_",None)
    if coef is None and hasattr(est,"named_steps"):
        for step in est.named_steps.values():
            if hasattr(step,"coef_"): coef=step.coef_; break
    if coef is not None:
        return pd.Series(np.abs(coef).ravel(), index=feat_names).sort_values(ascending=False)
    return None

def correlation_importance(model, X, feat_names):
    try:
        s=probas(model,X[feat_names]);
        return pd.Series({f:(np.corrcoef(X[f].values,s)[0,1] if X[f].std()>0 else 0) for f in feat_names}).abs().sort_values(ascending=False)
    except Exception:
        return None

# ---- sc_annot loader (UMAP + metadata)
@st.cache_data(show_spinner=False)
def load_sc_annot(path_csv: str):
    if not os.path.exists(path_csv): return None
    df = pd.read_csv(path_csv)
    candidates = [c for c in df.columns if "umap" in c.lower()] or [c for c in df.columns if c.lower() in ["x","y","dim1","dim2"]]
    if len(candidates) >= 2:
        xcol, ycol = candidates[0], candidates[1]
    else:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        xcol, ycol = (nums[0], nums[1]) if len(nums) >=2 else (df.columns[0], df.columns[1])
    df = df.rename(columns={xcol:"umap1", ycol:"umap2"})
    if "cell_id" not in df.columns: df["cell_id"] = df.index.astype(str)
    if "celltype" in df.columns and "cell_type" not in df.columns:
        df = df.rename(columns={"celltype":"cell_type"})
    label_col = None
    for k in ["cell_type","cluster","clusters","leiden","louvain","cluster_name","annot"]:
        if k in df.columns:
            label_col = k; break
    if label_col is None: df["cluster_label"] = "Cluster_" + df.index.astype(str)
    else: df["cluster_label"] = df[label_col].astype(str)
    for k in ["patient_id","patient","sample","donor","case","subject","pt"]:
        if k in df.columns:
            df["patient_id"] = df[k].astype(str); break
    if "patient_id" not in df.columns: df["patient_id"] = "unknown"
    keep = ["cell_id","umap1","umap2","cluster_label","patient_id"]
    more = [c for c in df.columns if c not in keep]
    return df[keep + more]

sc_df = load_sc_annot(SC_ANNOT)

# ---- expression helper
@st.cache_data(show_spinner=False)
def available_gene_list():
    if os.path.exists(SC_EXPR_PARQUET):
        try:
            import pyarrow.parquet as pq
            return [c for c in pq.ParquetFile(SC_EXPR_PARQUET).schema.names if c!="cell_id"]
        except Exception:
            try:
                return [c for c in pd.read_parquet(SC_EXPR_PARQUET, engine="pyarrow").columns if c!="cell_id"]
            except Exception:
                pass
    if os.path.exists(SC_EXPR_CSV):
        with open(SC_EXPR_CSV,"r",encoding="utf-8") as f:
            header=f.readline().strip().split(",")
        return [h for h in header if h!="cell_id"]
    return []

ALL_GENES = available_gene_list()
ALL_GENES_LUT = {g.lower(): g for g in ALL_GENES}  # case-insensitive mapper

def map_to_available_genes(markers: list[str]) -> list[str]:
    mapped = []
    for g in markers:
        key = str(g).strip().lower()
        if key in ALL_GENES_LUT:
            mapped.append(ALL_GENES_LUT[key])
    return mapped

def read_gene_cols(genes):
    genes = map_to_available_genes(genes)
    if not genes: return None
    if os.path.exists(SC_EXPR_PARQUET):
        try: return pd.read_parquet(SC_EXPR_PARQUET, columns=["cell_id"]+genes)
        except Exception: pass
    if os.path.exists(SC_EXPR_CSV):
        try: return pd.read_csv(SC_EXPR_CSV, usecols=["cell_id"]+genes)
        except Exception: pass
    return None

# ---- marker files loader (top-50 per cluster)
@st.cache_data(show_spinner=False)
def load_top50_markers(markers_dir: str):
    cluster2genes, union = {}, set()
    if not os.path.isdir(markers_dir):
        return cluster2genes, []
    files = glob.glob(os.path.join(markers_dir, "*.xlsx")) + glob.glob(os.path.join(markers_dir, "*.csv"))
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        cluster = re.sub(r'[_-]*top50$', '', re.sub(r'^celltype[_-]*', '', base)).strip()
        try:
            df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        gcol = next((c for c in df.columns if str(c).lower() in
                    ["gene","genes","symbol","gene_symbol","marker","features","feature"]), df.columns[0])
        genes_raw = (df[gcol].dropna().astype(str).str.strip().tolist())[:50]
        cluster2genes[cluster] = genes_raw
        union.update(genes_raw)
    return cluster2genes, sorted(union)

CLUSTER2MARKERS, ALL_MARKER_GENES_RAW = load_top50_markers(MARKERS_TOP50_DIR)

# ================== LOAD PATIENT FEATURES / LABELS / MODEL ==================
try:
    X_all = load_features()
    st.success(f"Loaded features: {X_all.shape[0]} patients × {X_all.shape[1]} features")
except Exception as e:
    st.error("Failed to load features file.")
    st.exception(e)
    st.stop()

labels_df = load_labels()
if labels_df is not None:
    matched = labels_df.index.intersection(X_all.index)
    labels_df = labels_df.loc[matched].astype(int)
    st.caption(f"Labels matched to features: {len(labels_df)}")

available_models=[m for m in listdir_safe(MODELS_DIR) if isinstance(m,str) and m.endswith(".joblib")]
if not available_models:
    st.error("No .joblib models found in MODELS_DIR.")
    st.stop()
model_file=st.sidebar.selectbox("Model (.joblib)", available_models, index=0)
MODEL_FILE=os.path.join(MODELS_DIR, model_file)

try:
    model, feat_names, default_thr = load_model(MODEL_FILE)
    if feat_names is None: feat_names=list(X_all.columns)
    st.sidebar.success(f"Model loaded: {os.path.basename(MODEL_FILE)}")
except Exception as e:
    st.sidebar.error("Failed to load model.")
    st.exception(e)
    st.stop()

thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(default_thr), 0.01)

st.sidebar.header("Cohort")
sel_patients = st.sidebar.multiselect("Select patients (optional)", list(X_all.index), default=list(X_all.index))

# ================== SHARED SMALL UTILS ==================
def get_predictions_dataframe():
    X_use = X_custom if X_custom is not None else X_all.loc[sel_patients]
    X_use = align(X_use, feat_names)
    scores = probas(model, X_use)
    preds  = (scores >= thr).astype(int)
    out = pd.DataFrame({"probability":scores,"prediction":preds}, index=X_use.index)
    out = out.sort_values("probability", ascending=False)
    return out

def attach_response(df: pd.DataFrame, labels_df: pd.DataFrame | None) -> pd.DataFrame:
    if labels_df is None:
        if "response" not in df.columns: df["response"] = np.nan
        return df
    if "patient_id" not in df.columns:
        df["response"] = np.nan
        return df
    m = labels_df.reset_index().rename(columns={"index":"patient_id"})
    return df.merge(m, on="patient_id", how="left")

# ================== KNOWLEDGE-BASE ANSWERS (NEW) ==================
def kb_answer(question: str, page: str | None = None) -> str:
    q = (question or "").strip().lower()

    # metrics / glossary
    if any(k in q for k in ["what are features", "what is feature", "features", "input features"]):
        return ("**Features** are the numeric inputs to the model (per patient). "
                "Examples: proportions of cell types, marker scores, clinical covariates.")
    if any(k in q for k in ["f1", "f1-score", "f1 score"]):
        return ("**F1-score** = harmonic mean of precision and recall (2·P·R/(P+R)). "
                "Balances false positives and false negatives.")
    if any(k in q for k in ["sensitivity", "recall", "tpr", "true positive rate"]):
        return ("**Sensitivity (Recall, TPR)** = TP/(TP+FN). Of the true responders, how many did we catch?")
    if any(k in q for k in ["specificity", "tnr", "true negative rate"]):
        return ("**Specificity (TNR)** = TN/(TN+FP). Of the true non-responders, how many did we correctly reject?")
    if any(k in q for k in ["precision", "ppv"]):
        return ("**Precision (PPV)** = TP/(TP+FP). Of the predicted responders, how many were actually responders?")
    if "confusion" in q and "matrix" in q:
        return ("A **confusion matrix** is a 2×2 table of TN, FP, FN, TP (true rows, predicted columns). "
                "It shows error types at a glance.")
    if "roc" in q:
        return ("**ROC curve** plots TPR vs FPR across thresholds; **AUC** measures discrimination (1 best, 0.5 random).")
    if "precision-recall" in q or "pr curve" in q or "pr-auc" in q:
        return ("**Precision–Recall curve** shows precision vs recall; **PR-AUC** summarizes performance on imbalanced data.")
    if any(k in q for k in ["what is a cluster", "what is cluster", "cluster", "cell type"]):
        return ("A **cluster/cell type** groups similar cells by expression; labels are assigned via annotation.")
    if any(k in q for k in ["responder status", "responders", "non-responders", "label"]):
        return ("**Responder status** is the patient outcome (1=Responder, 0=Non-responder).")

    # page-specific summaries
    if page == "Performance":
        if any(k in q for k in ["what is this page", "overview", "help"]):
            return ("**Performance** reports Accuracy, Sensitivity, Specificity, ROC-AUC, PR-AUC, and the confusion matrix. "
                    "If you upload labeled data, it evaluates on that too.")
        if "threshold" in q:
            return ("The **decision threshold** converts probabilities to 0/1. Higher threshold → fewer predicted responders "
                    "but higher precision (and lower recall).")
    if page == "Cell Map":
        if any(k in q for k in ["what is this page", "overview", "help"]):
            return ("**Cell Map** is a UMAP of all cells. Color by cluster, responder status, or a gene to see where signals live.")
    if page == "Gene Explorer":
        if "group distribution" in q or "violin" in q or "box" in q:
            return ("**Group distribution (Violin/Box)** compares a gene’s expression between Responders and Non-responders.")
        if "expression by cell type" in q or "by cluster" in q or "means" in q:
            return ("**Expression by cell type/cluster** shows per-cluster violins and a mean-by-cluster table/bar.")
        if "heatmap" in q and ("group" in q or "mean" in q):
            return ("**Heatmap — group mean expression** aggregates cells by response group for selected marker genes.")
        if "umap colored" in q:
            return ("**UMAP colored by gene** paints the UMAP by that gene’s expression.")
        if "co-expression" in q or "coexpression" in q:
            return ("**Co-expression scatter** plots Gene-1 vs Gene-2 per cell; a diagonal suggests joint expression.")
        if any(k in q for k in ["what is this page", "overview", "help"]):
            return ("**Gene Explorer** loads a cluster’s top-50 markers and lets you visualize violins/means/heatmaps/UMAP/co-expression.")
    if page == "Comparison":
        if any(k in q for k in ["what is this page", "overview", "help", "patient vs group"]):
            return ("**Patient vs Group Averages** shows one patient’s features vs responder and non-responder means. "
                    "Bars show Patient−Mean differences; the traffic light is the model prediction.")
    if page == "Explainability":
        if any(k in q for k in ["what is this page", "overview", "help", "which features matter"]):
            return ("**Explainability** ranks features by importance (SHAP if available; otherwise model/correlation importances).")
    if page == "Background":
        if any(k in q for k in ["what is this page", "overview", "help"]):
            return ("**Background** explains the app goal, why single-cell matters, and how to read each page.")

    # overall (Chat page)
    if page in [None, "Chat"]:
        if any(k in q for k in ["what is this app", "overall", "all pages", "how to use", "guide"]):
            return ("Overall guide:\n"
                    "• **Performance**: how good the model is.\n"
                    "• **Cell Map**: UMAP—color by cluster/label/gene.\n"
                    "• **Gene Explorer**: marker panels and multiple plots.\n"
                    "• **Comparison**: one patient vs group means.\n"
                    "• **Explainability**: which features mattered.\n"
                    "• **Background**: overview & tips.")
    return ("Ask about metrics (F1, sensitivity, specificity, ROC/PR), clusters, responder status, gene expression, "
            "or what each section on this page means.")

# ----- tiny chat helper shown at bottom of every page (UPDATED) -----
def render_page_chat(page_name: str):
    st.markdown("---")
    with st.expander("💬 Help on this page"):
        key = f"mini_chat_{page_name}"
        if key not in st.session_state:
            st.session_state[key] = []
        q = st.text_input("Ask about this page or your data", key=f"input_{key}")
        if st.button("Ask", key=f"btn_{key}"):
            st.session_state[key].append(("You", q or ""))
            ans = kb_answer(q, page=page_name)
            st.session_state[key].append(("App", ans))
        for who, msg in st.session_state[key]:
            st.markdown(f"**{who}:** {msg}")

# ================== PAGE: PERFORMANCE ==================
if page == "Performance":
    st.header("📊 Performance")

    out = get_predictions_dataframe()

    # --- EVALUATION FIRST (internal labels or uploaded)
    if labels_df is not None and X_custom is None and not out.empty:
        st.subheader("Evaluation on labeled patients")
        common = out.index.intersection(labels_df.index)
        if len(common) > 0:
            y_pred = out.loc[common,"prediction"].astype(int).values
            y_prob = out.loc[common,"probability"].values
            y_true = labels_df.loc[common,"response"].astype(int).values

            tn,fp,fn,tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
            acc  = (tp+tn)/(tp+tn+fp+fn)
            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            rocA = roc_auc_score(y_true,y_prob)
            prA  = average_precision_score(y_true,y_prob)

            c1,c2,c3 = st.columns(3)
            c1.metric("Accuracy",     f"{acc:.3f}")
            c2.metric("Sensitivity",  f"{sens:.3f}")
            c3.metric("Specificity",  f"{spec:.3f}")
            st.write(f"ROC-AUC: **{rocA:.3f}** · PR-AUC: **{prA:.3f}**")

            with st.expander("Show ROC, PR and Confusion Matrix"):
                fpr,tpr,_ = roc_curve(y_true,y_prob)
                prec,rec,_= precision_recall_curve(y_true,y_prob)
                figROC = go.Figure(); figROC.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                figROC.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="chance"))
                figROC.update_layout(title=f"ROC (AUC={auc(fpr,tpr):.3f})", xaxis_title="FPR", yaxis_title="TPR", height=400)
                st.plotly_chart(figROC, width="stretch")

                figPR  = go.Figure(); figPR.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
                figPR.update_layout(title="Precision–Recall", xaxis_title="Recall", yaxis_title="Precision", height=400)
                st.plotly_chart(figPR,  width="stretch")

                cm2 = confusion_matrix(y_true, y_pred, labels=[0,1])
                figCM = px.imshow(cm2, text_auto=True, color_continuous_scale="Blues",
                                  labels=dict(x="Predicted", y="True", color="Count"),
                                  x=["0 (NR)","1 (R)"], y=["0 (NR)","1 (R)"], title="Confusion Matrix")
                st.plotly_chart(figCM, width="stretch")

            with st.expander("Classification report (text)"):
                st.text(classification_report(y_true, y_pred, target_names=["0 (NR)", "1 (R)"], digits=3))
        else:
            st.info("No overlap between selected patients and label table.")

    if X_custom is not None:
        st.subheader("Evaluation on uploaded patients")
        tmp = align(X_custom, feat_names)
        scores = probas(model, tmp)
        preds  = (scores >= thr).astype(int)
        user_df = X_custom.copy()
        if "true_label" in user_df.columns:
            y_true = user_df["true_label"].map({"NR":0,"R":1,"0":0,"1":1}).astype(float).dropna().astype(int)
            valid_ids = y_true.index.intersection(tmp.index)
            if len(valid_ids) > 0:
                y = y_true.loc[valid_ids].values
                p = preds[np.isin(tmp.index, valid_ids)]
                s = scores[np.isin(tmp.index, valid_ids)]

                tn,fp,fn,tp = confusion_matrix(y, p, labels=[0,1]).ravel()
                acc  = (tp+tn)/(tp+tn+fp+fn)
                sens = tp/(tp+fn) if (tp+fn) else 0.0
                spec = tn/(tn+fp) if (tn+fp) else 0.0
                rocA = roc_auc_score(y, s)
                prA  = average_precision_score(y, s)

                c1,c2,c3 = st.columns(3)
                c1.metric("Accuracy",     f"{acc:.3f}")
                c2.metric("Sensitivity",  f"{sens:.3f}")
                c3.metric("Specificity",  f"{spec:.3f}")
                st.write(f"ROC-AUC: **{rocA:.3f}** · PR-AUC: **{prA:.3f}**")

                with st.expander("Show Confusion Matrix (uploaded)"):
                    figCMu = px.imshow(confusion_matrix(y, p, labels=[0,1]), text_auto=True, color_continuous_scale="Greens",
                                    labels=dict(x="Predicted", y="True", color="Count"),
                                    x=["0 (NR)","1 (R)"], y=["0 (NR)","1 (R)"], title="Confusion Matrix (uploaded)")
                    st.plotly_chart(figCMu, width="stretch")
            else:
                st.info("Uploaded file had no matching indices for labels.")
        else:
            st.info("Upload a file with a `true_label` column (R/NR) to see performance metrics.")

    # ---- Predictions table AFTER metrics
    st.subheader("Predictions table")
    out_all = get_predictions_dataframe()
    tbl = out_all.copy()
    tbl.insert(0, "traffic", tbl["prediction"].map(lambda p:"🟢" if p==1 else "🔴"))
    tbl.insert(0, "patient_id", tbl.index)
    st.dataframe(
        tbl[["patient_id","traffic","probability","prediction"]]
            .style.format({"probability":"{:.3f}"}),
        width="stretch"
    )
    st.download_button("Download predictions (CSV)", data=out_all.to_csv().encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")

    render_page_chat("Performance")

# ================== PAGE: CELL MAP ==================
elif page == "Cell Map":
    st.header("🗺️ Cell Map")
    st.caption("Each dot is a single cell; closer dots mean more similar biology.")
    if sc_df is None:
        st.info("sc_annot.csv not found in DATA_DIR.")
    else:
        df_plot = sc_df.copy()
        df_plot = attach_response(df_plot, labels_df)

        mode = st.radio("Color by", ["Cell type / Cluster","Responder status","Gene expression"], horizontal=True)
        if mode == "Cell type / Cluster":
            fig = px.scatter(df_plot, x="umap1", y="umap2", color="cluster_label", render_mode="webgl", height=650)
        elif mode == "Responder status":
            lab = df_plot["response"].map({1:"Responders",0:"Non-responders"}).fillna("Unknown")
            fig = px.scatter(df_plot, x="umap1", y="umap2", color=lab,
                             color_discrete_map={"Responders":"#2ecc71","Non-responders":"#e74c3c","Unknown":"#95a5a6"},
                             render_mode="webgl", height=650)
        else:
            if not ALL_GENES:
                st.warning("Gene expression matrix not available (sc_expr.* missing).")
                fig = px.scatter(df_plot, x="umap1", y="umap2", color="cluster_label", render_mode="webgl", height=650)
            else:
                gene = st.selectbox("Gene", ALL_GENES, index=0)
                expr = read_gene_cols([gene])
                if expr is not None:
                    df_plot = df_plot.merge(expr, on="cell_id", how="left")
                    fig = px.scatter(df_plot, x="umap1", y="umap2", color=gene, color_continuous_scale="RdBu_r",
                                     render_mode="webgl", height=650)
                else:
                    st.warning("Could not read expression for this gene.")
                    fig = px.scatter(df_plot, x="umap1", y="umap2", color="cluster_label", render_mode="webgl", height=650)

        try:
            cent = df_plot.groupby("cluster_label")[["umap1","umap2"]].median().reset_index()
            fig.add_trace(go.Scatter(x=cent["umap1"], y=cent["umap2"], mode="text",
                                     text=cent["cluster_label"], textposition="middle center",
                                     textfont=dict(size=12,color="black"), showlegend=False, hoverinfo="skip"))
        except Exception:
            pass
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    render_page_chat("Cell Map")

# ================== PAGE: GENE EXPLORER ==================
elif page == "Gene Explorer":
    st.title("🧪 Gene Explorer — expression by group and cluster")

    if not ALL_GENES:
        st.info("Single-cell expression matrix not found (sc_expr.parquet).")
        st.stop()

    with st.expander("🔎 Pick a cluster & its markers", expanded=True):
        clusters_for_picker = sorted(list(CLUSTER2MARKERS.keys()))
        if not clusters_for_picker:
            st.warning("No marker files found under /Data/markers (CSV/XLSX).")
            st.stop()

        cluster_pick = st.selectbox("Cluster", clusters_for_picker, index=0)
        cluster_genes_raw = CLUSTER2MARKERS.get(cluster_pick, [])
        cluster_genes_present = [g for g in cluster_genes_raw if g in ALL_GENES]

        st.caption(
            f"Found **{len(cluster_genes_raw)}** markers in files · "
            f"**{len(cluster_genes_present)}** present in expression matrix"
        )

        use_all = st.checkbox("Use all markers from this cluster", value=True)
        if use_all:
            panel_genes = cluster_genes_present
        else:
            panel_genes = st.multiselect(
                "Panel genes (for heatmap/UMAP coloring)",
                options=cluster_genes_present if cluster_genes_present else ALL_GENES,
                default=cluster_genes_present[:10] if cluster_genes_present else []
            )

        gene_options = cluster_genes_present if cluster_genes_present else ALL_GENES
        g1 = st.selectbox("Primary gene", options=gene_options, index=0)
        g2 = st.selectbox("Second gene (for co-expression)", options=["(none)"] + gene_options, index=0)

    need = []
    if g1 in ALL_GENES: need.append(g1)
    if g2 != "(none)" and g2 in ALL_GENES: need.append(g2)
    need += [g for g in panel_genes if g in ALL_GENES]
    need = list(dict.fromkeys(need))
    if not need:
        st.warning("Selected genes are not present in the matrix.")
        st.stop()

    expr = read_gene_cols(need)
    if expr is None or expr.empty:
        st.warning("Could not read expression; check sc_expr.parquet.")
        st.stop()

    df = sc_df.merge(expr, on="cell_id", how="left")
    df = attach_response(df, labels_df)

    label_map = {1:"Responders", 0:"Non-responders"}
    lab = df["response"].map(label_map).fillna("Unknown")
    green, red = "#2ecc71", "#e74c3c"
    colmap = {"Responders":green, "Non-responders":red, "Unknown":"#95a5a6"}

    with st.expander("🎯 Group distribution: Responders vs Non-responders (Violin / Box)", expanded=False):
        tabV, tabB = st.tabs(["Violin","Box"])
        with tabV:
            figv = px.violin(df, x=lab, y=g1, color=lab, box=True, points="all", height=440, color_discrete_map=colmap)
            figv.update_layout(xaxis_title="", yaxis_title=g1, legend_title_text="")
            st.plotly_chart(figv, width="stretch")
        with tabB:
            figb = px.box(df, x=lab, y=g1, color=lab, points="all", height=440, color_discrete_map=colmap)
            figb.update_layout(xaxis_title="", yaxis_title=g1, legend_title_text="")
            st.plotly_chart(figb, width="stretch")

    with st.expander("🧩 Expression by cell type / cluster (Violin + Means)", expanded=False):
        cluster_col = "cluster_label"
        figct = px.violin(df, x=cluster_col, y=g1, color=cluster_col, box=True, height=520)
        figct.update_layout(xaxis_title="", yaxis_title=g1, legend_title_text="")
        st.plotly_chart(figct, width="stretch")

        means = df.groupby(cluster_col)[g1].mean().sort_values(ascending=False)
        st.dataframe(means.to_frame(f"{g1} mean"), use_container_width=True)
        figbar = px.bar(means, title=f"{g1} mean by cluster", height=420)
        st.plotly_chart(figbar, width="stretch")

    with st.expander("🔥 Heatmap — group mean expression for selected markers", expanded=False):
        if panel_genes:
            exprH = read_gene_cols(panel_genes)
            if exprH is not None and not exprH.empty:
                dfH = sc_df.merge(exprH, on="cell_id", how="left")
                dfH = attach_response(dfH, labels_df)
                grp_lab = dfH["response"].map(label_map).fillna("Unknown")
                mat = dfH.groupby(grp_lab)[panel_genes].mean().sort_index()
                figH = px.imshow(mat, color_continuous_scale="RdBu_r", aspect="auto", height=420)
                figH.update_layout(xaxis_title="Gene", yaxis_title="Group", legend_title_text="")
                st.plotly_chart(figH, width="stretch")
            else:
                st.info("Selected heatmap genes were not found in the expression matrix.")
        else:
            st.caption("Pick genes above to render a heatmap.")

    with st.expander("🗺️ UMAP colored by gene", expanded=False):
        gene_for_umap = st.selectbox(
            "Gene to color UMAP",
            options=[g1] + [g for g in panel_genes if g != g1],
            index=0
        )
        exprU = read_gene_cols([gene_for_umap])
        if exprU is not None:
            dfU = sc_df.merge(exprU, on="cell_id", how="left")
            figU = px.scatter(dfU, x="umap1", y="umap2", color=gene_for_umap,
                              color_continuous_scale="RdBu_r", render_mode="webgl", height=650)
            figU.update_layout(legend_title_text="")
            st.plotly_chart(figU, width="stretch")
        else:
            st.info("Could not read expression for this gene.")

    with st.expander("🔗 Co-expression (scatter)", expanded=False):
        if g2 != "(none)" and (g2 in df.columns):
            fig2 = px.scatter(df, x=g1, y=g2, color=lab, trendline="ols", height=450,
                              color_discrete_map=colmap)
            fig2.update_layout(xaxis_title=g1, yaxis_title=g2, legend_title_text="")
            st.plotly_chart(fig2, width="stretch")
        else:
            st.caption("Pick a second gene above to enable co-expression.")

    render_page_chat("Gene Explorer")

# ================== PAGE: COMPARISON ==================
elif page == "Comparison":
    st.header("🆚 Patient vs Group Averages — traffic light")
    out = get_predictions_dataframe()
    if out.empty:
        st.info("Run predictions first.")
    else:
        psel = st.selectbox("Choose a patient", options=out.index.tolist(), index=0)
        row  = out.loc[psel]
        traffic = "🟢" if int(row["prediction"])==1 else "🔴"
        st.metric("Traffic Light (prediction)", f"{traffic}  {int(row['prediction'])}", delta=f"{row['probability']:.3f} prob")

        feat_common=X_all.columns.intersection(feat_names)
        vx=X_all.loc[[psel], feat_common]
        if labels_df is not None:
            dfj=X_all.join(labels_df, how="inner")
            mu_R=dfj[dfj["response"]==1][feat_common].mean()
            mu_N=dfj[dfj["response"]==0][feat_common].mean()
            comp=pd.DataFrame({"patient":vx.squeeze(),"mean_R":mu_R,"mean_N":mu_N})
            comp["delta_R"]=comp["patient"]-comp["mean_R"]; comp["delta_N"]=comp["patient"]-comp["mean_N"]
            k=st.slider("Show top N differing features",5,min(30,comp.shape[0]),10)
            top=comp.reindex(comp["delta_R"].abs().sort_values(ascending=False).head(k).index)
            figC=go.Figure()
            figC.add_trace(go.Bar(name="Patient − Mean(R)", x=top.index, y=top["delta_R"]))
            figC.add_trace(go.Bar(name="Patient − Mean(NR)", x=top.index, y=top["delta_N"]))
            figC.update_layout(barmode="group", xaxis_title="Feature", yaxis_title="Difference", height=450)
            st.plotly_chart(figC, width="stretch")
            with st.expander("Raw table"):
                st.dataframe(top, width="stretch")
        else:
            st.info("Labels not available; group averages require response labels.")

    render_page_chat("Comparison")

# ================== PAGE: EXPLAINABILITY ==================
elif page == "Explainability":
    st.header("🧠 Explainability — which features matter")
    fi=feature_importance_series(model, feat_names)
    try:
        import shap; shap_available=True
    except Exception:
        shap_available=False

    if shap_available:
        try:
            explainer=shap.Explainer(model, X_all[feat_names])
            sample=X_all[feat_names].sample(min(500,X_all.shape[0]), random_state=42)
            sv=explainer(sample)
            shap_df=pd.DataFrame(np.abs(sv.values).mean(axis=0), index=feat_names, columns=["mean|SHAP|"]).sort_values("mean|SHAP|", ascending=False)
            topk=st.slider("Top K",5,min(30,len(shap_df)),10)
            figImp=px.bar(shap_df.head(topk), height=420)
            st.plotly_chart(figImp, width="stretch")
        except Exception:
            fi = fi or correlation_importance(model, X_all, feat_names)
            if fi is not None:
                topk=st.slider("Top features",5,min(30,len(fi)),10)
                figImp=px.bar(fi.head(topk), height=420)
                st.plotly_chart(figImp, width="stretch")
            else:
                st.info("No importances available.")
    else:
        if fi is None: fi=correlation_importance(model, X_all, feat_names)
        if fi is not None:
            topk=st.slider("Top features",5,min(30,len(fi)),10)
            figImp=px.bar(fi.head(topk), height=420)
            st.plotly_chart(figImp, width="stretch")
        else:
            st.info("No importances available.")

    render_page_chat("Explainability")

# ================== PAGE: CHAT (UPDATED) ==================
elif page == "Chat":
    st.markdown("<h2 style='display:flex;align-items:center;gap:8px'>💬 <span>How can I help?</span></h2>", unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    q = st.text_input("Ask anything about the app, metrics, clusters, genes, or pages")
    if st.button("Ask"):
        st.session_state["chat"].append(("You", q))
        ql = (q or "").strip().lower()
        ans = None

        def has(*kw): return any(k in ql for k in kw)

        # keep your functional intents
        if has("cluster names","cell types","list clusters"):
            if sc_df is None:
                ans = "Single-cell table not loaded."
            else:
                names = sc_df["cluster_label"].astype(str).unique().tolist()
                st.dataframe(pd.Series(names, name="clusters"), width="stretch")
                ans = f"I listed {len(names)} cluster names above."
        elif "average expression of gene" in ql:
            m = re.search(r"average expression of gene ([a-z0-9\-_\.]+)", ql)
            if m and ALL_GENES:
                g = next((x for x in ALL_GENES if x.lower()==m.group(1).lower()), None)
                if g:
                    expr = read_gene_cols([g])
                    if expr is not None and sc_df is not None:
                        df = sc_df.merge(expr, on="cell_id", how="left")
                        df = attach_response(df, labels_df)
                        grp = df.groupby(df["response"].map({1:"Responders",0:"Non-responders"}).fillna("Unknown"))[g].mean()
                        st.bar_chart(grp, width="stretch")
                        ans = f"Plotted average {g} by response group."
                else:
                    ans = "That gene is not in the expression matrix."
            else:
                ans = "Gene matrix not available."
        elif has("top features","most important"):
            fi = feature_importance_series(model, feat_names) or correlation_importance(model, X_all, feat_names)
            if fi is not None:
                top = fi.head(12)
                st.dataframe(top.to_frame("importance"), width="stretch")
                st.plotly_chart(px.bar(top.sort_values(), orientation="h"), width="stretch")
                ans = "Top features shown above."
            else:
                ans = "No importances available."
        elif has("how many cells","total cells","number of cells"):
            ans = f"There are **{0 if sc_df is None else sc_df.shape[0]:,}** cells."
        elif has("how many patients","number of patients"):
            ans = f"There are **{X_all.shape[0]}** patients with features."

        # fallback to knowledge base
        if ans is None:
            ans = kb_answer(q, page="Chat")

        st.session_state["chat"].append(("App", ans))

    with st.expander("Common questions"):
        cols = st.columns(3)
        examples = [
            "What are features?",
            "What is F1 score?",
            "What is sensitivity?",
            "What is specificity?",
            "What is a confusion matrix?",
            "What is a cluster?",
            "What does the Cell Map show?",
            "What is co-expression?",
            "What is Patient vs Group Averages?",
            "Which features matter on Explainability?",
            "What is this app (overall)?",
        ]
        for i, e in enumerate(examples):
            if cols[i % 3].button(e, key=f"qa_{i}"):
                st.session_state["chat"].append(("You", e))
                st.session_state["chat"].append(("App", kb_answer(e, page="Chat")))
                st.rerun()

    for who, msg in st.session_state["chat"]:
        st.markdown(f"**{who}:** {msg}")

# ================== PAGE: BACKGROUND ==================
elif page == "Background":
    st.header("📘 Background")
    st.markdown("""
### What is this app?
This app uses single-cell data and patient-level immune features to predict who may respond to immunotherapy.
Every dot (cell) carries a genetic fingerprint, and by analyzing thousands of cells per patient, we can identify immune patterns linked to treatment success or failure.

### Why single-cell analysis matters
- 🧬 **High resolution:** Instead of averaging signals, we see each cell type (T cells, B cells, NK cells, etc.) separately.  
- 🔍 **Hidden patterns:** Responders often have specific immune cells (like CD8 T cells or memory-like T cells) that non-responders lack.  
- 🎯 **Personalized treatment:** Knowing patterns in advance can reduce trial-and-error, side effects, and patient suffering.  

### Why it matters for patients
Not everyone responds to immunotherapy. Trial-and-error treatment causes time loss, side effects, and weakens immunity.
By knowing cellular patterns ahead of time, clinicians can reduce suffering and move toward personalized treatments.

### How to use the pages
- **📊 Performance** — accuracy, sensitivity, specificity, ROC, PR, confusion matrix.  
- **🗺️ Cell Map** — UMAP of cells colored by cluster, responder status, or gene expression.  
- **🧪 Gene Explorer** — violins, box plots, heatmap, and UMAP coloring; browse top-50 markers per cluster.  
- **⚖️ Comparison** — traffic-light prediction and how a patient differs from group averages.  
- **🧠 Explainability** — SHAP/feature importance to see which features drove predictions.  
- **💬 Chat** — quick Q&A about clusters, genes, and features.
""")
    render_page_chat("Background")

# ================== REPORT EXPORT (appears on all pages) ==================
st.markdown("---")
st.subheader("📄 Download quick report")
def build_report_html():
    out = get_predictions_dataframe()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts=[f"<h1>Immunotherapy Response — Report</h1><p>Generated: {ts}</p>"]

    try:
        common = out.index.intersection(labels_df.index) if labels_df is not None else []
        if len(common)>0:
            y_pred = out.loc[common,"prediction"].astype(int).values
            y_prob = out.loc[common,"probability"].values
            y_true = labels_df.loc[common,"response"].astype(int).values
            tn,fp,fn,tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
            acc  = (tp+tn)/(tp+tn+fp+fn)
            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            rocA = roc_auc_score(y_true,y_prob)
            prA  = average_precision_score(y_true,y_prob)
            parts.append(f"<h2>Internal evaluation</h2><ul>"
                         f"<li>Accuracy: {acc:.3f}</li>"
                         f"<li>Sensitivity: {sens:.3f}</li>"
                         f"<li>Specificity: {spec:.3f}</li>"
                         f"<li>ROC-AUC: {rocA:.3f}</li>"
                         f"<li>PR-AUC: {prA:.3f}</li></ul>")
    except Exception:
        pass

    try:
        parts.append("<h2>Predictions (top 50)</h2>")
        parts.append(get_predictions_dataframe().head(50).to_html())
    except Exception:
        pass

    return "".join(parts).encode("utf-8")

st.download_button("Download HTML report", data=build_report_html(),
                   file_name="immunotherapy_report.html", mime="text/html")

# ================== SIDEBAR (bottom): Appearance + Clear cache ==================
def set_background(kind="gradient", color="#dff3c4", color2="#b7e39c", image_bytes=None):
    if kind=="image" and image_bytes is not None:
        b64=base64.b64encode(image_bytes).decode()
        css=f'<style>.stApp{{background:url("data:image/png;base64,{b64}") no-repeat center center fixed;background-size:cover;}}</style>'
    elif kind=="solid":
        css=f"<style>.stApp{{background:{color} !important;}}</style>"
    else:
        css=f"<style>.stApp{{background:linear-gradient(135deg,{color} 0%,{color2} 100%) !important;}}</style>"
    st.markdown(css, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("Appearance")
bg_mode = st.sidebar.selectbox("Background", ["gradient","solid","image"], index=0)
if bg_mode=="solid":
    set_background("solid", color=st.sidebar.color_picker("Background color", "#f7fbff"))
elif bg_mode=="gradient":
    c1=st.sidebar.color_picker("Gradient start", "#eef7ff")
    c2=st.sidebar.color_picker("Gradient end",   "#dbeeff")
    set_background("gradient", c1, c2)
else:
    up_bg=st.sidebar.file_uploader("Upload background image", type=["png","jpg","jpeg"], key="bg_upload")
    set_background("image", image_bytes=up_bg.read()) if up_bg else set_background("gradient")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear app cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
