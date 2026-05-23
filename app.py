# MinTexto v5 — Interfaz Streamlit 

import re, string, pathlib, io
from collections import Counter
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="MinTexto v5", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, div, p, span, h1, h2, h3, h4, h5, h6, button, input, label, textarea, select {
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] svg,
[data-testid="stFileUploaderDropzone"] svg,
[data-baseweb="radio"] svg,
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-icons {
    display: none !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f2e 0%, #141432 60%, #181845 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131330 0%, #181840 100%);
    border-right: 1px solid #2a2a55;
}
[data-testid="collapsedControl"] {
    color: #a5b4fc !important;
    background: rgba(129,140,248,0.15) !important;
    border: 1px solid rgba(129,140,248,0.3) !important;
}
[data-testid="collapsedControl"] svg {
    fill: #a5b4fc !important;
    stroke: #a5b4fc !important;
}

.hero-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 3.6rem; font-weight: 400;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; letter-spacing: 1px; margin-bottom: 4px;
}
.hero-sub {
    text-align: center; color: #6366a0; font-size: 0.95rem;
    font-weight: 400; margin-bottom: 1.5rem; letter-spacing: 0.5px;
}

.sec-head {
    font-size: 0.72rem; font-weight: 600; color: #818cf8;
    text-transform: uppercase; letter-spacing: 2.5px;
    border-left: 3px solid #818cf8; padding-left: 10px;
    margin: 2rem 0 0.8rem 0;
}

.mcard {
    flex: 1; background: linear-gradient(135deg,#13132e,#17173a);
    border: 1px solid #252550; border-radius: 16px;
    padding: 1.1rem 1.2rem; text-align: center;
    box-shadow: 0 4px 24px rgba(129,140,248,0.06);
}
.mcard-label { font-size: 0.65rem; color: #6366a0; text-transform: uppercase; letter-spacing: 1.8px; font-weight: 600; }
.mcard-value {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem; font-weight: 400;
    background: linear-gradient(90deg, #818cf8, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.token-wrap { display: flex; flex-wrap: wrap; gap: 6px; padding: 12px 0; }
.tok {
    display: inline-block; padding: 4px 12px; border-radius: 6px;
    font-size: 0.78rem; font-weight: 500;
    background: rgba(129,140,248,0.10); border: 1px solid rgba(129,140,248,0.2);
    color: #a5b4fc; letter-spacing: 0.2px;
}

.stButton>button {
    background: linear-gradient(90deg,#818cf8,#38bdf8) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.92rem !important;
    padding: 0.55rem 1.4rem !important; width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: opacity .2s, transform .1s;
}
.stButton>button:hover { opacity: .85; transform: scale(1.01); }

.stSlider label { color: #8888b8 !important; font-size: 0.82rem !important; }
hr { border-color: #1e1e42; }
[data-testid="stExpanderHeader"] { color: #a5b4fc !important; font-weight: 600 !important; }
[data-testid="stExpanderHeader"] svg { color: #a5b4fc !important; stroke: #a5b4fc !important; display: block !important; }
</style>
""", unsafe_allow_html=True)

_sw_ruta = pathlib.Path("stopwords-es.txt")
if _sw_ruta.exists():
    STOP_WORDS = {l.strip().lower() for l in _sw_ruta.read_text(encoding="utf-8").splitlines() if l.strip()}
else:
    STOP_WORDS = {"a","al","de","del","el","en","es","la","las","lo","los","no","o","para","por","que","se","si","su","un","una","y"}

def leer_bytes(data: bytes, ext: str) -> str:
    if ext == ".txt":
        return data.decode("utf-8", errors="replace")
    elif ext == ".pdf":
        import pdfplumber
        pags = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: pags.append(t)
        return "\n".join(pags)
    elif ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""

def preprocesar(texto: str) -> list:
    texto = texto.lower()
    texto = re.sub(f"[{re.escape(string.punctuation)}]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return [p for p in texto.split() if p not in STOP_WORDS and len(p) > 1]

def construir_vocabulario(tokens):
    vocab = {}
    for p in tokens:
        if p not in vocab: vocab[p] = len(vocab)
    return vocab, {i: p for p, i in vocab.items()}

def generar_parejas(sec, ventana):
    pares, n = [], len(sec)
    for pos, obj in enumerate(sec):
        for v in range(max(0, pos - ventana), min(n, pos + ventana + 1)):
            if v != pos: pares.append((obj, sec[v]))
    return pares

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def entrenar_con_progreso(vocab, parejas, dim, tasa, epocas, semilla):
    rng = np.random.default_rng(semilla)
    V, E = len(vocab), dim
    Pe = (rng.random((V, E)) - 0.5) / E
    Ps = (rng.random((E, V)) - 0.5) / E
    hist = []
    barra = st.progress(0, text="Entrenando…")
    c1, c2 = st.columns(2)
    ph_ep, ph_loss = c1.empty(), c2.empty()
    intervalo = max(1, epocas // 60)
    for ep in range(epocas):
        loss = 0.0
        for io_, ic in parejas:
            h = Pe[io_]; logits = Ps.T @ h; probs = softmax(logits)
            loss += -np.log(probs[ic] + 1e-9)
            err = probs.copy(); err[ic] -= 1.0
            Ps -= tasa * np.outer(h, err)
            Pe[io_] -= tasa * (Ps @ err)
        lp = loss / len(parejas); hist.append(lp)
        if (ep + 1) % intervalo == 0 or ep == epocas - 1:
            pct = (ep + 1) / epocas
            barra.progress(pct, text=f"Entrenando… época {ep+1}/{epocas}")
            ph_ep.metric("Época", f"{ep+1}/{epocas}")
            ph_loss.metric("Pérdida promedio", f"{lp:.4f}")
    barra.progress(1.0, text="Entrenamiento completado")
    return Pe, Ps, hist

def top_sim(pal, vocab, vocab_inv, Pe, n):
    if pal not in vocab: return []
    v      = Pe[vocab[pal]]
    norms  = np.linalg.norm(Pe, axis=1)
    norm_v = np.linalg.norm(v)
    sims   = np.where(norms > 0, (Pe @ v) / (norms * norm_v + 1e-10), 0.0)
    sims[vocab[pal]] = -1
    top_idx = np.argsort(sims)[::-1][:n]
    return [(float(sims[i]), vocab_inv[i]) for i in top_idx]

def top_ctx(pal, vocab, vocab_inv, Pe, Ps, n):
    if pal not in vocab: return []
    probs = softmax(Ps.T @ Pe[vocab[pal]])
    c = [(float(probs[i]), vocab_inv[i]) for i in range(len(vocab_inv)) if i != vocab[pal]]
    c.sort(reverse=True); return c[:n]

PALETTE = ["#818cf8","#a78bfa","#38bdf8","#34d399","#fb923c","#f472b6","#facc15"]

def fig_perdida(hist):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=hist, mode="lines",
        line=dict(color="#818cf8", width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(129,140,248,0.07)",
    ))
    fig.update_layout(
        title=dict(text="Curva de Aprendizaje — Pérdida por época",
                   font=dict(color="#a5b4fc", size=13, family="DM Sans")),
        xaxis=dict(title="Época", color="#555580", gridcolor="#141432", zeroline=False),
        yaxis=dict(title="Pérdida", color="#555580", gridcolor="#141432"),
        plot_bgcolor="#080814", paper_bgcolor="#080814",
        font=dict(color="#d0d0f0", family="DM Sans"),
        margin=dict(t=44, b=28, l=44, r=16), height=240,
    )
    return fig

def fig_barras(datos, titulo, es_pct=False):
    pals  = [d[1] for d in datos][::-1]
    vals  = [d[0] for d in datos][::-1]
    texts = [f"{v:.2%}" if es_pct else f"{v:.4f}" for v in vals]
    cols  = (PALETTE * 4)[:len(pals)][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=pals, orientation="h",
        marker=dict(color=cols, line=dict(width=0), opacity=0.88),
        text=texts, textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="white", size=11, family="DM Sans"),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(color="#a5b4fc", size=12, family="DM Sans")),
        xaxis=dict(color="#555580", gridcolor="#141432", zeroline=False,
                   tickformat=".2%" if es_pct else ".2f"),
        yaxis=dict(color="#c4c4e4", tickfont=dict(size=11, family="DM Sans")),
        plot_bgcolor="#080814", paper_bgcolor="#080814",
        font=dict(color="#d0d0f0", family="DM Sans"),
        margin=dict(t=40, b=24, l=10, r=20), height=260,
        showlegend=False,
    )
    return fig

with st.sidebar:
    st.markdown("## Configuración")
    st.markdown("---")
    st.markdown("**Fuente del corpus**")
    fuente = st.radio("", ["Subir archivo", "Escribir texto"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Hiperparámetros**")
    ventana     = st.slider("Ventana de contexto", 1, 5, 2)
    dimensiones = st.slider("Dimensiones del embedding", 10, 500, 50, step=10)
    tasa        = st.select_slider("Tasa de aprendizaje", [0.001,0.005,0.01,0.05,0.1], value=0.01)
    epocas      = st.slider("Épocas", 100, 2000, 500, step=100)
    top_n       = st.slider("No. de resultados", 3, 15, 5)
    semilla     = st.number_input("Semilla aleatoria", value=42, step=1)
    st.markdown("---")
    st.caption("MinTexto v5")

st.markdown('<div class="hero-title">MinTexto v5</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Minería de texto · Word2Vec / Skip-gram</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown('<div class="sec-head">Corpus de Entrenamiento</div>', unsafe_allow_html=True)
texto_corpus = ""

if "Subir archivo" in fuente:
    arch = st.file_uploader("Sube un archivo .txt, .pdf o .docx", type=["txt","pdf","docx"], label_visibility="collapsed")
    if arch:
        ext = pathlib.Path(arch.name).suffix.lower()
        with st.spinner("Leyendo archivo…"):
            texto_corpus = leer_bytes(arch.read(), ext)
        st.success(f"{arch.name} — {len(texto_corpus):,} caracteres")
else:
    texto_corpus = st.text_area("Corpus:", height=160,
        placeholder="La minería de texto es una disciplina…",
        label_visibility="collapsed")

st.markdown("")
iniciar = st.button("Preprocesar y Entrenar")

if "modelo" not in st.session_state:
    st.session_state.modelo = None

if iniciar and texto_corpus.strip():
    st.markdown("---")
    st.markdown('<div class="sec-head">Entrenando Nuevo Modelo</div>', unsafe_allow_html=True)

    tokens = preprocesar(texto_corpus)
    vocab, vocab_inv = construir_vocabulario(tokens)
    secuencia = [vocab[p] for p in tokens]
    parejas   = generar_parejas(secuencia, ventana)
    Pe, Ps, hist = entrenar_con_progreso(vocab, parejas, dimensiones, tasa, epocas, int(semilla))

    st.session_state.modelo = {
        "texto_len": len(texto_corpus),
        "tokens": tokens, "vocab": vocab, "vocab_inv": vocab_inv,
        "parejas": parejas, "Pe": Pe, "Ps": Ps, "hist": hist
    }

elif iniciar:
    st.warning("Ingresa o sube un corpus primero.")

if st.session_state.modelo:
    m         = st.session_state.modelo
    tokens    = m["tokens"]
    vocab     = m["vocab"]
    vocab_inv = m["vocab_inv"]
    parejas   = m["parejas"]
    Pe        = m["Pe"]
    Ps        = m["Ps"]
    hist      = m["hist"]
    texto_len = m["texto_len"]

    st.markdown("---")
    st.markdown('<div class="sec-head">Preprocesamiento</div>', unsafe_allow_html=True)
    metrics = [("Caracteres", f"{texto_len:,}"),
               ("Tokens útiles", f"{len(tokens):,}"),
               ("Palabras únicas", f"{len(vocab):,}"),
               ("Pares estimados", f"{len(parejas):,}")]
    cols = st.columns(4)
    for col, (label, val) in zip(cols, metrics):
        col.markdown(f'<div class="mcard"><div class="mcard-label">{label}</div><div class="mcard-value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    pills_html = '<div class="token-wrap">' + "".join(f'<span class="tok">{t}</span>' for t in tokens) + "</div>"
    with st.expander(f"Ver todos los tokens limpios — {len(tokens)} tokens", expanded=False):
        st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Parejas Skip-gram</div>', unsafe_allow_html=True)
    df_pares = pd.DataFrame([{"Objetivo": vocab_inv[o], "Contexto": vocab_inv[c]} for o, c in parejas])
    with st.expander(f"Ver todas las parejas — {len(parejas)} pares", expanded=False):
        st.dataframe(df_pares, use_container_width=True, hide_index=True, height=320)

    st.markdown("---")
    st.markdown('<div class="sec-head">Resultados del Entrenamiento</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_perdida(hist), use_container_width=True, key="fig_perdida")

    st.markdown("---")
    st.markdown('<div class="sec-head">Similitud Coseno</div>', unsafe_allow_html=True)

    pals_demo = [p for p, c in Counter(tokens).most_common(3)]
    cols_s = st.columns(3)
    for i, (col, pal) in enumerate(zip(cols_s, pals_demo)):
        datos = top_sim(pal, vocab, vocab_inv, Pe, top_n)
        with col:
            st.plotly_chart(fig_barras(datos, f'Similares a "{pal}"', es_pct=False), use_container_width=True, key=f"sim_demo_{i}")

    st.markdown("---")
    st.markdown('<div class="sec-head">Palabras que aparecen juntas en el texto</div>', unsafe_allow_html=True)
    cols_c = st.columns(3)
    for i, (col, pal) in enumerate(zip(cols_c, pals_demo)):
        datos = top_ctx(pal, vocab, vocab_inv, Pe, Ps, top_n)
        with col:
            st.plotly_chart(fig_barras(datos, f'Contexto de "{pal}"', es_pct=True), use_container_width=True, key=f"ctx_demo_{i}")

    st.markdown("---")
    st.markdown('<div class="sec-head">Explorador de Palabras</div>', unsafe_allow_html=True)

    m  = st.session_state.modelo
    c1, c2 = st.columns([4, 1])
    with c1:
        consulta = st.text_input("Palabra:", placeholder="ej: mineria", label_visibility="collapsed")
    with c2:
        buscar = st.button("Buscar")

    if buscar and consulta.strip():
        pal = consulta.strip().lower()
        if pal not in m["vocab"]:
            st.error(f'"{pal}" no está en el vocabulario.')
            sugs = [p for p in m["vocab"] if p.startswith(pal[:3])][:6]
            if sugs: st.info("Quizás quisiste decir: " + " · ".join(sugs))
        else:
            sim_d = top_sim(pal, m["vocab"], m["vocab_inv"], m["Pe"], top_n)
            ctx_d = top_ctx(pal, m["vocab"], m["vocab_inv"], m["Pe"], m["Ps"], top_n)
            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(fig_barras(sim_d, f'Similares a "{pal}"'), use_container_width=True, key="sim_explorer")
            with cb:
                st.plotly_chart(fig_barras(ctx_d, f'Contexto de "{pal}"', es_pct=True), use_container_width=True, key="ctx_explorer")
