"""
IA responsable en acción: una demo en 4 pasos
=============================================
Instalación:
    pip install streamlit diffprivlib "scikit-learn<1.7" scipy numpy pandas

Ejecución:
    streamlit run app.py

La historia en 4 frases:
  1. Entrenamos un modelo de riesgo crediticio protegiendo la privacidad de las personas.
  2. Cada día comprobamos si los datos nuevos se parecen a los que el modelo conoció.
  3. Si algo va mal, se aplica automáticamente la medida que la política prescribe:
     reentrenar, pasar a modo seguro o apagar el modelo.
  4. Cada decisión queda en un historial que cualquiera puede revisar.

Basado en buenas prácticas internacionales (NIST AI RMF). Demo académica, no usar en producción.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from diffprivlib.models import LogisticRegression as DPLogisticRegression

SEED = 42
N_FEATURES = 6
CLIP_BOUND = 4.0  # Límite por dato: necesario para que la protección de privacidad sea válida

# Nivel de protección elegido por la persona -> parámetro técnico (a menor valor, más protección)
PROTECTION = {"Alto": 0.5, "Medio": 1.0, "Bajo": 2.0}
TOTAL_CREDIT = 4.0       # Crédito de privacidad total disponible
PSI_WATCH, PSI_ALERT = 0.10, 0.20
MAX_DROP = 0.05          # Pérdida de acierto tolerada
ACTION_NAMES = {"NONE": "Ninguna", "RETRAIN": "Reentrenar", "FALLBACK": "Modo seguro",
                "KILL_SWITCH": "Apagar el modelo"}

# =============================================================================
# LÓGICA DE NEGOCIO (misma que el PoC; cambian solo los textos)
# =============================================================================
@dataclass
class GovernanceRecord:
    """Ficha de responsabilidad del modelo: la política que se debe cumplir."""
    model_name: str
    version: str
    owner: str
    epsilon: float
    max_total_epsilon: float
    psi_threshold: float
    ks_alpha: float
    max_accuracy_drop: float
    action_on_drift: str
    action_on_critical: str
    critical_psi: float
    status: str = "ACTIVE"  # ACTIVE | RETRAINED | FALLBACK | DISABLED
    audit_log: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        """Anota cada decisión con fecha y hora (UTC)."""
        self.audit_log.append(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}|{msg}")


def make_dataset(n: int = 6000):
    return make_classification(n_samples=n, n_features=N_FEATURES, n_informative=4,
                               n_redundant=0, class_sep=2.0, flip_y=0.01, random_state=SEED)


def preprocess(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Normaliza los datos y acota los valores extremos."""
    return np.clip(scaler.transform(X), -CLIP_BOUND, CLIP_BOUND)


def train_dp_model(X: np.ndarray, y: np.ndarray, epsilon: float) -> DPLogisticRegression:
    """Entrena el modelo con protección de privacidad."""
    model = DPLogisticRegression(epsilon=epsilon, data_norm=CLIP_BOUND * np.sqrt(X.shape[1]), max_iter=200)
    model.fit(X, y)
    return model


def population_stability_index(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Mide cuánto se parece una distribución a otra (0 = idénticas)."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def detect_drift(ref: np.ndarray, cur: np.ndarray, gov: GovernanceRecord) -> Dict:
    """Compara los datos de hoy con los de referencia, dato por dato."""
    alpha_adj = gov.ks_alpha / ref.shape[1]
    feats, alerts = [], []
    for j in range(ref.shape[1]):
        _, p = stats.ks_2samp(ref[:, j], cur[:, j])
        psi = population_stability_index(ref[:, j], cur[:, j])
        drifted = (p < alpha_adj) and (psi > gov.psi_threshold)
        feats.append({"j": j, "psi": round(psi, 4), "drift": drifted})
        if drifted:
            alerts.append(f"El dato {j + 1} ha cambiado de forma notable")
    return {"features": feats, "alerts": alerts, "drift_detected": bool(alerts),
            "max_psi": max(f["psi"] for f in feats)}


class AssuranceEvaluator:
    """Supervisión continua: contrasta la evidencia con la política y actúa."""

    def __init__(self, gov: GovernanceRecord, baseline_acc: float):
        self.gov, self.baseline_acc = gov, baseline_acc
        self.epsilon_spent = gov.epsilon
        self.evidence: List[Dict] = []

    def evaluate(self, batch_name: str, drift: Dict, acc: float) -> Dict:
        gov = self.gov
        drop = self.baseline_acc - acc
        violations = []
        if drift["drift_detected"]:
            violations.append(f"Los datos han cambiado en {len(drift['alerts'])} de {N_FEATURES} variables")
        if drop > gov.max_accuracy_drop:
            violations.append(f"El modelo pierde {drop:.1%} de acierto (se tolera {gov.max_accuracy_drop:.0%})")
        critical = drift["max_psi"] > gov.critical_psi
        action = (gov.action_on_critical if critical else gov.action_on_drift) if violations else "NONE"
        record = {"batch": batch_name, "max_psi": round(drift["max_psi"], 4),
                  "accuracy": round(acc, 4), "accuracy_drop": round(drop, 4),
                  "violations": violations, "critical": critical,
                  "action_prescribed": action, "action_result": None}
        gov.log(f"{batch_name}: {len(violations)} problema(s) detectado(s). "
                f"Medida prescrita: {ACTION_NAMES[action]}.")
        self.evidence.append(record)
        return record

    def apply_action(self, record: Dict, model, X_batch, y_batch) -> Optional[object]:
        """Ejecuta la medida prescrita y devuelve el modelo que queda activo."""
        gov, action = self.gov, record["action_prescribed"]
        if action == "RETRAIN":
            if self.epsilon_spent + gov.epsilon > gov.max_total_epsilon:
                gov.status = "DISABLED"
                record["action_result"] = "Sin crédito de privacidad para reentrenar: el modelo se apaga"
                gov.log(record["action_result"])
                return None
            new_model = train_dp_model(X_batch, y_batch, gov.epsilon)
            self.epsilon_spent += gov.epsilon
            major, minor = gov.version.split(".")
            gov.version = f"{major}.{int(minor) + 1}"
            gov.status = "RETRAINED"
            record["action_result"] = f"Modelo reentrenado (versión {gov.version})"
            gov.log(record["action_result"])
            return new_model
        if action == "FALLBACK":
            gov.status = "FALLBACK"
            record["action_result"] = "Modo seguro: se usan reglas simples hasta revisar el modelo"
        elif action == "KILL_SWITCH":
            gov.status = "DISABLED"
            record["action_result"] = "Modelo apagado. Requiere revisión humana"
        else:
            record["action_result"] = "Todo en orden: no hace falta intervenir"
            return model
        gov.log(record["action_result"])
        return model if action == "FALLBACK" else None


def make_drifted_batch(X: np.ndarray, shift: float, scale: float = 1.0) -> np.ndarray:
    """Simula que el mundo cambia: altera las 3 primeras variables."""
    Xd = X.copy()
    Xd[:, :3] = Xd[:, :3] * scale + shift
    return Xd


# =============================================================================
# ESTADO DE LA SESIÓN
# =============================================================================
STATE_KEYS = ["model", "gov", "evaluator", "scaler", "X_ref", "X_stream", "y_stream",
              "baseline_acc", "history", "last_batch", "trained_eps"]


def init_state(epsilon: float) -> None:
    """Entrena el modelo inicial, crea la ficha de responsabilidad y guarda todo."""
    np.random.seed(SEED)
    X, y = make_dataset()
    X_tr, X_prod, y_tr, y_prod = train_test_split(X, y, test_size=0.5, random_state=SEED)
    scaler = StandardScaler().fit(X_tr)
    X_tr_p = preprocess(X_tr, scaler)
    model = train_dp_model(X_tr_p, y_tr, epsilon)
    gov = GovernanceRecord(
        model_name="Riesgo crediticio (privado)", version="1.0", owner="Equipo de IA y Gobernanza",
        epsilon=epsilon, max_total_epsilon=TOTAL_CREDIT, psi_threshold=PSI_ALERT, ks_alpha=0.01,
        max_accuracy_drop=MAX_DROP, action_on_drift="RETRAIN", action_on_critical="KILL_SWITCH",
        critical_psi=2.0)
    gov.log("Modelo puesto en marcha (versión 1.0)")
    X_a, X_stream, y_a, y_stream = train_test_split(X_prod, y_prod, test_size=0.6, random_state=SEED)
    base = accuracy_score(y_a, model.predict(preprocess(X_a, scaler)))
    st.session_state.update(model=model, gov=gov, evaluator=AssuranceEvaluator(gov, base),
                            scaler=scaler, X_ref=X_tr_p, X_stream=X_stream, y_stream=y_stream,
                            baseline_acc=base, history=[], last_batch=None, trained_eps=epsilon)


def run_batch(shift: float) -> None:
    """Un día de producción: predecir, comparar datos, evaluar y actuar."""
    S = st.session_state
    gov, ev = S.gov, S.evaluator
    if gov.status == "RETRAINED":
        gov.status = "ACTIVE"
    scale = 1.0 + shift * 0.2  # el cambio también aumenta la dispersión de los datos
    name = f"Día {len(S.history) + 1}"
    Xb = preprocess(make_drifted_batch(S.X_stream, shift, scale), S.scaler)
    drift = detect_drift(S.X_ref, Xb, gov)
    acc = accuracy_score(S.y_stream, S.model.predict(Xb))
    record = ev.evaluate(name, drift, acc)
    ref_used = S.X_ref
    S.model = ev.apply_action(record, S.model, Xb, S.y_stream)
    if gov.status == "RETRAINED":  # tras reentrenar, lo nuevo pasa a ser la referencia
        S.X_ref = Xb
        ev.baseline_acc = accuracy_score(S.y_stream, S.model.predict(Xb))
    S.last_batch = {"drift": drift, "record": record, "ref": ref_used, "cur": Xb}
    S.history.append({"Día": name, "Estado de los datos": psi_light(record["max_psi"])[1],
                      "Acierto": f"{record['accuracy']:.1%}",
                      "Decisión": ACTION_NAMES[record["action_prescribed"]],
                      "Resultado": record["action_result"]})


# =============================================================================
# PRESENTACIÓN: iconos Lucide (SVG en línea, sin scripts), estilos y semáforos
# =============================================================================
ICONS = {  # Trazados oficiales de Lucide
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "refresh-cw": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    "life-buoy": '<circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m9.17 14.83-4.24 4.24"/><circle cx="12" cy="12" r="4"/>',
    "power-off": '<path d="M18.36 6.64A9 9 0 0 1 20.77 15"/><path d="M6.16 6.16a9 9 0 1 0 12.68 12.68"/><path d="M12 2v4"/><path d="m2 2 20 20"/>',
    "scroll-text": '<path d="M15 12h-5"/><path d="M15 8h-5"/><path d="M19 17V5a2 2 0 0 0-2-2H4"/><path d="M8 21h12a2 2 0 0 0 2-2v-1a1 1 0 0 0-1-1H11a1 1 0 0 0-1 1v1a2 2 0 1 1-4 0V5a2 2 0 1 0-4 0v2a1 1 0 0 0 1 1h3"/>',
    "check-circle-2": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "octagon-alert": '<path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/>',
}


def icon(name: str, size: int = 20) -> str:
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'{ICONS[name]}</svg>')


STATE_ICON = {"ok": "check-circle-2", "warn": "alert-triangle", "alert": "octagon-alert"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
:root{--bg:#FFF5F7;--card:#FDE2E7;--line:#F4A6B8;--strong:#B23A63;--btn:#E36B8A;--btn-h:#C2185B;--ink:#5A2A3A;
--ok-bg:#E6EFE6;--ok:#4F7A5A;--warn-bg:#F9E6D5;--warn:#9A5B3A;--al-bg:#F7C6D3;--al:#8E1046}
html,body,.stApp,[class*="css"]{font-family:Inter,system-ui,sans-serif;color:var(--ink)}
.stApp{background:var(--bg)}
.block-container{padding-top:2.5rem;max-width:1100px}
h1,h2,h3{color:var(--strong)!important;font-weight:600!important}
[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--line)}
.stButton>button{border-radius:12px;border:1px solid var(--line);background:#fff;color:var(--strong);font-weight:500}
.stButton>button[kind="primary"]{background:var(--btn);border-color:var(--btn);color:#fff}
.stButton>button[kind="primary"]:hover{background:var(--btn-h);border-color:var(--btn-h)}
.stTabs [data-baseweb="tab-list"]{gap:1.5rem}
.stTabs [data-baseweb="tab"]{color:var(--ink);font-weight:500}
.stTabs [aria-selected="true"]{color:var(--strong)}
.stTabs [data-baseweb="tab-highlight"]{background:var(--strong)}
.stProgress>div>div>div>div{background:var(--btn)}
.stProgress>div>div>div{background:var(--card)}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#fff}
.ic{vertical-align:-4px;margin-right:8px}
.lead{font-size:1.1rem;line-height:1.6;margin:.5rem 0 1.5rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1.2rem 1.4rem;
box-shadow:0 1px 3px rgba(178,58,99,.08);height:100%}
.card small{display:block;color:var(--strong);font-weight:500;margin-bottom:.4rem}
.card b{font-size:1.25rem;font-weight:600}
.card.ok{background:var(--ok-bg);color:var(--ok);border-color:#BFD3C2}
.card.warn{background:var(--warn-bg);color:var(--warn);border-color:#E8C7A8}
.card.alert{background:var(--al-bg);color:var(--al);border-color:var(--btn)}
.card.ok small{color:var(--ok)}.card.warn small{color:var(--warn)}.card.alert small{color:var(--al)}
.card.sel{border:2px solid var(--strong);background:#fff}
.log{border-left:3px solid var(--line);padding:.4rem 1rem;margin:.5rem 0;background:#fff;border-radius:0 12px 12px 0}
.log span{color:var(--strong);font-size:.85rem;margin-right:.8rem}
.step{border-left:3px solid var(--line);padding:.5rem 1rem;margin:.5rem 0;background:#fff;border-radius:0 12px 12px 0}
.step span{color:var(--strong);font-weight:600;margin-right:.8rem}
</style>
"""


def psi_light(psi: float):
    """Traduce la medida técnica a semáforo: Estable / Vigilar / Alerta."""
    if psi > PSI_ALERT:
        return "alert", "Alerta"
    if psi > PSI_WATCH:
        return "warn", "Vigilar"
    return "ok", "Estable"


def light_card(title: str, level: str, text: str) -> str:
    return (f'<div class="card {level}"><small>{title}</small>'
            f'<b>{icon(STATE_ICON[level])}{text}</b></div>')


def banner(level: str, text: str) -> None:
    st.markdown(f'<div class="card {level}">{icon(STATE_ICON[level])}{text}</div>', unsafe_allow_html=True)


def lead(text: str) -> None:
    st.markdown(f'<p class="lead">{text}</p>', unsafe_allow_html=True)


# =============================================================================
# INTERFAZ
# =============================================================================
st.set_page_config(page_title="IA responsable en acción", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f"<h1>{icon('shield-check', 32)}IA responsable en acción</h1>", unsafe_allow_html=True)
st.caption("Un modelo de IA que protege datos personales, se vigila solo y sabe cuándo parar. "
           "Basado en buenas prácticas internacionales (NIST AI RMF).")

# ---------------- Barra lateral: 3 controles ----------------
with st.sidebar:
    st.markdown(f"<h3>{icon('lock')}Panel de control</h3>", unsafe_allow_html=True)
    level_name = st.select_slider("Nivel de protección de los datos", ["Bajo", "Medio", "Alto"], value="Medio",
                                  help="Más protección significa más privacidad, con algo menos de precisión.")
    shift = st.slider("Cuánto cambia el mundo hoy", 0.0, 5.0, 0.0, 0.1,
                      help="0 = todo igual que siempre. Valores altos simulan un cambio fuerte en los clientes.")
    run_clicked = st.button("Simular un día de actividad", type="primary", use_container_width=True)
    reset_clicked = st.button("Empezar de nuevo", use_container_width=True)

epsilon = PROTECTION[level_name]
if reset_clicked or ("trained_eps" in st.session_state and st.session_state.trained_eps != epsilon):
    for k in STATE_KEYS:  # al cambiar el nivel de protección hay que entrenar de nuevo
        st.session_state.pop(k, None)
if "gov" not in st.session_state:
    with st.spinner("Preparando el modelo protegido..."):
        init_state(epsilon)

S = st.session_state
gov, ev, last = S.gov, S.evaluator, S.last_batch
if run_clicked:
    if S.model is None:
        st.sidebar.error("El modelo está apagado. Pulsa «Empezar de nuevo».")
    else:
        run_batch(shift)
        last = S.last_batch

# ---------------- Semáforos superiores ----------------
credit_left = max(0.0, 1 - ev.epsilon_spent / gov.max_total_epsilon)
exhausted = ev.epsilon_spent + gov.epsilon > gov.max_total_epsilon
acc_drop = last["record"]["accuracy_drop"] if last else 0.0
status_view = {"ACTIVE": ("ok", "Funcionando"), "RETRAINED": ("ok", "Reentrenado"),
               "FALLBACK": ("warn", "Modo seguro"), "DISABLED": ("alert", "Apagado")}[gov.status]
priv_view = ("alert", "Agotado") if exhausted else (("warn", f"{credit_left:.0%} restante") if credit_left <= 1 / 3
                                                    else ("ok", f"{credit_left:.0%} restante"))
data_view = psi_light(last["record"]["max_psi"]) if last else ("ok", "Sin cambios")
acc_level = "alert" if acc_drop > MAX_DROP else ("warn" if acc_drop > MAX_DROP / 2 else "ok")
acc_now = last["record"]["accuracy"] if last else S.baseline_acc

cols = st.columns(4)
cols[0].markdown(light_card("El modelo", *status_view), unsafe_allow_html=True)
cols[1].markdown(light_card("Crédito de privacidad", *priv_view), unsafe_allow_html=True)
cols[2].markdown(light_card("Datos de hoy vs. ayer", *data_view), unsafe_allow_html=True)
cols[3].markdown(light_card("Acierto del modelo", acc_level, f"{acc_now:.0%}"), unsafe_allow_html=True)
st.write("")

t0, t1, t2, t3, t4 = st.tabs(["Qué hace este sistema", "Cómo protegemos los datos",
                              "Cómo vigilamos el modelo", "Qué hacemos cuando algo va mal",
                              "Historial de decisiones"])

# ---------------- Pestaña 0: qué hace este sistema ----------------
# Resume en una sola pantalla qué es el sistema, qué decide el modelo, sus 4 piezas y su ciclo de vida.
with t0:
    st.markdown(f"<h3>{icon('shield-check')}Qué hace este sistema</h3>", unsafe_allow_html=True)
    lead("Un modelo de IA que evalúa el riesgo crediticio de una persona, protegiendo sus datos "
         "personales y vigilándose a sí mismo mientras está en funcionamiento.")

    st.markdown("**Qué decide el modelo**")
    st.markdown('<div class="card">Recibe los datos de una persona, como sus ingresos y su historial de pagos. '
                'Devuelve la probabilidad de que sea una buena o una mala pagadora. '
                'Esa probabilidad ayuda a decidir, no sustituye a quien decide.</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown("**Las 4 piezas del sistema**")
    piezas = [("Protección de datos", "Se añade ruido al aprendizaje para que nadie pueda reconocer a una persona concreta."),
              ("Vigilancia del modelo", "Cada día se comprueba si los datos nuevos se parecen a los que el modelo conoce."),
              ("Respuesta automática", "Si algo se sale de la política, se aplica la medida que corresponde."),
              ("Historial y transparencia", "Cada decisión queda anotada con fecha y hora para poder revisarla.")]
    for col, (title, desc) in zip(st.columns(4), piezas):
        col.markdown(f'<div class="card"><small>{title}</small>{desc}</div>', unsafe_allow_html=True)
    st.write("")

    st.markdown("**El ciclo de vida, paso a paso**")
    pasos = ["Se entrena el modelo con datos protegidos.",
             "El modelo se pone en producción y empieza a decidir.",
             "Cada día se comprueba si los datos nuevos se parecen a los de referencia.",
             "Si algo se sale de la política, el sistema reentrena el modelo, pasa a modo seguro "
             "o lo apaga, y lo anota en el historial."]
    for i, texto in enumerate(pasos, 1):
        st.markdown(f'<div class="step"><span>{i}</span>{texto}</div>', unsafe_allow_html=True)
    st.write("")

# ---------------- Pestaña 1: privacidad ----------------
with t1:
    st.markdown(f"<h3>{icon('lock')}Privacidad de los datos</h3>", unsafe_allow_html=True)
    lead("El modelo aprende de datos de personas reales, pero añadimos «ruido» para que nadie "
         "pueda reconocer a un individuo concreto.")
    st.markdown(f"**Nivel de protección actual: {level_name}**")
    st.progress(credit_left, text=f"Crédito de privacidad restante: {credit_left:.0%}")
    st.caption("Cada vez que reentrenamos el modelo gastamos parte del crédito. Si se agota, no se puede reentrenar.")
    with st.expander("¿Qué significa esto?"):
        st.write("Más protección implica más ruido: la privacidad mejora y el acierto puede bajar un poco. "
                 "El crédito limita cuántas veces se puede volver a usar los datos con garantías.")
    with st.expander("Ficha de responsabilidad del modelo"):
        ficha = {"Modelo": gov.model_name, "Versión": gov.version, "Responsable": gov.owner,
                 "Nivel de protección": level_name, "Cambio de datos a vigilar": f"desde {PSI_WATCH:.2f}",
                 "Cambio de datos en alerta": f"desde {gov.psi_threshold:.2f}",
                 "Pérdida de acierto tolerada": f"{gov.max_accuracy_drop:.0%}",
                 "Si hay problemas": ACTION_NAMES[gov.action_on_drift],
                 "Si es crítico": ACTION_NAMES[gov.action_on_critical]}
        st.dataframe(pd.DataFrame({"Campo": ficha.keys(), "Valor": ficha.values()}),
                     hide_index=True, use_container_width=True)

# ---------------- Pestaña 2: vigilancia ----------------
with t2:
    st.markdown(f"<h3>{icon('activity')}¿Los datos de hoy se parecen a los de ayer?</h3>", unsafe_allow_html=True)
    lead("Un modelo solo es fiable si el mundo sigue pareciéndose a aquel con el que aprendió.")
    if not last:
        st.info("Todavía no hay actividad. Pulsa «Simular un día de actividad» en la barra lateral.")
    else:
        lvl, label = psi_light(last["record"]["max_psi"])
        banner(lvl, f"Estado de los datos: <b>{label}</b>")
        worst = max(last["drift"]["features"], key=lambda f: f["psi"])["j"]
        st.write("")
        st.markdown(f"**El dato que más ha cambiado (dato {worst + 1})**")
        lo = min(last["ref"][:, worst].min(), last["cur"][:, worst].min())
        hi = max(last["ref"][:, worst].max(), last["cur"][:, worst].max())
        edges = np.linspace(lo, hi, 41)
        dist = pd.DataFrame({
            "Datos de referencia": np.histogram(last["ref"][:, worst], edges, density=True)[0],
            "Datos de hoy": np.histogram(last["cur"][:, worst], edges, density=True)[0],
        }, index=((edges[:-1] + edges[1:]) / 2).round(2))
        st.line_chart(dist, color=["#F4A6B8", "#B23A63"])
        with st.expander("¿Qué significa esto?"):
            st.write("Si las dos curvas se superponen, los datos son parecidos. Si se separan, "
                     "el modelo está viendo situaciones que no conoce.")

# ---------------- Pestaña 3: acciones ----------------
with t3:
    st.markdown(f"<h3>{icon('refresh-cw')}Supervisión continua</h3>", unsafe_allow_html=True)
    lead("Si algo se sale de la política, el sistema actúa solo y deja constancia, sin esperar a que alguien se dé cuenta.")
    if not last:
        st.info("Sin actividad todavía. Simula un día para ver qué decide el sistema.")
    else:
        rec = last["record"]
        act = rec["action_prescribed"]
        banner({"NONE": "ok", "RETRAIN": "warn", "FALLBACK": "warn", "KILL_SWITCH": "alert"}[act],
               f"<b>{ACTION_NAMES[act]}.</b> {rec['action_result']}")
        for v in rec["violations"]:
            st.markdown(f"- {v}")
        st.write("")
        opts = [("RETRAIN", "refresh-cw", "Reentrenar", "El modelo aprende de los datos nuevos."),
                ("FALLBACK", "life-buoy", "Modo seguro", "Se usan reglas simples mientras se revisa."),
                ("KILL_SWITCH", "power-off", "Apagar el modelo", "Se detiene hasta que una persona lo revise.")]
        for col, (code, ic, title, desc) in zip(st.columns(3), opts):
            sel = " sel" if act == code else ""
            col.markdown(f'<div class="card{sel}"><small>{title}</small>{icon(ic, 24)}{desc}</div>',
                         unsafe_allow_html=True)

# ---------------- Pestaña 4: historial ----------------
with t4:
    st.markdown(f"<h3>{icon('scroll-text')}Historial de decisiones</h3>", unsafe_allow_html=True)
    lead("Cada decisión queda registrada con fecha y hora para poder revisarla cuando haga falta.")
    for entry in reversed(gov.audit_log):
        ts, msg = entry.split("|", 1)
        st.markdown(f'<div class="log"><span>{ts} UTC</span>{msg}</div>', unsafe_allow_html=True)
    if S.history:
        with st.expander("Resumen por día"):
            st.dataframe(pd.DataFrame(S.history), hide_index=True, use_container_width=True)