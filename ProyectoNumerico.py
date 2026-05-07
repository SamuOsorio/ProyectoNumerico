import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Datos base
# ──────────────────────────────────────────────────────────────────────────────
X_BASE = np.array([6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
Y_BASE = np.array([22.0, 24.5, 27.0, 30.2, 31.5, 31.0, 29.8, 27.5, 25.0, 23.2])

# ──────────────────────────────────────────────────────────────────────────────
# Funciones del spline cúbico natural
# ──────────────────────────────────────────────────────────────────────────────
def trazadores_cubicos_naturales(x, y):
    n = len(x) - 1
    a = np.array(y, dtype=float)
    h = np.diff(x).astype(float)
    c = np.zeros(n + 1)
    if n > 1:
        A = np.zeros((n - 1, n - 1))
        B = np.zeros(n - 1)
        for i in range(1, n):
            row = i - 1
            if row > 0:       A[row, row - 1] = h[i - 1]
            A[row, row]     = 2 * (h[i - 1] + h[i])
            if row < n - 2:   A[row, row + 1] = h[i]
            B[row] = (3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1])
        c[1:n] = np.linalg.solve(A, B)
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n):
        b[j] = (a[j+1]-a[j])/h[j] - (h[j]*(2*c[j]+c[j+1]))/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    return a, b, c, d, h

def _tramo(x_nodos, xq):
    n = len(x_nodos) - 1
    for i in range(n - 1):
        if x_nodos[i] <= xq < x_nodos[i + 1]:
            return i
    return n - 1

def evaluar_spline(x_nodos, a, b, c, d, xq):
    j = _tramo(x_nodos, xq)
    dx = xq - x_nodos[j]
    return a[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3

def derivada_spline(x_nodos, b, c, d, xq):
    j = _tramo(x_nodos, xq)
    dx = xq - x_nodos[j]
    return (b[j] + 2*c[j]*dx + 3*d[j]*dx**2) / 60

def interp_lineal(x_nodos, y_nodos, xq):
    for i in range(len(x_nodos) - 1):
        if x_nodos[i] <= xq <= x_nodos[i + 1]:
            t = (xq - x_nodos[i]) / (x_nodos[i + 1] - x_nodos[i])
            return y_nodos[i] + t * (y_nodos[i + 1] - y_nodos[i])
    return y_nodos[-1]

def calcular_rmse_holdout(x, y, train_ratio=0.8):
    n_train = int(len(x) * train_ratio)
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_te, y_te = x[n_train:], y[n_train:]
    a2, b2, c2, d2, _ = trazadores_cubicos_naturales(x_tr, y_tr)
    err_sp, err_ln = [], []
    for xi, yi in zip(x_te, y_te):
        err_sp.append((yi - evaluar_spline(x_tr, a2, b2, c2, d2, xi))**2)
        err_ln.append((yi - interp_lineal(x_tr, y_tr, xi))**2)
    return np.sqrt(np.mean(err_sp)), np.sqrt(np.mean(err_ln))

# ──────────────────────────────────────────────────────────────────────────────
# Helper: decimal → "HH:MM"   ← CORRECCIÓN del bug 11:79
# ──────────────────────────────────────────────────────────────────────────────
def dec_a_hhmm(dec):
    """Convierte hora decimal (ej. 11.9833) a string 'HH:MM' sin errores de redondeo."""
    minutos_totales = int(dec * 60 + 0.5)   # +0.5 = redondeo correcto, sin float drift
    hh = minutos_totales // 60
    mm = minutos_totales % 60
    hh = min(hh, 24)                         # clamp por si acaso llega a 24:00
    return f"{hh:02d}:{mm:02d}"

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CoolSpline — EcoData Solutions", layout="wide")

st.markdown("""
<div style="background:#185FA5;border-radius:10px;padding:14px 20px;margin-bottom:20px">
  <span style="color:white;font-size:20px;font-weight:bold">❄ CoolSpline</span>
  <span style="color:#B5D4F4;font-size:13px;margin-left:12px">
    EcoData Solutions — Control de temperatura en tiempo real
  </span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Pestañas principales
# ──────────────────────────────────────────────────────────────────────────────
tab_dash, tab_sim = st.tabs(["Dashboard en tiempo real", "Simulador de escenarios"])

# ══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:

    # Precalcular con datos base
    a, b, c, d, _ = trazadores_cubicos_naturales(X_BASE, Y_BASE)
    t_fino = np.linspace(6, 24, 1080)
    y_sp  = np.array([evaluar_spline(X_BASE, a, b, c, d, t) for t in t_fino])
    y_lin = np.array([interp_lineal(X_BASE, Y_BASE, t) for t in t_fino])
    y_der = np.array([derivada_spline(X_BASE, b, c, d, t) for t in t_fino])
    rmse_sp, rmse_ln = calcular_rmse_holdout(X_BASE, Y_BASE)

    # ── CSS: ocultar valor flotante y límites del slider de hora ───────────
    st.markdown("""
    <style>
    /* Oculta el tooltip con el valor crudo (872) */
    [data-testid="stSlider"][aria-label*="Hora"] div[data-testid="stTickBarMin"],
    [data-testid="stSlider"][aria-label*="Hora"] div[data-testid="stTickBarMax"],
    div[class*="stSlider"]:has(label p:first-child) div[data-testid="stTickBarMin"],
    div[class*="stSlider"]:has(label p:first-child) div[data-testid="stTickBarMax"] { display: none !important; }
    /* Selector más amplio que cubre el thumb label y los min/max */
    [data-baseweb="slider"] [data-testid="stTickBarMin"],
    [data-baseweb="slider"] [data-testid="stTickBarMax"] { visibility: hidden !important; }
    </style>
    <style>
    /* Apunta específicamente al slider con key slider_hora_dashboard */
    div:has(> div > div > [data-testid="stSlider"]) span[data-testid="stTickBarMin"],
    div:has(> div > div > [data-testid="stSlider"]) span[data-testid="stTickBarMax"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Controles ──────────────────────────────────────────────────────────
    col_sl, col_lbl = st.columns([5, 1])
    with col_sl:
        hora_raw = st.slider(
            "Hora del día",
            min_value=360, max_value=1440, value=720, step=1,
            format="‎",           # carácter invisible — suprime el tooltip con el número crudo
            key="slider_hora_dashboard"
        )
        
        # Mostrar los límites en formato HH:MM justo debajo del slider
        st.markdown(
            "<div style='display:flex;justify-content:space-between;"
            "margin-top:-18px;margin-bottom:8px;"
            "font-size:12px;color:#888;padding:0 2px'>"
            "<span>06:00</span><span>24:00</span></div>",
            unsafe_allow_html=True
        )

    hora_decimal = hora_raw / 60.0
    with col_lbl:
        st.markdown(f"### {dec_a_hhmm(hora_decimal)}")

    col_umb, col_chk1, col_chk2 = st.columns([2, 1.5, 1.5])
    with col_umb:
        umbral = st.slider("🌡 Umbral de temperatura (°C)", 25.0, 34.0, 30.0, 0.5)
    with col_chk1:
        mostrar_lineal = st.checkbox("Mostrar interpolación lineal", value=True)
    with col_chk2:
        mostrar_deriv = st.checkbox("Mostrar derivada dT/dt", value=True)

    # ── Métricas ───────────────────────────────────────────────────────────
    temp  = evaluar_spline(X_BASE, a, b, c, d, hora_decimal)
    deriv = derivada_spline(X_BASE, b, c, d, hora_decimal)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Temperatura actual", f"{temp:.2f} °C")
    with m2:
        st.metric("Velocidad cambio", f"{deriv:+.4f} °C/min")
    with m3:
        idx_max = int(np.argmax(y_sp))
        st.metric("Temp. máxima del día", f"{y_sp[idx_max]:.2f} °C",
                  delta=f"a las {dec_a_hhmm(t_fino[idx_max])}")
    with m4:
        mejora = (rmse_ln - rmse_sp) / rmse_ln * 100 if rmse_ln != 0 else 0
        st.metric("RMSE Spline / Lineal",
                  f"{rmse_sp:.4f} / {rmse_ln:.4f}",
                  delta=f"mejora {mejora:.1f}%")

    # ── Alarma ─────────────────────────────────────────────────────────────
    if temp > 32:
        st.error(f"ALARMA CRÍTICA — T={temp:.2f}°C > 32°C. Activar protocolo de emergencia.")
    elif temp > umbral:
        st.warning(f"Ventiladores ON — T={temp:.2f}°C supera umbral de {umbral:.1f}°C.")
    elif abs(deriv) > 0.3:
        signo = "+" if deriv > 0 else ""
        st.warning(f"Cambio rápido — dT/dt = {signo}{deriv:.4f} °C/min")
    else:
        st.success("Sistema normal")

    # ── Gráfica ────────────────────────────────────────────────────────────
    n_plots = 2 if mostrar_deriv else 1
    fig, axes = plt.subplots(n_plots, 1,
                             figsize=(12, 7 if mostrar_deriv else 4),
                             sharex=True)
    ax1 = axes[0] if mostrar_deriv else axes

    ax1.plot(t_fino, y_sp, color="#185FA5", lw=2, label="Spline cúbico")
    if mostrar_lineal:
        ax1.plot(t_fino, y_lin, color="#A32D2D", lw=1.5, ls="--",
                 label="Interpolación lineal")
    ax1.scatter(X_BASE, Y_BASE, color="#185FA5", zorder=5, s=50, label="Datos medidos")
    ax1.axhline(umbral, color="#BA7517", lw=1.2, ls=":", label=f"Umbral {umbral:.1f}°C")
    ax1.axhline(32, color="#A32D2D", lw=1.2, ls=":", label="Alarma 32°C")
    ax1.axvline(hora_decimal, color="#888", lw=1, ls="--", alpha=0.6)
    ax1.scatter([hora_decimal], [temp], color="#BA7517", zorder=10, s=130,
                marker="D", label=f"T({dec_a_hhmm(hora_decimal)}) = {temp:.2f}°C")
    ax1.fill_between(t_fino, umbral, y_sp,
                     where=(y_sp > umbral), alpha=0.12, color="#A32D2D",
                     label="Zona > umbral")
    ax1.set_ylabel("Temperatura (°C)", fontsize=11)
    ax1.set_ylim(19, 35)
    ax1.legend(fontsize=8, loc="upper right", ncol=2)
    ax1.grid(alpha=0.25)
    ax1.set_title("EcoData Solutions — CoolSpline", fontsize=12)

    if mostrar_deriv:
        ax2 = axes[1]
        ax2.plot(t_fino, y_der, color="#0F6E56", lw=1.8, label="dT/dt (°C/min)")
        ax2.axhline( 0.3, color="#A32D2D", ls=":", lw=1, label="±0.3 °C/min")
        ax2.axhline(-0.3, color="#A32D2D", ls=":", lw=1)
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.axvline(hora_decimal, color="#888", lw=1, ls="--", alpha=0.6)
        ax2.scatter([hora_decimal], [deriv], color="#BA7517", zorder=10,
                    s=100, marker="D")
        ax2.fill_between(t_fino, -0.3, y_der, where=(y_der < -0.3),
                         alpha=0.12, color="#A32D2D")
        ax2.fill_between(t_fino,  0.3, y_der, where=(y_der >  0.3),
                         alpha=0.12, color="#A32D2D")
        ax2.set_ylabel("dT/dt (°C/min)", fontsize=11)
        ax2.set_xlabel("Hora (decimal)", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.25)
    else:
        ax1.set_xlabel("Hora (decimal)", fontsize=11)

    xticks = np.arange(6, 25, 2)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([dec_a_hhmm(float(t)) for t in xticks])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  PESTAÑA 2 — Simulador de escenarios
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:

    st.markdown("### Simulador de escenarios de temperatura")
    st.markdown(
        "Genera escenarios ficticios de temperatura y compara cómo responde "
        "el spline cúbico. Podés ajustar la curva base con desplazamiento, "
        "amplitud de ruido y agregar un pico de calor simulado."
    )

    # ── Controles de simulación ────────────────────────────────────────────
    st.markdown("#### Parámetros del escenario")

    c1, c2, c3 = st.columns(3)
    with c1:
        desplazamiento = st.slider(
            "Desplazamiento global (°C)",
            -5.0, 10.0, 0.0, 0.5,
            help="Sube o baja toda la curva. Simula días más cálidos o fríos."
        )
        ruido_std = st.slider(
            "Ruido aleatorio (°C)",
            0.0, 3.0, 0.0, 0.1,
            help="Agrega variabilidad aleatoria a las mediciones."
        )
    with c2:
        pico_activo = st.checkbox("Agregar pico de calor", value=False)
        if pico_activo:
            pico_hora = st.slider("Hora del pico", 6, 23, 14,
                                  help="Hora central del pico (formato decimal entero).")
            pico_magnitud = st.slider("Magnitud del pico (°C)", 1.0, 8.0, 3.0, 0.5)
            pico_ancho = st.slider("Duración del pico (horas)", 1.0, 4.0, 2.0, 0.5)
    with c3:
        n_sim = st.slider(
            "Número de simulaciones Monte Carlo",
            1, 50, 10,
            help="Repite la simulación con ruido aleatorio N veces para ver la envolvente."
        )
        semilla = st.number_input("Semilla aleatoria", value=42, step=1)
        umbral_sim = st.slider("Umbral alarma simulación (°C)", 25.0, 34.0, 30.0, 0.5)

    # ── Generar datos simulados ────────────────────────────────────────────
    np.random.seed(int(semilla))

    def generar_escenario(desp, ruido, pico=False, hora_p=14, mag_p=3, ancho_p=2):
        y_sim = Y_BASE.copy() + desp
        if ruido > 0:
            y_sim += np.random.normal(0, ruido, size=len(y_sim))
        if pico:
            for i, xi in enumerate(X_BASE):
                y_sim[i] += mag_p * np.exp(-0.5 * ((xi - hora_p) / (ancho_p/2))**2)
        return y_sim

    # Escenario principal (sin ruido, para mostrar la curva base modificada)
    y_sim_base = generar_escenario(
        desplazamiento, 0.0,
        pico_activo,
        pico_hora if pico_activo else 14,
        pico_magnitud if pico_activo else 3,
        pico_ancho if pico_activo else 2
    )

    a_s, b_s, c_s, d_s, _ = trazadores_cubicos_naturales(X_BASE, y_sim_base)
    t_fino_s = np.linspace(6, 24, 1080)
    y_sp_sim = np.array([evaluar_spline(X_BASE, a_s, b_s, c_s, d_s, t) for t in t_fino_s])
    y_lin_sim = np.array([interp_lineal(X_BASE, y_sim_base, t) for t in t_fino_s])

    # Monte Carlo: N simulaciones con ruido
    mc_curvas = []
    for _ in range(n_sim):
        y_mc = generar_escenario(
            desplazamiento, ruido_std,
            pico_activo,
            pico_hora if pico_activo else 14,
            pico_magnitud if pico_activo else 3,
            pico_ancho if pico_activo else 2
        )
        a_mc, b_mc, c_mc, d_mc, _ = trazadores_cubicos_naturales(X_BASE, y_mc)
        mc_curvas.append([evaluar_spline(X_BASE, a_mc, b_mc, c_mc, d_mc, t) for t in t_fino_s])

    mc_arr     = np.array(mc_curvas)
    mc_min     = mc_arr.min(axis=0)
    mc_max     = mc_arr.max(axis=0)
    mc_mean    = mc_arr.mean(axis=0)

    # ── Métricas del escenario ─────────────────────────────────────────────
    idx_max_sim = int(np.argmax(y_sp_sim))
    temp_max_sim = y_sp_sim[idx_max_sim]
    hora_max_sim = dec_a_hhmm(t_fino_s[idx_max_sim])
    minutos_sobre_umbral = int(np.sum(y_sp_sim > umbral_sim) * (18 * 60 / 1080))
    rmse_sp_sim, rmse_ln_sim = calcular_rmse_holdout(X_BASE, y_sim_base)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Temp. máxima simulada", f"{temp_max_sim:.2f} °C",
                  delta=f"a las {hora_max_sim}")
    with s2:
        st.metric("Minutos sobre umbral", f"{minutos_sobre_umbral} min",
                  delta=f"umbral {umbral_sim:.1f}°C")
    with s3:
        diferencia = temp_max_sim - y_sp[idx_max_sim]
        st.metric("Diferencia vs caso base",
                  f"{diferencia:+.2f} °C",
                  delta="sobre el máximo original")
    with s4:
        mejora_s = (rmse_ln_sim - rmse_sp_sim)/rmse_ln_sim*100 if rmse_ln_sim != 0 else 0
        st.metric("RMSE Spline / Lineal",
                  f"{rmse_sp_sim:.4f} / {rmse_ln_sim:.4f}",
                  delta=f"mejora {mejora_s:.1f}%")

    # Alarma del escenario
    t_actual = hora_decimal if "hora_decimal" in dir() else 12.0
    temp_sim_ahora = evaluar_spline(X_BASE, a_s, b_s, c_s, d_s, 14.0)
    if temp_max_sim > 32:
        st.error(f"ALARMA CRÍTICA en el escenario — máximo {temp_max_sim:.2f}°C > 32°C.")
    elif temp_max_sim > umbral_sim:
        st.warning(f"El escenario supera el umbral — máximo {temp_max_sim:.2f}°C > {umbral_sim:.1f}°C.")
    else:
        st.success(f"Escenario bajo control — máximo {temp_max_sim:.2f}°C ≤ {umbral_sim:.1f}°C.")

    # ── Gráfica de simulación ──────────────────────────────────────────────
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot izquierdo: escenario vs caso base
    axA.plot(t_fino_s, y_sp, color="#185FA5", lw=1.5, ls="--",
             label="Spline base (original)", alpha=0.7)
    axA.plot(t_fino_s, y_sp_sim, color="#A32D2D", lw=2,
             label="Spline escenario simulado")
    axA.plot(t_fino_s, y_lin_sim, color="#BA7517", lw=1.2, ls=":",
             label="Lineal escenario")
    axA.scatter(X_BASE, Y_BASE, color="#185FA5", s=40, zorder=5,
                label="Datos originales", alpha=0.7)
    axA.scatter(X_BASE, y_sim_base, color="#A32D2D", s=60, zorder=6,
                marker="^", label="Datos simulados")
    axA.axhline(umbral_sim, color="#BA7517", lw=1.2, ls=":",
                label=f"Umbral {umbral_sim:.1f}°C")
    axA.axhline(32, color="#A32D2D", lw=1, ls=":", alpha=0.5)
    axA.fill_between(t_fino_s, umbral_sim, y_sp_sim,
                     where=(y_sp_sim > umbral_sim), alpha=0.15, color="#A32D2D")
    axA.set_title("Escenario simulado vs caso base", fontsize=11)
    axA.set_ylabel("Temperatura (°C)", fontsize=10)
    axA.set_xlabel("Hora", fontsize=10)
    axA.set_ylim(15, 42)
    axA.legend(fontsize=7, ncol=2)
    axA.grid(alpha=0.25)
    axA.set_xticks(np.arange(6, 25, 2))
    axA.set_xticklabels([dec_a_hhmm(float(t)) for t in np.arange(6, 25, 2)], fontsize=8)

    # Subplot derecho: envolvente Monte Carlo
    if n_sim > 1 and ruido_std > 0:
        axB.fill_between(t_fino_s, mc_min, mc_max,
                         alpha=0.2, color="#185FA5", label="Rango Monte Carlo")
        for curva in mc_curvas:
            axB.plot(t_fino_s, curva, color="#185FA5", lw=0.4, alpha=0.3)
    axB.plot(t_fino_s, mc_mean, color="#185FA5", lw=2, label="Media Monte Carlo")
    axB.plot(t_fino_s, y_sp_sim, color="#A32D2D", lw=2, ls="--",
             label="Escenario sin ruido")
    axB.axhline(umbral_sim, color="#BA7517", lw=1.2, ls=":",
                label=f"Umbral {umbral_sim:.1f}°C")
    axB.axhline(32, color="#A32D2D", lw=1, ls=":", alpha=0.5)

    # Probabilidad de superar umbral en cada punto
    prob_sobre = (mc_arr > umbral_sim).mean(axis=0)
    ax_twin = axB.twinx()
    ax_twin.fill_between(t_fino_s, 0, prob_sobre * 100,
                         alpha=0.15, color="#0F6E56")
    ax_twin.plot(t_fino_s, prob_sobre * 100, color="#0F6E56",
                 lw=1, ls="-.", label="P(T>umbral) %")
    ax_twin.set_ylabel("Probabilidad de alarma (%)", fontsize=9, color="#0F6E56")
    ax_twin.set_ylim(0, 120)
    ax_twin.tick_params(axis='y', labelcolor="#0F6E56")

    axB.set_title(f"Envolvente Monte Carlo (N={n_sim})", fontsize=11)
    axB.set_ylabel("Temperatura (°C)", fontsize=10)
    axB.set_xlabel("Hora", fontsize=10)
    axB.set_ylim(15, 42)
    axB.legend(fontsize=7, loc="upper left")
    axB.grid(alpha=0.25)
    axB.set_xticks(np.arange(6, 25, 2))
    axB.set_xticklabels([dec_a_hhmm(float(t)) for t in np.arange(6, 25, 2)], fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Tabla comparativa de escenarios ───────────────────────────────────
    st.markdown("#### Tabla comparativa: escenario simulado vs caso base")
    tabla_data = {
        "Métrica":             ["Temp. máxima (°C)", "Hora del máximo",
                                "Minutos sobre umbral", "RMSE spline", "RMSE lineal"],
        "Caso base":           [f"{y_sp[int(np.argmax(y_sp))]:.2f}",
                                dec_a_hhmm(t_fino_s[int(np.argmax(y_sp))]),
                                str(int(np.sum(y_sp > umbral_sim) * (18*60/1080))),
                                f"{rmse_sp:.4f}", f"{rmse_ln:.4f}"],
        "Escenario simulado":  [f"{temp_max_sim:.2f}", hora_max_sim,
                                str(minutos_sobre_umbral),
                                f"{rmse_sp_sim:.4f}", f"{rmse_ln_sim:.4f}"],
    }
    st.table(tabla_data)

    # ── Nota metodológica ──────────────────────────────────────────────────
    with st.expander("¿Cómo funciona el simulador?"):
        st.markdown("""
**Desplazamiento global:** suma una constante a todos los puntos.
Simula un día más cálido (+) o más frío (−) que el promedio.

**Ruido aleatorio:** agrega `N(0, σ)` a cada medición.
Modela imprecisión de sensores o variabilidad climática.

**Pico de calor:** añade una gaussiana centrada en la hora elegida.
Simula eventos como apertura de puertas, fallo de un rack, etc.

**Monte Carlo:** repite la generación N veces con ruido independiente.
El área sombreada muestra el rango [mín, máx] de todos los splines resultantes.
La curva verde (eje derecho) estima la **probabilidad de que T supere el umbral**
en cada minuto del día.
        """)