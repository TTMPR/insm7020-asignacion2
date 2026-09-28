"""Analisis reproducible - INSM 7020 Asignacion 2 (version 2).

Tema: Selectividad institucional e ingresos posteriores de graduados de Computer Science.
Fuente: U.S. Department of Education, College Scorecard (publicacion de junio de 2026).

Archivos esperados en el mismo directorio (dentro de College_Scorecard_Raw_Data):
- FieldOfStudyData2122_2223_PP.csv
- MERGED2022_23_PP.csv

Este script produce TODAS las cifras reportadas en el documento, las tres figuras
y la imagen del tablero. Requisitos: pandas, numpy, scipy, statsmodels, matplotlib.
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

SEED = 7020
BASE = Path(__file__).resolve().parent
FOS = BASE / "FieldOfStudyData2122_2223_PP.csv"
INST = BASE / "MERGED2022_23_PP.csv"
OUT = BASE / "salidas_asignacion2"
OUT.mkdir(exist_ok=True)
FUENTE = "Fuente: U.S. Department of Education, College Scorecard (2026)."
log = []


def reg(txt):
    """Imprime y guarda cada resultado en resumen_resultados.txt."""
    log.append(txt)
    print(txt)


# 1. Importacion y conversion numerica ---------------------------------------
fos_cols = ["UNITID", "OPEID6", "INSTNM", "CONTROL", "CIPCODE", "CIPDESC",
            "CREDLEV", "CREDDESC", "IPEDSCOUNT1", "IPEDSCOUNT2",
            "EARN_COUNT_WNE_4YR", "EARN_MDN_4YR"]
inst_cols = ["UNITID", "OPEID6", "INSTNM", "MAIN", "CONTROL", "STABBR", "REGION",
             "ADM_RATE", "ADM_RATE_ALL", "SAT_AVG", "SAT_AVG_ALL", "UGDS"]
fos = pd.read_csv(FOS, usecols=fos_cols, dtype=str, low_memory=False)
inst = pd.read_csv(INST, usecols=inst_cols, dtype=str, low_memory=False)

# El codigo PS (PrivacySuppressed) y otros valores no numericos pasan a NaN.
for c in ["UNITID", "OPEID6", "CREDLEV", "IPEDSCOUNT1", "IPEDSCOUNT2",
          "EARN_COUNT_WNE_4YR", "EARN_MDN_4YR"]:
    fos[c] = pd.to_numeric(fos[c], errors="coerce")
for c in ["UNITID", "OPEID6", "MAIN", "CONTROL", "REGION", "ADM_RATE",
          "ADM_RATE_ALL", "SAT_AVG", "SAT_AVG_ALL", "UGDS"]:
    inst[c] = pd.to_numeric(inst[c], errors="coerce")

# 2. Poblacion: CS (CIP 11.07), bachillerato, publicas y privadas sin fines de lucro
cs = fos[fos["CIPCODE"].astype(str).eq("1107")
         & fos["CREDLEV"].eq(3)
         & fos["CONTROL"].isin(["Public", "Private, nonprofit"])].copy()
reg(f"Filtro CS bachillerato: filas={len(cs)}; OPEID6 unicos={cs['OPEID6'].nunique()}; "
    f"OPEID6 repetidos={(cs['OPEID6'].value_counts() > 1).sum()}")

# 3. Deduplicacion a OPEID6 (los earnings se agregan a ese nivel) -------------
cs["earn_nonmiss"] = cs["EARN_MDN_4YR"].notna().astype(int)
cs_ope = (cs.sort_values(["OPEID6", "earn_nonmiss", "UNITID"], ascending=[True, False, True])
            .drop_duplicates("OPEID6", keep="first").drop(columns="earn_nonmiss"))

# 4. Archivo institucional: una fila por OPEID6, priorizando el campus principal
inst["adm_nonmiss"] = inst["ADM_RATE_ALL"].notna().astype(int)
inst_ope = (inst[inst["CONTROL"].isin([1, 2])]
            .sort_values(["OPEID6", "MAIN", "adm_nonmiss", "UNITID"],
                         ascending=[True, False, False, True])
            .drop_duplicates("OPEID6", keep="first").drop(columns="adm_nonmiss"))
m = cs_ope.merge(
    inst_ope[["OPEID6", "UNITID", "INSTNM", "MAIN", "CONTROL", "STABBR", "REGION",
              "ADM_RATE", "ADM_RATE_ALL", "SAT_AVG", "SAT_AVG_ALL", "UGDS"]],
    on="OPEID6", how="left", suffixes=("_fos", "_inst"))
m["admit_rate"] = m["ADM_RATE_ALL"].where(m["ADM_RATE_ALL"].notna(), m["ADM_RATE"])
m["sat_avg"] = m["SAT_AVG_ALL"].where(m["SAT_AVG_ALL"].notna(), m["SAT_AVG"])

miss = (m[["EARN_MDN_4YR", "admit_rate", "CONTROL_fos", "REGION", "UGDS", "sat_avg"]]
        .isna().mean().mul(100).round(1))
miss.to_csv(OUT / "faltantes_marco_inicial.csv", header=["pct_faltante"])
reg(f"Marco inicial={len(m)}; ingreso publicado={m['EARN_MDN_4YR'].notna().sum()}; "
    f"admision valida={m['admit_rate'].notna().sum()}; "
    f"sin enlace institucional={m['UNITID_inst'].isna().sum()}")
reg("Faltantes en marco inicial (%): " + "; ".join(f"{k}={v}" for k, v in miss.items()))

# 5. Marco analitico (casos completos) y evaluacion del sesgo por descarte ---
frame = m.dropna(subset=["EARN_MDN_4YR", "admit_rate"]).copy()
frame = frame[(frame["EARN_MDN_4YR"] > 0) & frame["admit_rate"].between(0, 1)]
m["en_marco"] = m["OPEID6"].isin(frame["OPEID6"])
bias = m.groupby("en_marco").agg(
    n=("OPEID6", "size"),
    admision_mediana=("admit_rate", "median"),
    matricula_ug_mediana=("UGDS", "median"),
    graduados_cs_mediana=("IPEDSCOUNT1", "median"),
    pct_privadas=("CONTROL_fos", lambda s: (s == "Private, nonprofit").mean() * 100))
bias.to_csv(OUT / "sesgo_descarte.csv")
reg("Excluidas (False) vs. marco analitico (True):\n" + bias.round(3).to_string())

bins = [-np.inf, .20, .50, .80, np.inf]
labels = ["Altamente selectiva (<20%)", "Selectiva (20-49.9%)",
          "Moderada (50-79.9%)", "Acceso amplio (80%+)"]
short = ["<20%", "20-49.9%", "50-79.9%", "80%+"]
frame["selectivity_group"] = pd.cut(frame["admit_rate"], bins=bins, labels=labels, right=False)
reg(f"Marco analitico N={len(frame)}; por estrato: " + ", ".join(
    f"{k}={v}" for k, v in frame["selectivity_group"].value_counts(sort=False).items()))

# 6. Tamano muestral: proporcion, 95 %, +/-5 %, correccion por poblacion finita
N = len(frame)
z, p0, q0, e = 1.96, .50, .50, .05
n0 = z**2 * p0 * q0 / e**2
n_fpc = n0 / (1 + (n0 - 1) / N)
n_required = int(np.ceil(n_fpc))
reg(f"n0={n0:.2f}; n corregido={n_fpc:.2f}; n requerido={n_required}")

# 7. Muestreo aleatorio estratificado proporcional (semilla fija) ------------
counts = frame["selectivity_group"].value_counts(sort=False)
raw_alloc = counts / counts.sum() * n_required
alloc = np.floor(raw_alloc).astype(int)
for group in (raw_alloc - alloc).sort_values(ascending=False).index[:n_required - alloc.sum()]:
    alloc.loc[group] += 1
rng = np.random.default_rng(SEED)
parts = []
for group in labels:
    g = frame[frame["selectivity_group"] == group]
    parts.append(g.sample(n=int(alloc.loc[group]), random_state=int(rng.integers(0, 2**31 - 1))))
sample = pd.concat(parts).sort_values("OPEID6").copy()
reg("Afijacion: " + ", ".join(f"{k}={v}" for k, v in alloc.items()))

# 8. Estadistica descriptiva ---------------------------------------------------
y = sample["EARN_MDN_4YR"]
desc = pd.DataFrame([{
    "variable": v, "n": sample[v].notna().sum(), "media": sample[v].mean(),
    "mediana": sample[v].median(), "DE": sample[v].std(ddof=1), "min": sample[v].min(),
    "Q1": sample[v].quantile(.25), "Q3": sample[v].quantile(.75), "max": sample[v].max()}
    for v in ["EARN_MDN_4YR", "admit_rate", "UGDS", "sat_avg"]])
desc.to_csv(OUT / "estadisticos_descriptivos.csv", index=False)
reg("Descriptivos:\n" + desc.round(3).to_string(index=False))

sw = stats.shapiro(y)
q1, q3 = y.quantile([.25, .75])
n_out = int(((y < q1 - 1.5 * (q3 - q1)) | (y > q3 + 1.5 * (q3 - q1))).sum())
reg(f"Asimetria={stats.skew(y, bias=False):.2f}; curtosis (exceso)="
    f"{stats.kurtosis(y, bias=False):.2f}; Shapiro-Wilk W={sw.statistic:.3f}, p={sw.pvalue:.3g}; "
    f"atipicos 1.5*RIC={n_out}; SAT faltante={sample['sat_avg'].isna().sum()}")

freq = []
for var, s in [("Grupo de selectividad", sample["selectivity_group"].astype(str)),
               ("Control institucional", sample["CONTROL_fos"]),
               ("Region IPEDS", sample["REGION"].astype(int).astype(str))]:
    for cat, v in s.value_counts().sort_index().items():
        freq.append({"variable": var, "categoria": cat, "n": v, "pct": round(v / len(s) * 100, 1)})
freq = pd.DataFrame(freq)
freq.to_csv(OUT / "frecuencias_categoricas.csv", index=False)
reg("Frecuencias:\n" + freq.to_string(index=False))

grp = sample.groupby("selectivity_group", observed=True)["EARN_MDN_4YR"].agg(
    ["count", "mean", "median", "std"])
grp.to_csv(OUT / "ingreso_por_grupo.csv")
reg("Ingreso por grupo:\n" + grp.round(1).to_string())

# 9. Prueba principal: Spearman con IC bootstrap percentil --------------------
rho, p_s = stats.spearmanr(sample["admit_rate"], y)
ns = len(sample)
t_approx = rho * np.sqrt((ns - 2) / (1 - rho**2))
rng = np.random.default_rng(SEED)
x_a, y_a = sample["admit_rate"].to_numpy(), y.to_numpy()
boot = []
for _ in range(10000):
    idx = rng.integers(0, ns, ns)
    boot.append(stats.spearmanr(x_a[idx], y_a[idx]).statistic)
ci_s = np.nanpercentile(boot, [2.5, 97.5])
reg(f"Spearman rho={rho:.3f}; t({ns - 2})={t_approx:.2f}; p={p_s:.3g}; "
    f"IC95 bootstrap=[{ci_s[0]:.3f}, {ci_s[1]:.3f}]")

r_p, p_p = stats.pearsonr(sample["admit_rate"], y)
reg(f"Sensibilidad Pearson r={r_p:.3f}; p={p_p:.3g}")
rho_c, p_c = stats.spearmanr(frame["admit_rate"], frame["EARN_MDN_4YR"])
reg(f"Contraste con el marco completo (N={N}): rho={rho_c:.3f}; p={p_c:.3g}")
no_pr = sample[sample["STABBR"] != "PR"]
rho_pr, p_pr = stats.spearmanr(no_pr["admit_rate"], no_pr["EARN_MDN_4YR"])
reg("Instituciones de Puerto Rico en la muestra:\n" + sample.loc[
    sample["STABBR"] == "PR", ["INSTNM_fos", "admit_rate", "EARN_MDN_4YR", "selectivity_group"]
].to_string(index=False))
reg(f"Sensibilidad sin Puerto Rico (n={len(no_pr)}): rho={rho_pr:.3f}; p={p_pr:.3g}")

# 10. Comparacion secundaria: Levene, Welch, omega cuadrada, Welch-Holm -----
arrays = [sample.loc[sample["selectivity_group"] == g, "EARN_MDN_4YR"].to_numpy() for g in labels]
levene = stats.levene(*arrays, center="median")
welch = anova_oneway(arrays, use_var="unequal", welch_correction=True)
k = len(arrays)
omega2 = (k - 1) * (welch.statistic - 1) / ((k - 1) * (welch.statistic - 1) + ns)
reg(f"Levene W={levene.statistic:.2f}; p={levene.pvalue:.3g}")
reg(f"Welch F({k - 1}, {welch.df_denom:.2f})={welch.statistic:.2f}; p={welch.pvalue:.3g}; "
    f"omega2 estimada={omega2:.2f}")
pairs = list(combinations(range(k), 2))
raw_p = [stats.ttest_ind(arrays[a], arrays[b], equal_var=False).pvalue for a, b in pairs]
posthoc = pd.DataFrame({"grupo_1": [short[a] for a, _ in pairs],
                        "grupo_2": [short[b] for _, b in pairs],
                        "p_bruto": raw_p, "p_holm": multipletests(raw_p, method="holm")[1]})
posthoc.to_csv(OUT / "comparaciones_holm.csv", index=False)
reg("Comparaciones Welch con ajuste de Holm:\n" + posthoc.round(4).to_string(index=False))

# 11. Sensibilidad: regresion del log-ingreso con errores HC3 ----------------
sample["log_earn"] = np.log(y)
sample["private_nonprofit"] = (sample["CONTROL_inst"] == 2).astype(int)
sens = sample.dropna(subset=["sat_avg", "UGDS", "REGION"]).copy()
model = smf.ols("log_earn ~ admit_rate + sat_avg + private_nonprofit + np.log1p(UGDS) + C(REGION)",
                data=sens).fit(cov_type="HC3", use_t=True)
ci_b = model.conf_int().loc["admit_rate"]
reg(f"Regresion n={len(sens)}; b_admit={model.params['admit_rate']:.3f}; "
    f"t={model.tvalues['admit_rate']:.2f}; p={model.pvalues['admit_rate']:.4f}; "
    f"IC95=[{ci_b[0]:.3f}, {ci_b[1]:.3f}]; R2aj={model.rsquared_adj:.3f}")

# 12. Figuras (sin titulo interno: el titulo APA va en el documento) ---------
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10})
AZUL, ROJO, GRIS, NARANJA = "#4C72B0", "#C44E52", "#333333", "#DD8452"
is_pr = sample["STABBR"].eq("PR").to_numpy()

# Figura 1. Histograma con media y mediana en colores contrastantes
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.hist(y / 1000, bins=18, color=AZUL, alpha=.85, edgecolor="white", linewidth=.8)
ax.axvline(y.mean() / 1000, color=ROJO, linewidth=2, label=f"Media = ${y.mean() / 1000:,.1f}k")
ax.axvline(y.median() / 1000, color=GRIS, linestyle="--", linewidth=2,
           label=f"Mediana = ${y.median() / 1000:,.1f}k")
ax.set(xlabel="Ingreso mediano a cuatro años (miles de USD)", ylabel="Número de instituciones")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "figura1_histograma.png", dpi=220); plt.close(fig)

# Figura 3. Caja + puntos individuales + media de cada grupo (el tablero reutiliza su estilo)
fig, ax = plt.subplots(figsize=(7.8, 4.8))
data_k = [a / 1000 for a in arrays]
box_style = dict(showfliers=False, patch_artist=True,
                 boxprops=dict(facecolor="#DCE6F2", edgecolor=GRIS),
                 medianprops=dict(color=GRIS, linewidth=1.6),
                 whiskerprops=dict(color=GRIS), capprops=dict(color=GRIS))
ax.boxplot(data_k, widths=.5, **box_style)
jit = np.random.default_rng(SEED)
for i, g in enumerate(labels, start=1):
    sub = sample[sample["selectivity_group"] == g]
    xs = i + jit.uniform(-.17, .17, len(sub))
    vals = sub["EARN_MDN_4YR"].to_numpy() / 1000
    pr = sub["STABBR"].eq("PR").to_numpy()
    ax.scatter(xs[~pr], vals[~pr], s=12, color=AZUL, alpha=.55, zorder=3)
    ax.scatter(xs[pr], vals[pr], s=42, marker="^", color=NARANJA, edgecolor="black",
               linewidth=.5, zorder=4)
    ax.scatter(i, vals.mean(), marker="D", s=48, color=ROJO, edgecolor="black",
               linewidth=.5, zorder=5)
ax.set_xticks(range(1, 5), [f"{s}\n(n = {len(a)})" for s, a in zip(short, arrays)])
ax.set(xlabel="Tasa de admisión de la institución (grupo de selectividad)",
       ylabel="Ingreso mediano a cuatro años (miles de USD)")
ax.scatter([], [], marker="D", color=ROJO, edgecolor="black", label="Media del grupo")
ax.scatter([], [], marker="^", color=NARANJA, edgecolor="black", label="Institución de Puerto Rico")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "figura3_caja_puntos.png", dpi=220); plt.close(fig)

# Figura 2. Dispersion con curva LOWESS y banda bootstrap del 95 %
xg = np.linspace(x_a.min(), x_a.max(), 120)
fit = lowess(y_a / 1000, x_a, frac=.5, xvals=xg)
rng_l = np.random.default_rng(SEED)
bands = []
for _ in range(500):
    idx = rng_l.integers(0, ns, ns)
    bands.append(lowess(y_a[idx] / 1000, x_a[idx], frac=.5, xvals=xg))
lo, hi = np.nanpercentile(np.array(bands), [2.5, 97.5], axis=0)
fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.fill_between(xg * 100, lo, hi, color=ROJO, alpha=.15, linewidth=0, label="Banda bootstrap 95 %")
ax.scatter(x_a[~is_pr] * 100, y_a[~is_pr] / 1000, s=20, color=AZUL, alpha=.6, label="Institución")
ax.scatter(x_a[is_pr] * 100, y_a[is_pr] / 1000, s=46, marker="^", color=NARANJA,
           edgecolor="black", linewidth=.5, label="Institución de Puerto Rico")
ax.plot(xg * 100, fit, color=ROJO, linewidth=2, label="Tendencia LOWESS")
ax.set(xlabel="Tasa de admisión (%)", ylabel="Ingreso mediano a cuatro años (miles de USD)")
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "figura2_dispersion.png", dpi=220); plt.close(fig)

# Tablero de una pagina para el informe publicado (Apendice B)
fig = plt.figure(figsize=(13, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[.55, 2.2], hspace=.45, wspace=.32)
kpis = [(f"{ns}", "Instituciones en la muestra"),
        (f"${y.median():,.0f}", "Ingreso mediano a 4 años"),
        (f"ρs = {rho:.2f}", f"IC 95 % [{ci_s[0]:.2f}, {ci_s[1]:.2f}]")]
for j, (big, small) in enumerate(kpis):
    a = fig.add_subplot(gs[0, j]); a.axis("off")
    a.text(.5, .62, big, ha="center", va="center", fontsize=26, fontweight="bold", color=GRIS)
    a.text(.5, .12, small, ha="center", va="center", fontsize=11, color=GRIS)
a1 = fig.add_subplot(gs[1, 0])
a1.hist(y / 1000, bins=18, color=AZUL, alpha=.85, edgecolor="white")
a1.axvline(y.mean() / 1000, color=ROJO, lw=2, label="Media")
a1.axvline(y.median() / 1000, color=GRIS, ls="--", lw=2, label="Mediana")
a1.set(title="Distribución del ingreso", xlabel="Miles de USD", ylabel="Instituciones")
a1.legend(frameon=False, fontsize=9)
a2 = fig.add_subplot(gs[1, 1])
a2.boxplot(data_k, **box_style)
for i, a in enumerate(data_k, start=1):
    a2.scatter(i, a.mean(), marker="D", color=ROJO, edgecolor="black", zorder=3)
a2.set_xticks(range(1, 5), short, fontsize=9)
a2.set(title="Ingreso por selectividad", xlabel="Tasa de admisión", ylabel="Miles de USD")
a3 = fig.add_subplot(gs[1, 2])
a3.fill_between(xg * 100, lo, hi, color=ROJO, alpha=.15, linewidth=0)
a3.scatter(x_a * 100, y_a / 1000, s=12, color=AZUL, alpha=.55)
a3.plot(xg * 100, fit, color=ROJO, lw=2)
a3.set(title="Admisión vs. ingreso", xlabel="Tasa de admisión (%)", ylabel="Miles de USD")
for a in (a1, a2, a3):
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Selectividad institucional e ingresos de graduados de Computer Science",
             fontsize=15, fontweight="bold", y=.98)
fig.text(.5, .015, f"{FUENTE} Muestra aleatoria estratificada, n = {ns}. "
         f"Welch F(3, {welch.df_denom:.1f}) = {welch.statistic:.1f}, ω² ≈ {omega2:.2f}. "
         "Asociación observacional; no implica causalidad.", ha="center", fontsize=9, color=GRIS)
fig.savefig(OUT / "tablero_informe.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# 13. Exportacion de la muestra y del resumen ---------------------------------
keep = ["OPEID6", "INSTNM_fos", "STABBR", "CONTROL_fos", "REGION", "UGDS", "admit_rate",
        "sat_avg", "EARN_COUNT_WNE_4YR", "EARN_MDN_4YR", "selectivity_group"]
sample[keep].to_csv(OUT / "muestra_analitica.csv", index=False)
(OUT / "resumen_resultados.txt").write_text("\n".join(log), encoding="utf-8")
