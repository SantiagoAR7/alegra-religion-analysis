"""
Análisis completo — Pew Research Center 2025
Religión y Espiritualidad: ¿Cómo se compara EE.UU. con otros países?
Todo en español. Técnicas: EDA, KMeans, PCA, Storytelling with Data.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Paleta ────────────────────────────────────────────────────────────────────
PRIMARIO  = "#1B4FD8"
ACENTO    = "#F59E0B"
PELIGRO   = "#DC2626"
EXITO     = "#059669"
NEUTRO    = "#94A3B8"
BG        = "#F8FAFC"
DARK      = "#0F172A"
CARD      = "#FFFFFF"
AZUL_CLARO= "#DBEAFE"

COLOR_REGION = {
    "África":         "#DC2626",
    "Medio Oriente":  "#F59E0B",
    "Asia-Pacífico":  "#7C3AED",
    "Américas":       "#059669",
    "Europa":         "#1B4FD8",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    BG,
    "figure.facecolor":  BG,
    "grid.color":        "#E2E8F0",
    "grid.linewidth":    0.7,
    "axes.labelcolor":   DARK,
    "xtick.color":       DARK,
    "ytick.color":       DARK,
    "axes.titlepad":     12,
})

# ── Cargar datos ───────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/alegra_v2/data/religion_pew2025.csv")
df_clean = df.dropna(subset=["Creen_vida_después_muerte_pct",
                              "Oran_diariamente_pct"]).copy()  # excluye Túnez

# ── ML: Clustering ─────────────────────────────────────────────────────────────
features_ml = ["Afiliación_religiosa_pct", "Creen_vida_después_muerte_pct",
                "Creen_espíritus_naturaleza_pct", "Oran_diariamente_pct"]
X = df_clean[features_ml].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

best_k, best_s = 3, -1
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    s = silhouette_score(X_sc, km.fit_predict(X_sc))
    if s > best_s:
        best_k, best_s = k, s

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean = df_clean.copy()
df_clean["Cluster"] = km_final.fit_predict(X_sc)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_sc)
df_clean["PCA1"] = X_pca[:, 0]
df_clean["PCA2"] = X_pca[:, 1]

means = df_clean.groupby("Cluster")["Afiliación_religiosa_pct"].mean().sort_values()
nombres = ["Secular", "Moderadamente\nreligioso", "Muy religioso", "Altamente\nreligioso"]
cluster_etiq = {idx: nombres[i] for i, idx in enumerate(means.index)}
df_clean["Cluster_etiq"] = df_clean["Cluster"].map(cluster_etiq)

CLUSTER_COL = {
    "Secular":            "#1B4FD8",
    "Moderadamente\nreligioso": "#F59E0B",
    "Muy religioso":      "#DC2626",
    "Altamente\nreligioso":"#7B0000",
}

# ── Resumen regional ───────────────────────────────────────────────────────────
reg = df_clean.groupby("Región").agg(
    Afiliación        =("Afiliación_religiosa_pct", "mean"),
    VidaDespués       =("Creen_vida_después_muerte_pct", "mean"),
    EspíritusNatura   =("Creen_espíritus_naturaleza_pct", "mean"),
    OraciónDiaria     =("Oran_diariamente_pct", "mean"),
    Adivinación       =("Consultan_adivinación_pct", "mean"),
).round(1).reset_index()

# ── KEY INSIGHTS ───────────────────────────────────────────────────────────────
print("="*65)
print("HALLAZGOS CLAVE — Pew Research Center 2025")
print("="*65)
print(f"\n1. País más afiliado: {df.loc[df.Afiliación_religiosa_pct.idxmax(), 'País']} ({df.Afiliación_religiosa_pct.max()}%)")
print(f"2. País menos afiliado: {df.loc[df.Afiliación_religiosa_pct.idxmin(), 'País']} ({df.Afiliación_religiosa_pct.min()}%)")
print(f"3. Mayor oración diaria: {df_clean.loc[df_clean.Oran_diariamente_pct.idxmax(), 'País']} ({df_clean.Oran_diariamente_pct.max()}%)")
print(f"4. Menor oración diaria: {df_clean.loc[df_clean.Oran_diariamente_pct.idxmin(), 'País']} ({df_clean.Oran_diariamente_pct.min()}%)")
print(f"5. Mayor brecha afiliación vs oración: ", end="")
df_clean["brecha_orac"] = df_clean["Afiliación_religiosa_pct"] - df_clean["Oran_diariamente_pct"]
idx_brecha = df_clean["brecha_orac"].idxmax()
print(f"{df_clean.loc[idx_brecha,'País']} ({df_clean.loc[idx_brecha,'brecha_orac']:.0f}pp)")
print(f"6. Más creyentes en espíritus naturaleza: {df_clean.loc[df_clean.Creen_espíritus_naturaleza_pct.idxmax(), 'País']} ({df_clean.Creen_espíritus_naturaleza_pct.max()}%)")
print(f"7. EE.UU. vs promedio Europa — Afiliación: {df[df.País=='EE.UU.'].Afiliación_religiosa_pct.values[0]}% vs {reg[reg.Región=='Europa'].Afiliación.values[0]:.0f}%")
print(f"8. EE.UU. vs promedio Europa — Oración: {df_clean[df_clean.País=='EE.UU.'].Oran_diariamente_pct.values[0]}% vs {reg[reg.Región=='Europa'].OraciónDiaria.values[0]:.0f}%")
print(f"\n9. Clusters ML (k={best_k}, silhouette={best_s:.3f})")
print(df_clean.groupby("Cluster_etiq")[["País"]].agg(lambda x: ", ".join(x)).to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — DASHBOARD PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 14), facecolor=BG)
gs = GridSpec(2, 3, figure=fig,
              hspace=0.50, wspace=0.40,
              left=0.05, right=0.97, top=0.89, bottom=0.06)

# ── Encabezado ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.965,
         "Religión y Espiritualidad: ¿Dónde está el mundo en 2025?",
         ha="center", fontsize=27, fontweight="bold", color=DARK)
fig.text(0.5, 0.930,
         "EE.UU. practica más su fe que cualquier país rico comparable — pero aún muy lejos de África e Indonesia.\n"
         "Europa reza poco; América Latina cree mucho; y creer en espíritus de la naturaleza es global.",
         ha="center", fontsize=12.5, color="#475569", linespacing=1.6)
fig.text(0.5, 0.898,
         "Fuente: Pew Research Center, junio 2025 · 36 países · Spring 2024 Global Attitudes Survey",
         ha="center", fontsize=9, color=NEUTRO)

# ── A: Barras horizontales — Afiliación religiosa por país (top/bottom 10) ────
ax1 = fig.add_subplot(gs[0, 0])
df_sort = df.sort_values("Afiliación_religiosa_pct", ascending=True)
colores = [COLOR_REGION[r] for r in df_sort["Región"]]

bars = ax1.barh(df_sort["País"], df_sort["Afiliación_religiosa_pct"],
                color=colores, height=0.72, zorder=3)

# Resaltar EE.UU.
for bar, pais, val in zip(bars, df_sort["País"], df_sort["Afiliación_religiosa_pct"]):
    if pais == "EE.UU.":
        bar.set_edgecolor(ACENTO)
        bar.set_linewidth(2.5)
    ax1.text(val + 0.4, bar.get_y() + bar.get_height()/2,
             f"{val:.0f}%", va="center", fontsize=7, color=DARK)

ax1.axvline(50, color=DARK, linewidth=0.8, linestyle="--", alpha=0.35)
ax1.set_xlim(0, 112)
ax1.set_xticks([0, 25, 50, 75, 100])
ax1.set_xlabel("% adultos con afiliación religiosa", fontsize=9)
ax1.set_title("① Afiliación Religiosa por País", fontsize=11.5, fontweight="bold", color=DARK)
ax1.tick_params(axis='y', labelsize=6.8)
ax1.grid(axis='x', zorder=0)

leyenda = [mpatches.Patch(color=v, label=k) for k, v in COLOR_REGION.items()]
ax1.legend(handles=leyenda, fontsize=6.5, loc="lower right", framealpha=0.8, ncol=1)

# ── B: Scatter — Afiliación vs Oración diaria ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

for region, grp in df_clean.groupby("Región"):
    ax2.scatter(grp["Afiliación_religiosa_pct"], grp["Oran_diariamente_pct"],
                c=COLOR_REGION[region], s=65, alpha=0.82, zorder=4,
                edgecolors="white", linewidths=0.6, label=region)

# Línea de referencia (si todos los afiliados oran)
ax2.plot([0,100],[0,100], color=NEUTRO, lw=1, linestyle=":", alpha=0.5, zorder=2)
ax2.text(62, 74, "Si todos rezaran", fontsize=7, color=NEUTRO, rotation=38)

# Anotar países clave
destacados = {
    "EE.UU.": (1.2, 1.5),
    "Indonesia": (1.2, -5),
    "Suecia": (1.2, 1.5),
    "India": (-15, -6),
    "Japón": (1.5, 1.5),
    "Polonia": (1.5, -5),
}
for pais, (dx, dy) in destacados.items():
    row = df_clean[df_clean.País == pais]
    if len(row) == 0: continue
    r = row.iloc[0]
    col = PELIGRO if pais == "EE.UU." else DARK
    ax2.annotate(pais,
                 xy=(r.Afiliación_religiosa_pct, r.Oran_diariamente_pct),
                 xytext=(r.Afiliación_religiosa_pct + dx, r.Oran_diariamente_pct + dy),
                 fontsize=7.5, color=col, fontweight="bold" if pais=="EE.UU." else "normal",
                 arrowprops=dict(arrowstyle="-", color=NEUTRO, lw=0.7))

ax2.set_xlabel("% con afiliación religiosa", fontsize=9.5)
ax2.set_ylabel("% que ora al menos una vez al día", fontsize=9.5)
ax2.set_title("② Afiliación vs. Práctica (Oración Diaria)", fontsize=11.5, fontweight="bold", color=DARK)
ax2.set_xlim(38, 106); ax2.set_ylim(0, 102)
ax2.grid(zorder=0)

# Cuadrante EE.UU.
ax2.axvline(69, color=PELIGRO, lw=0.8, linestyle="--", alpha=0.4)
ax2.axhline(44, color=PELIGRO, lw=0.8, linestyle="--", alpha=0.4)
ax2.text(70, 1, "EE.UU.\n(69%, 44%)", fontsize=7, color=PELIGRO, alpha=0.7)

# ── C: Barras agrupadas — Promedio regional 4 variables ───────────────────────
ax3 = fig.add_subplot(gs[0, 2])

reg_ord = reg.sort_values("Afiliación", ascending=False)
x = np.arange(len(reg_ord))
w = 0.2

vars_plot = [("Afiliación", ACENTO, "Afiliación"),
             ("VidaDespués", PRIMARIO, "Vida después\nde muerte"),
             ("EspíritusNatura", EXITO, "Espíritus\nnaturaleza"),
             ("OraciónDiaria", PELIGRO, "Oración\ndiaria")]

for i, (col, color, label) in enumerate(vars_plot):
    offset = (i - 1.5) * w
    ax3.bar(x + offset, reg_ord[col], w, label=label, color=color,
            alpha=0.88, zorder=3)

ax3.set_xticks(x)
ax3.set_xticklabels(reg_ord["Región"], fontsize=7.5, rotation=15, ha="right")
ax3.set_ylabel("% de adultos", fontsize=9)
ax3.set_title("③ Perfil Espiritual por Región", fontsize=11.5, fontweight="bold", color=DARK)
ax3.legend(fontsize=7, framealpha=0.8, loc="upper right", ncol=2)
ax3.set_ylim(0, 105)
ax3.grid(axis='y', zorder=0)

# ── D: Lollipop — Brecha afiliación vs oración por país ───────────────────────
ax4 = fig.add_subplot(gs[1, 0])

df_brecha = df_clean.copy()
df_brecha["brecha"] = df_brecha["Afiliación_religiosa_pct"] - df_brecha["Oran_diariamente_pct"]
df_brecha = df_brecha.sort_values("brecha", ascending=True)

y = np.arange(len(df_brecha))
colores_b = [COLOR_REGION[r] for r in df_brecha["Región"]]

ax4.barh(y, df_brecha["brecha"], color=colores_b, height=0.6, alpha=0.8, zorder=3)
ax4.axvline(0, color=DARK, lw=1, alpha=0.5)
ax4.set_yticks(y)
ax4.set_yticklabels(df_brecha["País"], fontsize=6.8)
ax4.set_xlabel("Brecha (% afiliados − % que oran diariamente, pp)", fontsize=8.5)
ax4.set_title("④ Brecha: ¿Cuántos creen pero no rezan?", fontsize=11.5, fontweight="bold", color=DARK)
ax4.grid(axis='x', zorder=0)

# Anotar EE.UU.
idx_eeuu = df_brecha[df_brecha.País == "EE.UU."].index[0]
pos_eeuu = list(df_brecha.index).index(idx_eeuu)
brecha_eeuu = df_brecha.loc[idx_eeuu, "brecha"]
ax4.annotate("EE.UU.", xy=(brecha_eeuu, pos_eeuu),
             xytext=(brecha_eeuu + 3, pos_eeuu + 1.5),
             fontsize=7.5, color=PELIGRO, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=PELIGRO, lw=1))

# ── E: PCA Clusters ML ────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])

for etiq, grp in df_clean.groupby("Cluster_etiq"):
    col = CLUSTER_COL.get(etiq, PRIMARIO)
    ax5.scatter(grp["PCA1"], grp["PCA2"], c=col, s=65, alpha=0.85,
                label=etiq, edgecolors="white", linewidths=0.5, zorder=4)

for pais in ["Indonesia","Suecia","EE.UU.","India","Polonia","Nigeria","Japón"]:
    row = df_clean[df_clean.País == pais]
    if len(row) == 0: continue
    r = row.iloc[0]
    ax5.annotate(pais, xy=(r.PCA1, r.PCA2),
                 xytext=(r.PCA1 + 0.12, r.PCA2 + 0.12),
                 fontsize=7, color=DARK,
                 arrowprops=dict(arrowstyle="-", color=NEUTRO, lw=0.7))

ax5.set_xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.0f}% varianza)", fontsize=9)
ax5.set_ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.0f}% varianza)", fontsize=9)
ax5.set_title(f"⑤ Segmentación ML (KMeans k={best_k} · PCA)", fontsize=11.5, fontweight="bold", color=DARK)
ax5.legend(fontsize=7.5, framealpha=0.85, loc="best")
ax5.grid(zorder=0)
ax5.text(0.02, 0.04, f"Silhouette: {best_s:.3f}", transform=ax5.transAxes,
         fontsize=8, color=NEUTRO)

# ── F: Scorecard métricas clave ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
ax6.set_title("⑥ Métricas Clave", fontsize=11.5, fontweight="bold", color=DARK)
ax6.set_xlim(0, 1); ax6.set_ylim(0, 1)

metricas = [
    ("36",      "Países analizados",            PRIMARIO),
    ("100%",    "Afiliación: Bangladesh,\nIndonesia, Sri Lanka, Tailandia...", EXITO),
    ("44%",     "Japón: país con menor afiliación\nentre los religioso", ACENTO),
    ("95%",     "Indonesia: mayor tasa de\noración diaria", PELIGRO),
    ("8%",      "Suecia: la que menos ora al día\nen el mundo encuestado", NEUTRO),
    (f"k={best_k}", f"Grupos identificados por ML\n(Silhouette {best_s:.2f})", DARK),
]

for i, (val, label, color) in enumerate(metricas):
    y0 = 0.93 - i * 0.155
    ax6.add_patch(plt.Rectangle((0.01, y0 - 0.06), 0.97, 0.115,
                                 facecolor=AZUL_CLARO, alpha=0.45,
                                 transform=ax6.transAxes, clip_on=False))
    ax6.text(0.16, y0 + 0.005, val, fontsize=15, fontweight="bold",
             color=color, va="center", ha="center", transform=ax6.transAxes)
    ax6.text(0.60, y0 + 0.005, label, fontsize=8.2, color=DARK,
             va="center", ha="left", transform=ax6.transAxes, linespacing=1.35)

out1 = "/home/claude/alegra_v2/charts/slide1_dashboard.png"
plt.savefig(out1, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\n✅ Slide 1 guardado: {out1}")


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — EL INSIGHT PRINCIPAL: EE.UU. es el país rico más religioso
# ═══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 11), facecolor=BG)
gs2 = GridSpec(1, 2, figure=fig2,
               hspace=0.3, wspace=0.42,
               left=0.05, right=0.97, top=0.84, bottom=0.09)

fig2.text(0.5, 0.96,
          "EE.UU. es el país de altos ingresos más religioso y devoto del mundo",
          ha="center", fontsize=24, fontweight="bold", color=DARK)
fig2.text(0.5, 0.905,
          "Entre los países de ingresos altos, EE.UU. supera a todos en afiliación religiosa (69%) y oración diaria (44%).\n"
          "Esta brecha es aún más llamativa frente a países europeos con contextos culturales similares.",
          ha="center", fontsize=12.5, color="#475569", linespacing=1.6)
fig2.text(0.5, 0.863,
          "Fuente: Pew Research Center, junio 2025 · Spring 2024 Global Attitudes Survey",
          ha="center", fontsize=9, color=NEUTRO)

# ── Panel A: Países ricos — afiliación vs oración ────────────────────────────
ax_a = fig2.add_subplot(gs2[0, 0])

ricos = df_clean[df_clean.Nivel_ingreso == "Alto"].copy()
colores_ricos = [PELIGRO if p == "EE.UU." else NEUTRO for p in ricos["País"]]
sizes = [150 if p == "EE.UU." else 70 for p in ricos["País"]]

sc = ax_a.scatter(ricos["Afiliación_religiosa_pct"], ricos["Oran_diariamente_pct"],
                  c=colores_ricos, s=sizes, zorder=5, edgecolors="white", linewidths=1.2)

# Etiquetas
for _, row in ricos.iterrows():
    ax_a.annotate(row["País"],
                  xy=(row["Afiliación_religiosa_pct"], row["Oran_diariamente_pct"]),
                  xytext=(row["Afiliación_religiosa_pct"] + 0.5,
                          row["Oran_diariamente_pct"] + 1.2),
                  fontsize=8.5,
                  color=PELIGRO if row["País"] == "EE.UU." else DARK,
                  fontweight="bold" if row["País"] == "EE.UU." else "normal")

# Promedio países ricos
avg_afil = ricos["Afiliación_religiosa_pct"].mean()
avg_orac = ricos["Oran_diariamente_pct"].mean()
ax_a.axvline(avg_afil, color=NEUTRO, lw=1.2, linestyle="--", alpha=0.6)
ax_a.axhline(avg_orac, color=NEUTRO, lw=1.2, linestyle="--", alpha=0.6)
ax_a.text(avg_afil + 0.4, 1, f"Promedio países ricos\n{avg_afil:.0f}%", fontsize=7.5, color=NEUTRO)
ax_a.text(ricos["Afiliación_religiosa_pct"].min(), avg_orac + 0.5,
          f"{avg_orac:.0f}%", fontsize=7.5, color=NEUTRO)

ax_a.set_xlabel("% adultos con afiliación religiosa", fontsize=10.5)
ax_a.set_ylabel("% que ora al menos una vez al día", fontsize=10.5)
ax_a.set_title("Solo países de ingresos altos (World Bank 2024)", fontsize=12, fontweight="bold", color=DARK)
ax_a.set_xlim(38, 108)
ax_a.set_ylim(0, 52)
ax_a.grid(zorder=0)

# Etiqueta destacada EE.UU.
eeuu_r = ricos[ricos.País == "EE.UU."].iloc[0]
ax_a.annotate("",
              xy=(eeuu_r.Afiliación_religiosa_pct, eeuu_r.Oran_diariamente_pct),
              xytext=(eeuu_r.Afiliación_religiosa_pct - 10, eeuu_r.Oran_diariamente_pct + 6),
              arrowprops=dict(arrowstyle="->", color=PELIGRO, lw=1.5))
ax_a.text(eeuu_r.Afiliación_religiosa_pct - 10.5,
          eeuu_r.Oran_diariamente_pct + 7,
          "EE.UU. supera a todos\nlos países ricos en ambos\nindicadores", fontsize=8.5,
          color=PELIGRO, ha="center", fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PELIGRO, alpha=0.9))

# ── Panel B: Creer en espíritus naturaleza — sorpresa global ─────────────────
ax_b = fig2.add_subplot(gs2[0, 1])

df_esp = df_clean.sort_values("Creen_espíritus_naturaleza_pct", ascending=True)
colores_esp = [PELIGRO if p == "EE.UU." else COLOR_REGION[r]
               for p, r in zip(df_esp["País"], df_esp["Región"])]
bars_b = ax_b.barh(df_esp["País"], df_esp["Creen_espíritus_naturaleza_pct"],
                   color=colores_esp, height=0.72, alpha=0.85, zorder=3)

for bar, val in zip(bars_b, df_esp["Creen_espíritus_naturaleza_pct"]):
    ax_b.text(val + 0.5, bar.get_y() + bar.get_height()/2,
              f"{val:.0f}%", va="center", fontsize=7, color=DARK)

ax_b.axvline(48, color=PELIGRO, lw=1.5, linestyle="--", alpha=0.7)
ax_b.text(48.5, 1, "EE.UU.\n48%", fontsize=8, color=PELIGRO, fontweight="bold")

ax_b.set_xlim(0, 100)
ax_b.set_xlabel("% adultos que creen que partes de la naturaleza\n(montañas, ríos, árboles) tienen espíritus", fontsize=9.5)
ax_b.set_title("La espiritualidad de la naturaleza es sorprendentemente global", fontsize=12, fontweight="bold", color=DARK)
ax_b.tick_params(axis='y', labelsize=7)
ax_b.grid(axis='x', zorder=0)

leg2 = [mpatches.Patch(color=v, label=k) for k, v in COLOR_REGION.items()]
leg2.append(mpatches.Patch(color=PELIGRO, label="EE.UU."))
ax_b.legend(handles=leg2, fontsize=6.5, loc="lower right", framealpha=0.8)

out2 = "/home/claude/alegra_v2/charts/slide2_insight.png"
plt.savefig(out2, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✅ Slide 2 guardado: {out2}")
print("\n🎉 Análisis completo.")
