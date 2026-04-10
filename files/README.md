# 🌍 Religión y Espiritualidad en el Mundo — Alegra BI Challenge

> **Reto Makeover Monday · Análisis Explicativo**  
> Técnicas: EDA · KMeans Clustering · PCA · Storytelling with Data  
> Datos reales: Pew Research Center, junio 2025

---

## 🔍 Insight Principal

> **EE.UU. es el país de ingresos altos más religioso y devoto del mundo.**  
> Con un 69% de afiliación religiosa y 44% de oración diaria, supera a todos sus pares económicos.  
> El promedio europeo de oración es apenas **21%** — la mitad que EE.UU.

---

## 📊 Visualizaciones

### Slide 1 — Dashboard Principal
![Slide 1](charts/slide1_dashboard.png)

### Slide 2 — EE.UU. vs el Mundo
![Slide 2](charts/slide2_insight.png)

---

## 🧠 Hallazgos Clave

| Métrica | Valor |
|---|---|
| Países analizados | 36 |
| Mayor afiliación religiosa | Bangladesh, Indonesia, Sri Lanka, Tailandia (100%) |
| Menor afiliación religiosa | Japón (44%) |
| Mayor oración diaria | Indonesia (95%) |
| Menor oración diaria | Suecia (8%) |
| Mayor brecha creer vs rezar | Polonia (95% afiliados, solo 18% ora diariamente = 77pp) |
| EE.UU. oración vs Europa promedio | 44% vs 21% |
| Clusters ML identificados | 2 (Silhouette: 0.434) |

---

## 🤖 Metodología

```
data/
├── build_dataset.py      # Dataset basado en tabla oficial Pew Research Center 2025
└── religion_pew2025.csv  # 36 países · 8 variables

analysis.py               # EDA + KMeans + PCA + generación de gráficos
charts/
├── slide1_dashboard.png  # Dashboard 6-paneles
└── slide2_insight.png    # Deep dive: EE.UU. vs países ricos + espíritus naturaleza
index.html                # Reporte interactivo (GitHub Pages)
```

### Técnicas aplicadas
- **EDA**: distribuciones, correlaciones, perfiles regionales, brecha creencia vs práctica
- **KMeans Clustering**: segmentación de países por perfil espiritual
- **PCA**: reducción a 2 componentes para visualizar los clusters
- **Silhouette Score**: selección automática del k óptimo
- **Storytelling with Data**: título que afirma, foco visual, decluttering, narrativa clara

---

## 🚀 Cómo reproducir

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/alegra-religion-analysis.git
cd alegra-religion-analysis

# 2. Instalar dependencias
pip install pandas numpy matplotlib scikit-learn

# 3. Construir el dataset
python3 data/build_dataset.py

# 4. Correr el análisis completo
python3 analysis.py
```

---

## 📋 Fuente de Datos

**Pew Research Center** — *"Spirituality and Religion: How Does the U.S. Compare With Other Countries?"*  
Publicado: 25 de junio de 2025  
Encuesta: Spring 2024 Global Attitudes Survey  
36 países · ~4 mil millones de personas representadas  
🔗 https://www.pewresearch.org/religion/2025/06/25/spirituality-and-religion-us-comparison-to-other-countries/

---

## 👤 Autor

Análisis desarrollado como parte del **Alegra BI Challenge** — Reto Makeover Monday  
Principios aplicados: *Storytelling with Data* (Cole Nussbaumer Knaflic)
