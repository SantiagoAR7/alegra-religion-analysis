#!/bin/bash
# ═══════════════════════════════════════════════════════
#  ALEGRA BI CHALLENGE — Subir a GitHub en 3 pasos
#  Ejecutar dentro de la carpeta del proyecto
# ═══════════════════════════════════════════════════════

echo ""
echo "🚀 Alegra BI — Setup GitHub"
echo "═══════════════════════════"

git init
git add .
git commit -m "feat: Alegra BI Challenge — Religión y Espiritualidad 2025

Dataset real: Pew Research Center, junio 2025
- 36 países, Spring 2024 Global Attitudes Survey
- EDA + KMeans Clustering + PCA
- Insight: EE.UU. es el país rico más religioso del mundo
- Todo en español, con GitHub Pages incluido"

echo ""
echo "✅ Repositorio local creado."
echo ""
echo "📋 PRÓXIMOS PASOS:"
echo "──────────────────"
echo ""
echo "1. Ir a https://github.com/new"
echo "   → Nombre: alegra-religion-analysis"
echo "   → Visibilidad: Public"
echo "   → NO marcar 'Initialize with README'"
echo ""
echo "2. Conectar y subir:"
echo "   git remote add origin https://github.com/TU_USUARIO/alegra-religion-analysis.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Activar GitHub Pages:"
echo "   → Settings → Pages → Source: Deploy from branch → main / root → Save"
echo ""
echo "4. Tu reporte quedará en:"
echo "   → https://TU_USUARIO.github.io/alegra-religion-analysis/"
echo ""
echo "   (Reemplaza TU_USUARIO con tu username de GitHub)"
echo "═══════════════════════════════════════════════════"
