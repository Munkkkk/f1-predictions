# Prédictions F1

Probabilité de chaque pilote de remporter le championnat du monde de F1, mise à jour
automatiquement après chaque Grand Prix.

**Méthode :** deux modèles LightGBM (position d'arrivée, abandon) entraînés depuis 2014,
puis 10 000 simulations Monte Carlo de la fin de saison. Détails sur la page « Méthode » du site.

## Lancer en local

```bash
uv sync
uv run python -m f1pred run            # télécharge les nouvelles données et génère site/
uv run python -m f1pred run --offline  # sans réseau, à partir de data/raw/
uv run pytest
```

## Automatisation

- `.github/workflows/update.yml` : lundi et mardi 8 h UTC (et à la demande), met à jour
  `data/`, puis déploie `site/` sur GitHub Pages.
- `.github/workflows/test.yml` : ruff + pytest à chaque push.

Données : [Jolpica F1 API](https://github.com/jolpica/jolpica-f1). Projet non officiel.
