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

- `.github/workflows/update.yml` : lundi et mardi 8 h UTC (et à la demande). Le job `build`
  met à jour et commite `data/`, puis prépare `site/` ; le job `deploy` le publie sur GitHub
  Pages, seulement s'il y a eu du nouveau.
- Hors saison, si le dernier commit date de plus de 45 jours, `build` pousse un commit vide
  `chore: keepalive` pour que GitHub ne désactive pas le cron.
- `.github/workflows/test.yml` : ruff + pytest à chaque push.

## Si le déploiement a échoué

Il faut une fois activer GitHub Pages : Settings → Pages → Source = « GitHub Actions ».

Si le job `deploy` échoue alors que `build` a déjà commité les données, le prochain lancement
planifié ne fait rien : la manche et l'empreinte des résultats figurent déjà dans
`data/history.json` et `data/state.json`. Dans la journée, « Re-run failed jobs » sur
l'exécution suffit (l'artefact Pages n'est gardé qu'un jour). Sinon, forcer un nouveau calcul
et un nouveau déploiement en relançant le workflow `update` à la main avec `force` coché :

```bash
gh workflow run update -f force=true
```

Données : [Jolpica F1 API](https://github.com/jolpica/jolpica-f1). Projet non officiel.
