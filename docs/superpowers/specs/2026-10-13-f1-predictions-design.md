# F1 Predictions : probabilités de titre mondial — Design

Date : 2026-10-13
Statut : validé en conversation, en attente de relecture de la spec

## 1. Objectif

Un site public qui affiche, pour chaque pilote, la probabilité de remporter le championnat du monde de F1 de la saison en cours, mise à jour automatiquement après chaque Grand Prix.

**Public :** à la fois l'auteur (suivi perso de la saison), le grand public, et un portfolio (qualité du code et de la méthode visibles).

**Critères de succès :**
- Après chaque GP, le site est à jour sans intervention manuelle (au plus tard le mardi 8 h UTC).
- Les probabilités viennent d'un modèle ML entraîné sur l'historique, puis d'une simulation Monte Carlo de la fin de saison.
- La page « méthode » affiche une mesure honnête de la précision (backtest sur la saison en cours).
- Hébergement et exécution 100 % gratuits (GitHub Pages + GitHub Actions).
- Fonctionne tel quel pour la saison 2027 et suivantes.

## 2. Périmètre v1

**Inclus :**
1. Probabilités du titre pilotes (barres horizontales, triées).
2. Évolution des probabilités au fil de la saison (courbe, un point par GP couru).
3. Page « méthode » : modèle, features, pondération, précision du backtest, limites.

**Exclus de la v1 (données produites mais pas affichées, ajout ultérieur peu coûteux) :**
- Titre constructeurs.
- Prédiction détaillée du prochain GP (victoire / podium).
- Scénarios mathématiques (« X est champion si… »).

## 3. Stack

- Python 3.12, dépendances gérées par `uv` (`pyproject.toml`).
- pandas, numpy, LightGBM, requests, Jinja2, Plotly (graphiques intégrés au HTML statique).
- Tests : pytest ; lint : ruff.
- Données : API Jolpica (successeur d'Ergast). Faits vérifiés dans `docs/research/2026-10-13-f1-data-and-rules.md`.
- Hébergement : GitHub Pages ; automatisation : GitHub Actions ; repo créé via la CLI `gh`.

## 4. Architecture

Un package Python unique, `f1pred/`. Chaque module a une responsabilité et échange des DataFrames pandas ou des dicts JSON-sérialisables.

```
f1pred/
  rules.py      barème de points (GP et sprint), règles de départage, constantes de saison
  data.py       récupération Jolpica + cache parquet dans data/raw/
  features.py   une ligne par (course, pilote), sans fuite d'information
  model.py      entraînement LightGBM (position + abandon), prédiction, backtest
  simulate.py   Monte Carlo vectorisé sur les courses restantes
  history.py    lecture / écriture de data/history.json (probabilités après chaque GP)
  site.py       rendu Jinja2 → site/index.html, site/methode.html
  cli.py        point d'entrée : `python -m f1pred run [--offline] [--seed N]`
templates/      gabarits Jinja2
data/raw/       résultats bruts en parquet (committés)
data/history.json
site/           sortie générée (non committée, publiée par Actions)
tests/
  fixtures/     réponses Jolpica réelles sauvegardées, petits jeux de données
```

**Flux :** `data` → `features` → `model` → `simulate` → `history` → `site`.

### 4.1 `rules.py`
Les règles sportives sont des données, pas du code dispersé : barème GP, barème sprint, liste des week-ends sprint de la saison (issue du calendrier Jolpica), ordre de départage (nombre de victoires, puis de 2es places, etc.). Valeurs 2026 tirées du document de recherche.

### 4.2 `data.py`
- Endpoints : calendrier, résultats de course, résultats sprint, classement pilotes.
- Historique depuis 2014 (ère hybride), stocké en parquet par saison dans `data/raw/`.
- Mode incrémental : seules les manches absentes du cache sont téléchargées.
- Retry : 3 tentatives, backoff exponentiel ; respect des limites de débit de l'API.
- Détection de résultats incomplets : manche ignorée si le nombre de classés est anormalement bas ou si la manche n'est pas encore publiée.
- `--offline` : aucune requête, lecture du cache uniquement.

### 4.3 `features.py`
Une ligne par (course, pilote). Chaque feature d'une course R n'utilise **que** des courses strictement antérieures à R.
- Forme pilote : place moyenne sur les 3 et 5 dernières courses.
- Forme écurie : place moyenne des deux pilotes sur les 5 dernières courses.
- Historique du pilote sur le circuit (place moyenne, valeur neutre si absent).
- Écart moyen avec le coéquipier.
- Taux d'abandon récent (10 dernières courses).
- Part des points du pilote dans la saison en cours.
- **Pas de position de grille** (inconnue pour les courses futures).

### 4.4 `model.py`
- **Régresseur LightGBM** : cible = position d'arrivée (abandons exclus de l'entraînement de ce modèle).
- **Classifieur LightGBM** : cible = abandon (oui/non).
- **Pondération** : poids par échantillon ; la saison en cours pèse environ ×5 par rapport aux précédentes, avec décroissance exponentielle par ancienneté. Le facteur exact est choisi par backtest. Justification : le règlement 2026 (nouveaux moteurs et châssis) rend l'historique ancien moins prédictif.
- **Résidus** : écart-type des erreurs du régresseur sur validation, utilisé comme amplitude du bruit en simulation.
- Réentraînement complet à chaque exécution (moins d'une minute).
- **Backtest** : pour chaque manche déjà courue de la saison, réentraîner avec les données antérieures, prédire, et calculer le score de Brier sur le vainqueur et le podium. Exposé sur la page « méthode ».

### 4.5 `simulate.py`
- N = 10 000 simulations, numpy vectorisé, seed configurable.
- Pour chaque course restante (sprints compris, aux dates du calendrier) :
  1. performance = position prédite + bruit gaussien (σ = résidus du modèle) ;
  2. abandons tirés selon la probabilité du classifieur ; un pilote qui abandonne est relégué en fin de classement, sans points ;
  3. classement, puis attribution des points selon `rules.py`.
- Points simulés ajoutés au classement réel actuel.
- Champion = maximum de points ; égalité départagée selon `rules.py`.
- Sorties par pilote : P(titre), points finaux attendus.
- Titre mathématiquement acquis : la simulation donne 100 %, et un drapeau `decided=True` déclenche le bandeau « Champion du monde ».

### 4.6 `history.py`
- `data/history.json` : liste ordonnée de `{round, race_name, date, probabilities: {driver_id: p}}`.
- Remplissage initial : pour chaque manche courue de la saison, recalcul avec les données disponibles à ce moment-là (réutilise le backtest).
- Idempotent : relancer sur une manche déjà présente ne crée pas de doublon.

### 4.7 `site.py`
- `index.html` : barres de probabilités, courbe d'évolution, date de dernière mise à jour, bandeau champion si décidé.
- `methode.html` : explication du modèle, des features et de la pondération, scores du backtest, limites.
- Graphiques Plotly embarqués, lisibles en mode clair et sombre ; mise en page responsive (mobile).

## 5. Automatisation

### `.github/workflows/update.yml`
- Déclencheurs : cron lundi et mardi 08:00 UTC ; `workflow_dispatch`.
- Étapes : checkout → Python 3.12 + uv → `pytest` → `python -m f1pred run` → commit de `data/` si modifié → upload de `site/` → déploiement Pages.
- Pas de nouvelle manche depuis la dernière exécution : arrêt propre, sans commit ni déploiement.

### `.github/workflows/test.yml`
- À chaque push : `ruff check` + `pytest`.

## 6. Gestion d'erreurs

| Cas | Réaction |
|---|---|
| API en panne / timeout | 3 tentatives avec backoff ; ensuite échec du job, le site garde la version précédente, notification GitHub par e-mail. |
| Pas de nouvelle manche | Sortie propre (code 0), rien publié. |
| Résultats incomplets | Manche ignorée, reprise au passage suivant (mardi). |
| Sorties incohérentes (somme ≠ 1 à ±0,001, NaN, pilote du classement absent) | Échec **avant** déploiement. |
| Titre mathématiquement décidé | Bandeau « Champion du monde ». |

## 7. Tests

| Module | Tests |
|---|---|
| `data` | Parsing de réponses Jolpica réelles (fixtures) ; retry avec HTTP simulé ; détection de manche incomplète. Aucun appel réseau. |
| `features` | Anti-fuite : valeurs absurdes dans les courses futures sans effet sur les features de R ; moyennes glissantes sur jeu fait main. |
| `model` | Entraînement sur petit jeu : positions prédites dans [1, 20], probabilités d'abandon dans [0, 1] ; backtest renvoie un score fini. |
| `simulate` | Leader inrattrapable → 100 % ; deux pilotes identiques à égalité → ~50/50 (±3 points) ; barème appliqué ; départage ; somme = 1 ; reproductible à seed égale. |
| `history` | Ajout idempotent ; ordre des manches. |
| `site` | Chaque pilote présent dans le HTML ; deux pages générées. |
| Bout en bout | `run --offline` sur fixtures produit un site complet. |

## 8. Limites connues

- Le règlement 2026 réduit la valeur de l'historique ; la précision en début de saison est faible.
- Le modèle ignore les informations du week-end en cours (qualifs, météo, pénalités moteur).
- Peu de manches restent en octobre 2026 : la v1 sert surtout à valider le pipeline avant 2027.
