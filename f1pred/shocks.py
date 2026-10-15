"""Taille des chocs persistants, mesurée sur les erreurs de prévision à plusieurs courses.

À chaque checkpoint c, le modèle prévoit les courses c+1..c+HORIZON (comme pour la
simulation). On compare ces prévisions aux résultats réels, puis on décompose l'erreur :

    résidu = forme de l'écurie (par checkpoint) + forme du pilote (par checkpoint) + aléa

par la méthode des moments sur des groupes emboîtés (checkpoint, écurie, pilote).
"""

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred.chrono import rounds_after

HORIZON = 6  # courses prévues à partir de chaque checkpoint
MIN_ROWS = 3  # courses minimales par (checkpoint, pilote) pour entrer dans l'estimation
SIGMA_FLOOR = 0.5  # plancher de l'aléa par course, pour qu'il reste strictement positif
MAX_DNF_SHOCK = 1.0
RECORD_COLUMNS = ["season", "checkpoint", "round", "driver_id", "team", "resid", "finished",
                  "p_dnf"]


@dataclass(frozen=True)
class Shocks:
    sigma: float  # écart-type de l'aléa indépendant d'une course à l'autre (places)
    tau_driver: float = 0.0  # forme persistante du pilote, en écart à son écurie (places)
    tau_team: float = 0.0  # forme persistante de l'écurie (places)
    dnf_shock: float = 0.0  # log-écart-type du multiplicateur de fiabilité


def residual_records(preds: pd.DataFrame, results: pd.DataFrame, season: int,
                     checkpoint: int, horizon: int = HORIZON) -> pd.DataFrame:
    """Associe les prévisions faites après `checkpoint` aux `horizon` GP courus ensuite."""
    targets = rounds_after(results, season, checkpoint, horizon)
    gp = results[(results["season"] == season) & ~results["is_sprint"].astype(bool)
                 & results["round"].isin(targets)]
    rec = preds[["round", "driver_id", "pred_pos", "p_dnf"]].merge(
        gp[["round", "driver_id", "constructor_id", "position", "finished"]],
        on=["round", "driver_id"], how="inner")
    rec = rec.assign(season=season, checkpoint=checkpoint, team=rec["constructor_id"],
                     resid=rec["position"].astype(float) - rec["pred_pos"],
                     finished=rec["finished"].astype(bool))
    return rec[RECORD_COLUMNS].reset_index(drop=True)


def usable_records(records: pd.DataFrame, season: int, known: Collection[int]) -> pd.DataFrame:
    """Paires (checkpoint, course) dont la course de `season` est déjà courue (`known`)."""
    return records[(records["season"] < season) | records["round"].isin(list(known))]


def _form_shocks(fin: pd.DataFrame) -> tuple[float, float]:
    """(tau_driver, tau_team) par moments emboîtés ; 0 si les données manquent."""
    drv = (fin.groupby(["season", "checkpoint", "team", "driver_id"])["resid"]
           .agg(["mean", "var", "size"]).reset_index())
    drv = drv[drv["size"] >= MIN_ROWS]
    if len(drv) < 2:
        return 0.0, 0.0
    # Aléa intra-groupe : variance des résidus autour de la moyenne de leur groupe.
    sigma_e2 = float(((drv["size"] - 1) * drv["var"]).sum() / (drv["size"] - 1).sum())
    drv["noise"] = sigma_e2 / drv["size"]
    team = drv.groupby(["season", "checkpoint", "team"])
    drv["k"] = team["mean"].transform("size")
    drv["team_mean"] = team["mean"].transform("mean")

    # Pilote : écart à la moyenne de son écurie, corrigé de k/(k-1), moins le bruit.
    pairs = drv[drv["k"] >= 2]
    tau_d2 = 0.0
    if len(pairs):
        dev2 = (pairs["mean"] - pairs["team_mean"]) ** 2 * pairs["k"] / (pairs["k"] - 1)
        tau_d2 = max(0.0, float(dev2.mean() - pairs["noise"].mean()))

    # Écurie : dispersion des moyennes d'écurie, moins la part pilote + bruit qu'elles portent.
    teams = drv.assign(part=(tau_d2 + drv["noise"]) / drv["k"] ** 2).groupby(
        ["season", "checkpoint", "team"]).agg(m=("team_mean", "first"), part=("part", "sum"))
    tau_t2 = 0.0
    if len(teams) >= 2:
        tau_t2 = max(0.0, float(teams["m"].var(ddof=1) - teams["part"].mean()))
    return float(np.sqrt(tau_d2)), float(np.sqrt(tau_t2))


def _dnf_shock(records: pd.DataFrame) -> float:
    """Surdispersion des abandons observés par rapport aux probabilités du classifieur.

    Avec un multiplicateur m de moyenne 1 et de variance cv², par groupe :
    E[(O - E)²] = (E - Σp²) + cv² (E² - Σp²). On en tire cv², puis s = sqrt(log(1 + cv²)).
    """
    g = (records.assign(dnf=(~records["finished"]).astype(float), p2=records["p_dnf"] ** 2)
         .groupby(["season", "checkpoint", "driver_id"])
         .agg(O=("dnf", "sum"), E=("p_dnf", "sum"), S2=("p2", "sum"), n=("dnf", "size")))
    g = g[g["n"] >= MIN_ROWS]
    denom = float((g["E"] ** 2 - g["S2"]).sum())
    if g.empty or denom <= 0.0:
        return 0.0
    excess = float(((g["O"] - g["E"]) ** 2 - (g["E"] - g["S2"])).sum())
    cv2 = max(0.0, excess) / denom
    return float(min(np.sqrt(np.log1p(cv2)), MAX_DNF_SHOCK))


def estimate_shocks(records: pd.DataFrame, fallback_sigma: float) -> Shocks:
    """Chocs persistants et aléa par course ; sans données, aucun choc et `fallback_sigma`."""
    fin = records[records["finished"].astype(bool)]
    if len(fin) < 2:
        return Shocks(sigma=fallback_sigma)
    tau_d, tau_t = _form_shocks(fin)
    total2 = float(fin["resid"].var(ddof=1))
    # L'aléa par course complète la variance totale des résidus.
    sigma = float(np.sqrt(max(total2 - tau_d**2 - tau_t**2, SIGMA_FLOOR**2)))
    return Shocks(sigma=sigma, tau_driver=tau_d, tau_team=tau_t, dnf_shock=_dnf_shock(records))
