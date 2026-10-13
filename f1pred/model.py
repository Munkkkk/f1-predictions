"""Deux modèles LightGBM : position d'arrivée (régression) et abandon (classification)."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from f1pred.features import FEATURE_COLUMNS

PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
              random_state=0, verbose=-1)
HOLDOUT_RACES = 10
DEFAULT_SIGMA = 3.0
MIN_GROUP_ROWS = 3  # lignes minimales par pilote / écurie pour estimer un choc
MIN_DNF_DRIVERS = 5
MAX_DNF_SHOCK = 1.0


@dataclass
class Models:
    reg: lgb.LGBMRegressor
    clf: lgb.LGBMClassifier | None
    dnf_base: float
    sigma: float
    tau_driver: float = 0.0
    tau_team: float = 0.0
    dnf_shock: float = 0.0


def sample_weights(seasons: pd.Series, current_season: int, boost: float = 5.0,
                   decay: float = 0.8) -> np.ndarray:
    """La saison en cours pèse `boost` ; les précédentes décroissent depuis 1.0."""
    age = current_season - seasons.to_numpy()
    return np.where(age <= 0, boost, decay ** np.maximum(age - 1, 0))


def _fit_reg(X, y, w) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**PARAMS).fit(X, y, sample_weight=w)


def _between_group_std(resid: np.ndarray, groups: np.ndarray, sigma: float) -> float:
    """Écart-type entre groupes des résidus moyens, bruit d'échantillonnage (sigma²/n) retiré."""
    df = pd.DataFrame({"r": resid, "g": groups}).groupby("g")["r"].agg(["mean", "size"])
    df = df[df["size"] >= MIN_GROUP_ROWS]
    if len(df) < 2:
        return 0.0
    tau2 = df["mean"].var(ddof=1) - float(np.mean(sigma**2 / df["size"]))
    return float(np.sqrt(max(0.0, tau2)))


def _dnf_shock(dnf: np.ndarray, drivers: np.ndarray) -> float:
    """Log-écart-type de la fiabilité : surdispersion des taux d'abandon par pilote."""
    df = pd.DataFrame({"dnf": dnf, "d": drivers}).groupby("d")["dnf"].agg(["mean", "size"])
    df = df[df["size"] >= MIN_GROUP_ROWS]
    p_bar = float(dnf[np.isin(drivers, df.index)].mean()) if len(df) else 0.0
    if len(df) < MIN_DNF_DRIVERS or p_bar <= 0.0:
        return 0.0
    v_bin = float(np.mean(p_bar * (1 - p_bar) / df["size"]))
    cv2 = max(0.0, df["mean"].var(ddof=1) - v_bin) / p_bar**2
    return float(min(np.sqrt(np.log1p(cv2)), MAX_DNF_SHOCK))


def train(features: pd.DataFrame, current_season: int) -> Models:
    known = features.dropna(subset=["position"]).reset_index(drop=True)
    X = known[FEATURE_COLUMNS]
    y = known["position"].astype(float)
    w = sample_weights(known["season"], current_season)
    finished = known["finished"].astype(bool).to_numpy()
    dnf = (~finished).astype(int)

    races = np.sort(known["race_idx"].unique())
    sigma = DEFAULT_SIGMA
    shocks = {}
    if len(races) > 2 * HOLDOUT_RACES:
        cutoff = races[-HOLDOUT_RACES]
        is_train = (known["race_idx"] < cutoff).to_numpy() & finished
        is_hold = (known["race_idx"] >= cutoff).to_numpy() & finished
        holdout_model = _fit_reg(X[is_train], y[is_train], w[is_train])
        resid = (y[is_hold] - holdout_model.predict(X[is_hold])).to_numpy()
        sigma = float(np.std(resid))
        hold = known[is_hold]
        is_current = (known["season"] == current_season).to_numpy()
        shocks = {
            # Forme persistante d'un pilote : dispersion de son erreur moyenne sur le holdout.
            "tau_driver": _between_group_std(resid, hold["driver_id"].to_numpy(), sigma),
            # Forme persistante d'une écurie : même calcul, regroupé par constructeur.
            "tau_team": _between_group_std(resid, hold["constructor_id"].to_numpy(), sigma),
            # Fiabilité : écarts de taux d'abandon entre pilotes de la saison, au-delà du hasard.
            "dnf_shock": _dnf_shock(dnf[is_current], known.loc[is_current, "driver_id"].to_numpy()),
        }

    reg = _fit_reg(X[finished], y[finished], w[finished])
    dnf_base = float(np.average(dnf, weights=w))
    clf = None
    if 0 < dnf.sum() < len(dnf):
        clf = lgb.LGBMClassifier(**PARAMS).fit(X, dnf, sample_weight=w)
    return Models(reg=reg, clf=clf, dnf_base=dnf_base, sigma=sigma, **shocks)


def predict(models: Models, rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[FEATURE_COLUMNS]
    out = rows[["season", "round", "driver_id"]].copy().reset_index(drop=True)
    out["pred_pos"] = np.clip(models.reg.predict(X), 1.0, None)
    if models.clf is None:
        out["p_dnf"] = models.dnf_base
    else:
        out["p_dnf"] = models.clf.predict_proba(X)[:, 1]
    return out
