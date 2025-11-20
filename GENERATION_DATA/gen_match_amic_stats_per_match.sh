#!/usr/bin/env bash
set -euo pipefail

# =========================
# Options via variables d'env
# =========================
# L1F_SEED             : fixe la graine (ex: 123) pour des résultats reproductibles (sinon aléatoire).
# L1F_MATCH_MODE       : "by_rule" (par défaut) ou "constant".
#   - by_rule = nb de matchs amicaux/an varie selon l'époque (voir fonction matches_per_year).
#   - constant = nb constant défini par L1F_MATCHES (ex: 120).
# L1F_MATCHES          : nb de matchs/an si L1F_MATCH_MODE=constant (ex: 120).

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.ligue1friendliesenv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Erreur: $PYTHON_BIN introuvable. Installe Python 3 et relance."
  exit 1
fi

# venv
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade --quiet pip
python -m pip install --quiet pandas numpy openpyxl XlsxWriter

python - <<'PYCODE'
import os, random
import numpy as np
import pandas as pd

# =========================
# Paramètres généraux
# =========================
YEAR_START = 1970
YEAR_END   = 2025
POSITIONS  = ["GK", "DF", "MF", "FW"]

METRICS = [
    "shots_total","shots_on_target","shots_off_target","shots_blocked","goals","expected_goals",
    "big_chances_created","big_chances_missed","penalties_won","penalties_taken","penalties_scored",
    "penalties_missed","dribbles_attempted","dribbles_successful","dribble_success_rate",
    "total_passes","passes_completed","pass_accuracy_pct","key_passes","assists","expected_assists",
    "progressive_passes","through_balls","crosses_attempted","crosses_completed","cross_accuracy_pct",
    "long_passes_attempted","long_passes_completed","forward_passesback","ward_passes",
    "tackles_attempted","tackles_won","tackle_success_rate","interceptions","clearances","blocks",
    "duels_won","duels_lost","duel_success_rate","aerial_duels_won","aerial_duels_lost","aerial_success_rate",
    "pressures","ball_recoveries","errors_leading_to_shot","saves","shots_faced","goals_conceded",
    "clean_sheet","save_pct","penalties_faced","penalties_saved","yellow_cards","red_cards",
    "fouls_committed","fouls_suffered","offsides","handballs","own_goals"
]
PERCENT_COLS = [c for c in METRICS if c.endswith("_pct") or c.endswith("_rate")]

# =========================
# Aléatoire : reproductible si L1F_SEED est défini
# =========================
seed_env = os.getenv("L1F_SEED")
if seed_env and seed_env.strip():
    try:
        seed_val = int(seed_env)
    except ValueError:
        seed_val = abs(hash(seed_env)) % (2**32)
    np.random.seed(seed_val)
    random.seed(seed_val)
else:
    # aléatoire à chaque exécution
    s = np.random.SeedSequence()
    seed_val = int(s.generate_state(1, dtype=np.uint32)[0])
    np.random.seed(seed_val)
    random.seed(seed_val)

# =========================
# Matches amicaux par an (par match = division par ce nombre)
# =========================
MATCH_MODE = os.getenv("L1F_MATCH_MODE", "by_rule").strip().lower()
MATCHES_CONST = int(os.getenv("L1F_MATCHES", "120"))

def matches_per_year(y:int)->int:
    if MATCH_MODE == "constant":
        return max(1, MATCHES_CONST)
    # by_rule : tendance à la hausse sur les décennies, légère baisse récente
    if y < 1990:     # 70s-80s : moins d'amicaux organisés/int'l
        return 60
    if y < 2000:     # 90s
        return 80
    if y < 2010:     # 2000s
        return 100
    if y < 2020:     # 2010s
        return 120
    if y == 2020:    # année particulière (covid) → un peu moins
        return 80
    if y <= 2025:    # 2021-2025 : un peu moins qu'avant, mais stable
        return 110
    return 110

# =========================
# Profils "amicaux" : multiplicateurs vs championnat
# =========================
FRIENDLY_FACTORS = {
    # un peu plus de jeu offensif
    "shots_total": 1.10, "shots_on_target": 1.10, "shots_off_target": 1.10, "shots_blocked": 1.05,
    "goals": 1.15, "expected_goals": 1.12,
    "big_chances_created": 1.12, "big_chances_missed": 1.12,
    "penalties_won": 1.05, "penalties_taken": 1.05, "penalties_scored": 1.05, "penalties_missed": 1.05,
    "offsides": 1.05,

    # dribbles : un peu plus d'initiatives
    "dribbles_attempted": 1.10, "dribbles_successful": 1.12, "dribble_success_rate": 1.02,

    # passes : un chouïa plus de fluidité, mais pas énorme
    "total_passes": 1.03, "passes_completed": 1.03, "pass_accuracy_pct": 1.01,
    "key_passes": 1.10, "assists": 1.12, "expected_assists": 1.12,
    "progressive_passes": 1.05, "through_balls": 1.08,
    "crosses_attempted": 1.05, "crosses_completed": 1.06, "cross_accuracy_pct": 1.01,
    "long_passes_attempted": 1.00, "long_passes_completed": 1.00,
    "forward_passesback": 0.98, "ward_passes": 1.02,

    # défense/pressing : intensité un peu moindre
    "tackles_attempted": 0.92, "tackles_won": 0.92, "tackle_success_rate": 1.00,
    "interceptions": 0.95, "clearances": 0.95, "blocks": 0.95,
    "pressures": 0.80, "ball_recoveries": 0.90,
    "errors_leading_to_shot": 1.05,
    "duels_won": 0.95, "duels_lost": 0.95, "duel_success_rate": 1.00,
    "aerial_duels_won": 0.95, "aerial_duels_lost": 0.95, "aerial_success_rate": 1.00,

    # gardiens : un peu moins de tirs encaissés, % arrêts légèrement plus bas (rotation/essais)
    "saves": 0.98, "shots_faced": 0.95, "goals_conceded": 1.08, "clean_sheet": 0.95, "save_pct": 0.98,
    "penalties_faced": 1.00, "penalties_saved": 0.95,

    # discipline : moins d'engagement
    "yellow_cards": 0.65, "red_cards": 0.50,
    "fouls_committed": 0.85, "fouls_suffered": 0.90,
    "handballs": 0.90, "own_goals": 1.00,
}

# =========================
# Tendances/époques (réutilisées, mais plus douces)
# =========================
def era_factor(y:int, m:str)->float:
    # Amicaux : effets d'époque un peu atténués
    if y < 1990:
        if m in {"total_passes","passes_completed","progressive_passes","pressures","ball_recoveries"}:
            return 0.65
        return 0.85
    if y < 2000:
        if m in {"total_passes","passes_completed","progressive_passes","pressures","ball_recoveries"}:
            return 0.80
        return 0.92
    return 1.00

def smooth_trend(m:str, y:int)->float:
    t = max(0, y - 2000) / 25.0
    if m in {"total_passes","passes_completed","progressive_passes","pressures","ball_recoveries"}:
        return 1.0 + 0.28 * t
    if m in {"shots_total","goals","expected_goals","shots_on_target"}:
        return 1.0 + 0.08 * t
    if m.endswith("_pct") or m.endswith("_rate"):
        return 1.0 + 0.02 * t
    return 1.0 + 0.04 * t

# =========================
# Baselines "championnat" par saison (on réutilise des ordres de grandeur)
# =========================
def base_season_mean_competitive(metric:str, pos:str)->float:
    if metric in {"saves","shots_faced","goals_conceded","clean_sheet","save_pct","penalties_faced","penalties_saved"}:
        if pos != "GK": return 0.0
        return {"saves":120.0,"shots_faced":180.0,"goals_conceded":45.0,"clean_sheet":12.0,
                "save_pct":72.0,"penalties_faced":4.0,"penalties_saved":1.0}[metric]

    if metric in {"shots_total","shots_on_target","shots_off_target","shots_blocked","goals","expected_goals",
                  "big_chances_created","big_chances_missed","penalties_won","penalties_taken",
                  "penalties_scored","penalties_missed","offsides"}:
        if pos=="FW":
            return {"shots_total":260,"shots_on_target":110,"shots_off_target":90,"shots_blocked":60,
                    "goals":65,"expected_goals":60,"big_chances_created":35,"big_chances_missed":40,
                    "penalties_won":8,"penalties_taken":12,"penalties_scored":10,"penalties_missed":2,
                    "offsides":65}[metric]
        if pos=="MF":
            return {"shots_total":160,"shots_on_target":55,"shots_off_target":65,"shots_blocked":40,
                    "goals":35,"expected_goals":32,"big_chances_created":45,"big_chances_missed":25,
                    "penalties_won":4,"penalties_taken":5,"penalties_scored":4,"penalties_missed":1,
                    "offsides":25}[metric]
        if pos=="DF":
            return {"shots_total":80,"shots_on_target":25,"shots_off_target":35,"shots_blocked":20,
                    "goals":12,"expected_goals":10,"big_chances_created":20,"big_chances_missed":10,
                    "penalties_won":2,"penalties_taken":2,"penalties_scored":2,"penalties_missed":0.2,
                    "offsides":10}[metric]
        if pos=="GK":
            return {"shots_total":5,"shots_on_target":2,"shots_off_target":2,"shots_blocked":1,
                    "goals":0.2,"expected_goals":0.1,"big_chances_created":1,"big_chances_missed":0.5,
                    "penalties_won":0.2,"penalties_taken":0.05,"penalties_scored":0.03,"penalties_missed":0.02,
                    "offsides":0.0}[metric]

    if metric in {"dribbles_attempted","dribbles_successful","dribble_success_rate"}:
        if metric=="dribble_success_rate":
            return {"FW":48.0,"MF":55.0,"DF":60.0,"GK":20.0}[pos]
        return {
            ("FW","dribbles_attempted"):650, ("FW","dribbles_successful"):310,
            ("MF","dribbles_attempted"):520, ("MF","dribbles_successful"):290,
            ("DF","dribbles_attempted"):220, ("DF","dribbles_successful"):130,
            ("GK","dribbles_attempted"):12,  ("GK","dribbles_successful"):6
        }[(pos,metric)]

    if metric in {"total_passes","passes_completed","pass_accuracy_pct","key_passes","assists","expected_assists",
                  "progressive_passes","through_balls","crosses_attempted","crosses_completed","cross_accuracy_pct",
                  "long_passes_attempted","long_passes_completed","forward_passesback","ward_passes"}:
        if metric=="pass_accuracy_pct":
            return {"GK":80.0,"DF":87.0,"MF":84.0,"FW":76.0}[pos]
        if metric=="cross_accuracy_pct":
            return {"GK":0.0,"DF":28.0,"MF":26.0,"FW":24.0}[pos]
        if pos=="MF":
            return {"total_passes":31000,"passes_completed":26000,"key_passes":850,"assists":320,"expected_assists":300,
                    "progressive_passes":5200,"through_balls":140,"crosses_attempted":2800,"crosses_completed":720,
                    "long_passes_attempted":2800,"long_passes_completed":1600,"forward_passesback":9000,"ward_passes":18000}[metric]
        if pos=="DF":
            return {"total_passes":36000,"passes_completed":32000,"key_passes":380,"assists":110,"expected_assists":95,
                    "progressive_passes":4800,"through_balls":60,"crosses_attempted":3400,"crosses_completed":920,
                    "long_passes_attempted":4200,"long_passes_completed":2400,"forward_passesback":16000,"ward_passes":20000}[metric]
        if pos=="FW":
            return {"total_passes":14000,"passes_completed":10000,"key_passes":720,"assists":360,"expected_assists":340,
                    "progressive_passes":2600,"through_balls":120,"crosses_attempted":2500,"crosses_completed":600,
                    "long_passes_attempted":900,"long_passes_completed":420,"forward_passesback":3800,"ward_passes":9600}[metric]
        if pos=="GK":
            return {"total_passes":8000,"passes_completed":6200,"key_passes":30,"assists":8,"expected_assists":6,
                    "progressive_passes":400,"through_balls":4,"crosses_attempted":0,"crosses_completed":0,
                    "long_passes_attempted":2500,"long_passes_completed":1100,"forward_passesback":4200,"ward_passes":3800}[metric]

    if metric in {"tackles_attempted","tackles_won","tackle_success_rate","interceptions","clearances","blocks",
                  "pressures","ball_recoveries","errors_leading_to_shot","duels_won","duels_lost","duel_success_rate",
                  "aerial_duels_won","aerial_duels_lost","aerial_success_rate"}:
        if metric in {"tackle_success_rate","duel_success_rate","aerial_success_rate"}:
            return {
                ("GK","tackle_success_rate"):10.0, ("DF","tackle_success_rate"):66.0,
                ("MF","tackle_success_rate"):62.0, ("FW","tackle_success_rate"):50.0,
                ("GK","duel_success_rate"):20.0,   ("DF","duel_success_rate"):54.0,
                ("MF","duel_success_rate"):52.0,   ("FW","duel_success_rate"):48.0,
                ("GK","aerial_success_rate"):15.0, ("DF","aerial_success_rate"):55.0,
                ("MF","aerial_success_rate"):50.0, ("FW","aerial_success_rate"):48.0,
            }[(pos,metric)]
        if pos=="DF":
            return {"tackles_attempted":5400,"tackles_won":3600,"interceptions":5200,"clearances":11500,"blocks":2100,
                    "pressures":11000,"ball_recoveries":9800,"errors_leading_to_shot":22,"duels_won":8200,"duels_lost":7000,
                    "aerial_duels_won":4200,"aerial_duels_lost":3600}[metric]
        if pos=="MF":
            return {"tackles_attempted":6200,"tackles_won":4000,"interceptions":4700,"clearances":3200,"blocks":1600,
                    "pressures":22000,"ball_recoveries":15000,"errors_leading_to_shot":18,"duels_won":9200,"duels_lost":8400,
                    "aerial_duels_won":2800,"aerial_duels_lost":3000}[metric]
        if pos=="FW":
            return {"tackles_attempted":1800,"tackles_won":1000,"interceptions":1600,"clearances":1400,"blocks":900,
                    "pressures":16000,"ball_recoveries":7200,"errors_leading_to_shot":12,"duels_won":5200,"duels_lost":5700,
                    "aerial_duels_won":2200,"aerial_duels_lost":2600}[metric]
        if pos=="GK":
            return {"tackles_attempted":140,"tackles_won":60,"interceptions":120,"clearances":800,"blocks":220,
                    "pressures":800,"ball_recoveries":1600,"errors_leading_to_shot":10,"duels_won":300,"duels_lost":500,
                    "aerial_duels_won":120,"aerial_duels_lost":180}[metric]

    if metric in {"yellow_cards","red_cards","fouls_committed","fouls_suffered","handballs","own_goals"}:
        if pos=="DF":
            return {"yellow_cards":480,"red_cards":24,"fouls_committed":9800,"fouls_suffered":6200,"handballs":40,"own_goals":18}[metric]
        if pos=="MF":
            return {"yellow_cards":620,"red_cards":20,"fouls_committed":11200,"fouls_suffered":11000,"handballs":30,"own_goals":8}[metric]
        if pos=="FW":
            return {"yellow_cards":520,"red_cards":18,"fouls_committed":7600,"fouls_suffered":12800,"handballs":28,"own_goals":6}[metric]
        if pos=="GK":
            return {"yellow_cards":60,"red_cards":4,"fouls_committed":400,"fouls_suffered":700,"handballs":6,"own_goals":2}[metric]

    return 50.0

# =========================
# Génération d'une ligne (année × poste), normalisé PAR MATCH
# =========================
def generate_row(y:int, pos:str)->dict:
    row = {"year": y, "position": pos}
    mpy = max(1, matches_per_year(y))

    for m in METRICS:
        # xG/xA indisponibles historiquement -> NaN avant 2014
        if m in {"expected_goals","expected_assists"} and y < 2014:
            row[f"{m}_min"] = np.nan
            row[f"{m}_max"] = np.nan
            row[f"{m}_mean"] = np.nan
            continue

        base = base_season_mean_competitive(m, pos)
        # applique facteurs "amicaux"
        friendly_mult = FRIENDLY_FACTORS.get(m, 1.0)
        val = base * friendly_mult * era_factor(y, m) * smooth_trend(m, y)

        # bruit (un peu plus large qu'en compétition, car hétérogène)
        noise_scale = 0.22 if y < 1995 else (0.14 if y < 2010 else 0.10)
        val *= (1.0 + np.random.normal(0.0, noise_scale))
        val = max(0.0, float(val))

        if m in PERCENT_COLS:
            mean_val = float(np.clip(val, 0.0, 100.0))
            width = 12.0
            mn = float(np.clip(mean_val - (width + np.random.uniform(0, 5)), 0.0, 100.0))
            mx = float(np.clip(mean_val + (width + np.random.uniform(0, 5)), 0.0, 100.0))
        else:
            per_match = val / mpy
            # bandes min/max ±(25%..45%)
            spread_low  = np.random.uniform(0.25, 0.45)
            spread_high = np.random.uniform(0.25, 0.45)
            mn = max(0.0, per_match * (1.0 - spread_low))
            mx = per_match * (1.0 + spread_high)
            mean_val = round(per_match, 3)
            mn = round(mn, 3)
            mx = round(mx, 3)

        row[f"{m}_min"]  = mn
        row[f"{m}_max"]  = mx
        row[f"{m}_mean"] = mean_val

    return row

def build_dataframe()->pd.DataFrame:
    rows = []
    for y in range(YEAR_START, YEAR_END+1):
        for p in POSITIONS:
            rows.append(generate_row(y, p))
    return pd.DataFrame(rows).sort_values(["year","position"]).reset_index(drop=True)

def export_files(df:pd.DataFrame)->None:
    csv_path  = "./Ligue1_friendlies_estimated_per_match_1970_2025.csv"
    xlsx_path = "./Ligue1_friendlies_estimated_per_match_1970_2025.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    readme = pd.DataFrame({
        "README":[
            "Synthetic FRIENDLY (amicaux) Ligue 1 estimates (1970–2025) per match, by year × position (GK/DF/MF/FW).",
            "Randomised yet realistic profiles DIFFERENT from competitive matches.",
            "Per-match = seasonal friendly estimates divided by matches_per_year(year).",
            "xG/xA -> NaN before 2014 for realism.",
            "Less intensity (pressures/fouls/cards↓), more offensive flavour (shots/goals/key_passes↑).",
            "Env vars: L1F_SEED, L1F_MATCH_MODE=('by_rule'|'constant'), L1F_MATCHES (if constant)."
        ]
    })

    engine_used = None
    for eng in ("openpyxl","xlsxwriter"):
        try:
            with pd.ExcelWriter(xlsx_path, engine=eng) as wr:
                df.to_excel(wr, index=False, sheet_name="friendlies_per_match_estimated")
                readme.to_excel(wr, index=False, sheet_name="README")
            engine_used = eng
            break
        except Exception:
            continue
    if engine_used is None:
        raise RuntimeError("Excel write failed. Install openpyxl or XlsxWriter.")

def main():
    df = build_dataframe()
    export_files(df)
    print("OK ✅  Fichiers générés :")
    print(" - ./Ligue1_friendlies_estimated_per_match_1970_2025.csv")
    print(" - ./Ligue1_friendlies_estimated_per_match_1970_2025.xlsx")

if __name__ == "__main__":
    main()
PYCODE

echo
echo "Terminé ✅"
echo "Fichiers attendus :"
echo " - ./Ligue1_friendlies_estimated_per_match_1970_2025.csv"
echo " - ./Ligue1_friendlies_estimated_per_match_1970_2025.xlsx"
echo
echo "Exemples :"
echo "  L1F_SEED=123 ./gen_ligue1_friendlies_per_match.sh                   # résultats reproductibles"
echo "  L1F_MATCH_MODE=constant L1F_MATCHES=120 ./gen_ligue1_friendlies_per_match.sh  # 120 amicaux/an"

