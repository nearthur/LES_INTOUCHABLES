#!/usr/bin/env bash
# Générateur de stats par match (ECE Paris) – v6
# Dépendances : Python 3.8+, pandas, numpy, openpyxl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HISTO="${SCRIPT_DIR}/histo_contrats.csv"
L1TEAMS="${SCRIPT_DIR}/equipes_ligue1.csv"
OUT="${SCRIPT_DIR}/stats_par_match.csv"
RAPPORT="${SCRIPT_DIR}/validation_report.xlsx"
START_YEAR=1970
END_YEAR=2025
SEED=42

python3 - <<'PY' "$HISTO" "$L1TEAMS" "$OUT" "$RAPPORT" "$START_YEAR" "$END_YEAR" "$SEED"
import sys, random, re
from datetime import date, timedelta
import pandas as pd, numpy as np

# -------------------- Args --------------------
HISTO_PATH   = sys.argv[1]
L1_PATH      = sys.argv[2]
OUT_PATH     = sys.argv[3]
RAPPORT_PATH = sys.argv[4]
START_YEAR   = int(sys.argv[5])
END_YEAR     = int(sys.argv[6])
SEED         = int(sys.argv[7])

random.seed(SEED); np.random.seed(SEED)
TEAM_NAME = "ECE Paris"

# -------------------- Helpers --------------------
DEF = {"DD","DG","DC"}; MID = {"MC","MOC","MDC","MD","MG"}; FWD = {"BU","AD","AG"}
def macro(pos):
    if pos=="GK": return "GK"
    if pos in DEF: return "DEF"
    if pos in MID: return "MID"
    if pos in FWD: return "FWD"
    return "MID"
def clamp(x, lo, hi): return max(lo, min(hi, x))
def meteo_by_month(m):
    r = np.random.rand()
    if m in (6,7,8):       return "Ensoleillé" if r<0.6 else ("Couvert" if r<0.85 else "Pluie")
    if m in (12,1,2):      return "Pluie" if r<0.6 else ("Neige" if r<0.1 else "Couvert")
    return "Couvert" if r<0.5 else ("Pluie" if r<0.8 else "Ensoleillé")

# -------------------- Load histo_contrats --------------------
histo = pd.read_csv(HISTO_PATH)
histo.rename(columns={
    "Player ID":"player_id","Nom Joueur":"nom joueur","Prénom Joueur":"prénom joueur",
    "Poste Joueur":"poste","Date de signature":"date_deb","Date de fin":"date_fin"
}, inplace=True)
histo["date_deb"] = pd.to_datetime(histo["date_deb"]).dt.date
histo["date_fin"] = pd.to_datetime(histo["date_fin"]).dt.date
histo["macro"] = histo["poste"].apply(macro)

# -------------------- Load equipes_ligue1.csv (robuste) --------------------
def _clean_colname(c: str) -> str:
    c = str(c).replace("\u00a0"," ").replace("\u2007"," ").replace("\u202f"," ")
    return re.sub(r"\s+"," ",c).strip()

def _normalize_l1_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c:_clean_colname(c) for c in df.columns})
    saison_alias = {"saison","season","annee","année","year"}
    saison_col = None
    for c in df.columns:
        if _clean_colname(c).lower() in saison_alias:
            saison_col = c; break
    if saison_col is None:
        raise SystemExit("Le fichier L1 doit contenir une colonne 'Saison' (ou Season/Année/Year).")
    club_cols = {}
    for c in df.columns:
        cn = _clean_colname(c).lower()
        m = re.match(r"(club|equipe|équipe|team)\s*([0-9]{1,2})$", cn)
        if m:
            idx = int(m.group(2))
            if 1 <= idx <= 20:
                club_cols[idx] = c
    if not club_cols: raise SystemExit("Aucune colonne 'Club 1..20' trouvée.")
    out = df[[saison_col] + [club_cols[i] for i in sorted(club_cols)]].copy()
    out.rename(columns={saison_col:"Saison"}, inplace=True)
    for i in sorted(club_cols): out.rename(columns={club_cols[i]:f"Club {i}"}, inplace=True)
    return out

def parse_saison_start(v):
    if pd.isna(v): return None
    s = str(v).replace("\u00a0"," ").strip()
    for sep in ["-","/","–","—"]:
        if sep in s:
            left = s.split(sep)[0].strip()
            try: return int(left)
            except: return None
    try: return int(float(s))
    except: return None

def load_l1_table(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
        except Exception:
            try:
                df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
    df = _normalize_l1_columns(df)
    df["saison_start"] = df["Saison"].apply(parse_saison_start)
    df = df.dropna(subset=["saison_start"]).copy()
    df["saison_start"] = df["saison_start"].astype(int)
    return df

l1tab = load_l1_table(L1_PATH)

def clubs_for_season(Y):
    df = l1tab[l1tab["saison_start"]==Y]
    if df.empty: return []
    row = df.iloc[0]
    clubs = [str(row[f"Club {i}"]).strip() for i in range(1,21) if f"Club {i}" in row and pd.notna(row[f"Club {i}"])]
    clubs = [c for c in clubs if c and c != TEAM_NAME]
    if len(clubs) >= 19:
        return list(pd.Series(clubs).sample(19, random_state=np.random.randint(0,1_000_000)))
    base = clubs[:]
    while len(clubs) < 19 and base: clubs.append(random.choice(base))
    while len(clubs) < 19: clubs.append(f"Club_Filler_{len(clubs)+1}")
    return clubs

# -------------------- Calendrier --------------------
def first_weekend_on_or_after(d0: date) -> date:
    wd = d0.weekday()  # Mon=0..Sun=6
    delta = (5 - wd) if wd <= 5 else 0   # viser samedi
    return d0 + timedelta(days=delta)

def gen_l1_dates_two_phases(Y):
    start = date(Y,8,15); end_limit = date(Y+1,5,15)
    dates=[]; cur = first_weekend_on_or_after(start)
    for i in range(38):
        if cur > end_limit: cur = end_limit
        dates.append(cur)
        gap = 3 if i%2==0 else 4  # alterner midweek/weekend pour rester ≤4
        nxt = cur + timedelta(days=gap)
        if i%2==1:
            sat = nxt + timedelta(days=(5 - nxt.weekday())%7)
            if (sat - dates[-1]).days <= 4: nxt = sat
        cur = nxt
    return dates

def gen_friendlies_dates_for_year(Y):
    dates=[]; cur=date(Y,7,1)
    for _ in range(8):
        dates.append(min(cur, date(Y,8,14)))
        cur += timedelta(days=np.random.randint(3,5))
    dates = sorted(list(dict.fromkeys([min(d,date(Y,8,14)) for d in dates])))
    while len(dates)<8:
        dates.append(min(dates[-1]+timedelta(days=np.random.randint(1,3)), date(Y,8,14)))
    return dates[:8]

FRIENDS = list(dict.fromkeys([
 "Real Madrid","FC Barcelona","Manchester United","Bayern Munich","Liverpool FC","Manchester City","Chelsea FC",
 "Paris Saint-Germain","Juventus","Arsenal FC","AC Milan","Inter Milan","Borussia Dortmund","Atlético de Madrid",
 "Tottenham Hotspur","Ajax","SL Benfica","FC Porto","Boca Juniors","River Plate","Flamengo","Palmeiras","Santos FC",
 "Celtic FC","Rangers FC","AS Roma","SSC Napoli","Lazio","Olympique de Marseille","Olympique Lyonnais","AS Monaco",
 "Sevilla FC","Valencia CF","Villarreal CF","Real Sociedad","Athletic Bilbao","RB Leipzig","Bayer 04 Leverkusen",
 "Eintracht Frankfurt","PSV Eindhoven","Feyenoord","Galatasaray","Fenerbahçe","Beşiktaş","Sporting CP","Shakhtar Donetsk",
 "Dynamo Kyiv","Zenit Saint Petersburg","Club Brugge","Red Bull Salzburg","Olympiacos","LA Galaxy","Inter Miami CF","Al Nassr",
 "Al-Hilal","Aston Villa","Newcastle United","Brighton & Hove Albion","West Ham United","Everton FC","Leicester City",
 "Leeds United","Real Betis","Fiorentina","Atalanta","Torino FC","VfL Wolfsburg","Borussia Mönchengladbach","Werder Bremen",
 "Stade Rennais FC","Lille OSC","OGC Nice","RC Lens","SC Braga","Anderlecht","Genk","FC Copenhagen","Slavia Prague",
 "Sparta Prague","Red Star Belgrade","Partizan Belgrade","Dinamo Zagreb","Hajduk Split","PAOK","AEK Athens","CSKA Moscow",
 "Spartak Moscow","Club América","C.D. Guadalajara","Tigres UANL","CF Montréal","Toronto FC","Vancouver Whitecaps FC",
 "Corinthians","São Paulo FC","Grêmio"
]))

def schedule_season(Y):
    clubs = clubs_for_season(Y)
    if not clubs: return []
    first_legs = clubs[:] ; np.random.shuffle(first_legs)
    second_legs = clubs[:] ; np.random.shuffle(second_legs)
    dates = gen_l1_dates_two_phases(Y)
    fixtures=[]; seen={}
    for d,opp in zip(dates[:19], first_legs):
        home = np.random.rand()<0.5
        seen[opp] = "HOME" if home else "AWAY"
        fixtures.append((d, opp, seen[opp]))
    for d,opp in zip(dates[19:], second_legs):
        where = "AWAY" if seen.get(opp,"HOME")=="HOME" else "HOME"
        fixtures.append((d, opp, where))
    return fixtures

def schedule_friendlies(Y):
    dates=gen_friendlies_dates_for_year(Y)
    pool=FRIENDS[:]; random.shuffle(pool); used=set()
    fixtures=[]
    for d in dates:
        opp=None
        for c in pool:
            if c not in used: opp=c; used.add(c); break
        if opp is None: opp=random.choice(FRIENDS)
        fixtures.append((d, opp, "AWAY"))
    return fixtures

# -------------------- Favoris & Issue --------------------
def compute_outcome(is_home, is_fav):
    if is_home: pV,pN,pD=0.40,0.30,0.30
    else:
        pV,pN,pD=0.30,0.30,0.40
        if is_fav: pV=min(0.80,pV+0.10); pD=max(0.05,pD-0.10)
    r=np.random.rand()
    if r<pV: return "V"
    if r<pV+pN: return "N"
    return "D"

# -------------------- Joueurs & formation --------------------
def players_under_contract(match_day):
    m=(histo["date_deb"]<=match_day)&(histo["date_fin"]>=match_day)
    df=histo[m].copy()
    if df.empty: df=histo.sort_values("date_fin",ascending=False).head(30).copy()
    df=df.sort_values(["player_id","date_deb"]).drop_duplicates("player_id", keep="last")
    return df

def pick_lineup(contracted):
    def pick(df, macro_name, n):
        pool=df[df["macro"]==macro_name]
        if len(pool)>=n: return pool.sample(n, replace=False, random_state=np.random.randint(0,1_000_000))
        if macro_name=="DEF": alt=df[df["poste"].isin(["MDC","MC","DD","DG","DC"])]
        elif macro_name=="MID": alt=df[df["poste"].isin(list(MID|DEF))]
        elif macro_name=="FWD": alt=df[df["poste"].isin(["MOC","AD","AG","MC","BU"])]
        else: alt=pd.DataFrame(columns=df.columns)
        need=n-len(pool); supl=alt.sample(min(len(alt),need), replace=False) if len(alt)>0 else pd.DataFrame(columns=df.columns)
        return pd.concat([pool,supl]).drop_duplicates("player_id").head(n)
    XI=pd.concat([pick(contracted,"GK",1),pick(contracted,"DEF",4),pick(contracted,"MID",4),pick(contracted,"FWD",2)])
    if len(XI)<11:
        rest=contracted[~contracted["player_id"].isin(XI["player_id"])].sample(11-len(XI), replace=True)
        XI=pd.concat([XI,rest])
    return XI

# -------------------- Cartons & Substitutions --------------------
YELLOW_RATE = {"GK":0.04,"DEF":0.18,"MID":0.16,"FWD":0.10}
RED_DIRECT  = {"GK":0.005,"DEF":0.020,"MID":0.015,"FWD":0.010}
SECOND_YELLOW_PROB = 0.12
MIN_SUB_MINUTE = 10

def subs_distribution(comp):
    if comp == "Ligue 1":
        return np.random.choice([0,1,2,3], p=[0.15,0.35,0.35,0.15])
    else:
        return np.random.choice([0,1,2,3,4,5], p=[0.10,0.20,0.30,0.25,0.10,0.05])

def draw_card_events(duration, mac, start=0):
    span = duration - start
    yc_minutes=[]; rc_min=None
    if np.random.rand() < YELLOW_RATE.get(mac, 0.1):
        m1 = start + np.random.randint(1, span+1)
        yc_minutes=[m1]
        if np.random.rand() < SECOND_YELLOW_PROB and m1 < duration:
            m2 = np.random.randint(m1+1, duration+1)
            yc_minutes=[m1,m2]; rc_min=m2
    if rc_min is None and np.random.rand() < RED_DIRECT.get(mac, 0.01):
        rc_min = start + np.random.randint(1, span+1)
    yc_minutes.sort()
    return yc_minutes, rc_min

# ====== VERSION ROBUSTE (intervalles [start, end), reconciliation) ======
def assign_minutes_and_cards(competition, XI, bench):
    def clamp_interval(start, end, duration):
        start = int(max(0, min(start, duration)))
        end   = int(max(0, min(end, duration)))
        if end < start: end = start
        return start, end

    subs_max     = 3 if competition == "Ligue 1" else 5
    subs_target  = min(subs_distribution(competition), subs_max)
    stoppage     = np.random.randint(1, 8)
    duration     = 90 + stoppage

    # XI initial
    timeline = []  # dict: {player_id, macro, start, end, row, yc_minutes, rc_min, is_sub}
    XI = XI.sample(frac=1.0, random_state=np.random.randint(0,1_000_000))
    for _, r in XI.iterrows():
        yc, rc = draw_card_events(duration, r["macro"], start=0)
        end = rc if rc is not None else duration
        start, end = clamp_interval(0, end, duration)
        timeline.append({
            "player_id": r["player_id"], "macro": r["macro"],
            "start": start, "end": end, "row": r,
            "yc_minutes": yc, "rc_min": rc, "is_sub": False
        })

    bench_pool = bench[~bench["player_id"].isin([p["player_id"] for p in timeline])]
    bench_pool = bench_pool.sample(frac=1.0, random_state=np.random.randint(0,1_000_000))
    subs_done = 0
    subs_min_time = None

    def try_substitute_one(sub_row):
        nonlocal subs_done, bench_pool, subs_min_time
        macro_name = sub_row["macro"]
        candidates = [p for p in timeline if p["macro"]==macro_name and p["end"] > MIN_SUB_MINUTE]
        if not candidates:
            return False
        victim = random.choice(candidates)

        m_max = min(victim["end"] - 1, duration - 1)
        if m_max < MIN_SUB_MINUTE:
            return False
        m = np.random.randint(MIN_SUB_MINUTE, m_max + 1)

        if victim["rc_min"] is not None and victim["rc_min"] > m:
            victim["rc_min"] = None
        victim["yc_minutes"] = [t for t in victim["yc_minutes"] if t <= m]

        victim["end"] = m
        yc_in, rc_in = draw_card_events(duration, macro_name, start=m)
        end_in = rc_in if rc_in is not None else duration
        start_in, end_in = clamp_interval(m, end_in, duration)
        timeline.append({
            "player_id": sub_row["player_id"], "macro": macro_name,
            "start": start_in, "end": end_in, "row": sub_row,
            "yc_minutes": yc_in, "rc_min": rc_in, "is_sub": True
        })
        bench_pool = bench_pool[bench_pool["player_id"] != sub_row["player_id"]]
        subs_done += 1
        subs_min_time = m if subs_min_time is None else min(subs_min_time, m)
        return True

    # 1er passage DEF/MID/FWD
    for macro_name in ["DEF","MID","FWD"]:
        if subs_done >= subs_target: break
        cand = bench_pool[bench_pool["macro"]==macro_name]
        if cand.empty: continue
        try_substitute_one(cand.iloc[0])
    # compléter
    while subs_done < subs_target and not bench_pool.empty:
        if not try_substitute_one(bench_pool.iloc[0]):
            bench_pool = bench_pool.iloc[1:]

    # Finalisation (yc/rc effectifs)
    final=[]
    for p in timeline:
        start, end = clamp_interval(p["start"], p["end"], duration)
        yc_eff = [t for t in p["yc_minutes"] if start < t <= end]
        rc_eff = 1 if (p["rc_min"] is not None and start < p["rc_min"] <= end) else 0
        final.append((p["row"], start, end, duration, len(yc_eff), rc_eff, p["is_sub"]))

    # Reconciliation (si total != attendu)
    def reconcile(final_list, duration):
        total = sum(end - start for _, start, end, _, _, _, _ in final_list)
        reds_loss = sum((duration - end) for _, start, end, _, _, rc, _ in final_list if rc == 1)
        expected = 11 * duration - reds_loss
        delta = total - expected
        if delta == 0:
            return final_list, expected, total
        final_mut = list(final_list)
        if delta > 0:
            # Trop de minutes → raccourcir segments non-rouges, en priorité subs
            for i in range(len(final_mut)-1, -1, -1):
                row, s, e, dur, yc, rc, is_sub = final_mut[i]
                if rc == 1: continue
                span = e - s
                if span <= 0: continue
                cut = min(delta, span)
                final_mut[i] = (row, s, e - cut, dur, yc, rc, is_sub)
                delta -= cut
                if delta == 0: break
        else:
            need = -delta
            for i in range(len(final_mut)-1, -1, -1):
                row, s, e, dur, yc, rc, is_sub = final_mut[i]
                if rc == 1 or e >= duration: continue
                room = duration - e
                add = min(need, room)
                if add > 0:
                    final_mut[i] = (row, s, e + add, dur, yc, rc, is_sub)
                    need -= add
                    if need == 0: break
        total2 = sum(end - start for _, start, end, _, _, _, _ in final_mut)
        reds_loss2 = sum((duration - end) for _, start, end, _, _, rc, _ in final_mut if rc == 1)
        expected2 = 11 * duration - reds_loss2
        return final_mut, expected2, total2

    final, expected_after, total_after = reconcile(final, duration)
    return final, duration, subs_done, subs_min_time if subs_min_time is not None else ""

# -------------------- Générateur de stats joueur --------------------
BASE = {
 "Ligue 1": {"GK":{"shots_faced":(3,4),"saves":(2,3),"passes":(25,12),"long":(8,5)},
             "DEF":{"shots_total":(0.5,1.0),"passes":(55,18),"tackles":(2.5,1.5)},
             "MID":{"shots_total":(1.2,1.2),"passes":(65,20),"key":(1.0,1.0),"tackles":(2.0,1.2)},
             "FWD":{"shots_total":(2.0,1.8),"passes":(30,12),"key":(0.8,0.9)}},
 "Amical" : {"GK":{"shots_faced":(2.5,3.0),"saves":(2.0,2.5),"passes":(20,10),"long":(7,4)},
             "DEF":{"shots_total":(0.4,0.8),"passes":(50,16),"tackles":(2.2,1.3)},
             "MID":{"shots_total":(1.0,1.0),"passes":(60,18),"key":(0.8,0.9),"tackles":(1.8,1.0)},
             "FWD":{"shots_total":(1.6,1.4),"passes":(28,10),"key":(0.7,0.8)}}}

def gen_stats_row(comp, mac, minutes_played, outcome, yc_override, rc_override):
    def N(mu_sigma, lo=0): return max(lo, int(np.random.normal(*mu_sigma)))
    shots_total = N(BASE[comp][mac].get("shots_total",(0.6,1.2)))
    shots_on = np.random.binomial(shots_total, 0.4 if mac in ("FWD","MID") else 0.2)
    shots_off = max(0, shots_total - shots_on - np.random.randint(0,2))
    shots_blocked = max(0, shots_total - shots_on - shots_off)
    goals = np.random.binomial(shots_on, 0.25 if mac=="FWD" else 0.12 if mac=="MID" else 0.05)
    xg = round(abs(np.random.normal(0.25 + 0.2*(goals>0), 0.2)),2)

    key_passes = N(BASE[comp][mac].get("key",(0.4,0.8)))
    total_passes = N(BASE[comp][mac].get("passes",(40,15)), lo=5)
    passes_completed = clamp(int(np.random.normal(total_passes*0.86, total_passes*0.06)), 0, total_passes)
    pass_acc = round(100*passes_completed/max(1,total_passes),1)

    dr_a = N((2,1)) if mac in ("FWD","MID") else N((0.6,1))
    dr_s = clamp(np.random.binomial(dr_a, 0.52 if mac in ("FWD","MID") else 0.35), 0, dr_a)

    prog = N((6,3)) if mac!="GK" else 0
    through = N((0.6,0.8)) if mac in ("MID","FWD") else N((0.2,0.5))
    cross_a = N((1.5,1.2)) if mac in ("MID","FWD") else N((0.3,0.8))
    cross_c = clamp(np.random.binomial(cross_a, 0.28), 0, cross_a)
    cross_rate = round(100*cross_c/max(1,cross_a),1)
    long_a = N(BASE[comp][mac].get("long",(5,4)))
    long_c = clamp(np.random.binomial(long_a, 0.55 if mac=="GK" else 0.48), 0, long_a)
    fwd_p = max(0, int(np.random.normal(total_passes*0.55, total_passes*0.15)))
    back_p = clamp(total_passes - fwd_p, 0, total_passes)

    tack_a = N(BASE[comp][mac].get("tackles",(1.4,1.2)))
    tack_w = clamp(np.random.binomial(tack_a, 0.58), 0, tack_a)
    tack_rate = round(100*tack_w/max(1,tack_a),1)
    inter = N((1.4,1.0)) if mac!="FWD" else N((0.6,1.0))
    clear = N((3.0,1.5)) if mac in ("DEF","GK") else N((0.8,1.0))
    blocks = N((0.8,0.8)) if mac in ("DEF","GK") else N((0.4,0.7))
    duels_w = N((5,2.5)) if mac!="GK" else N((0.6,0.7))
    duels_l = N((5,2.5)) if mac!="GK" else N((0.6,0.7))
    duel_rate = round(100*duels_w/max(1,duels_w+duels_l),1)

    aer_w = N((2.0,1.5)) if mac in ("DEF","FWD") else N((1.0,1.0))
    aer_l = N((2.0,1.5)) if mac in ("DEF","FWD") else N((1.0,1.0))
    aer_rate = round(100*aer_w/max(1,aer_w+aer_l),1)

    press = N((8,4)) if mac!="GK" else N((0,1))
    recov = N((6,3)) if mac!="GK" else N((2,1))
    err_shot = np.random.binomial(1, 0.01 if mac!="GK" else 0.005)

    if mac=="GK":
        shots_faced = N(BASE[comp]["GK"]["shots_faced"])
        saves = clamp(int(np.random.normal(*BASE[comp]["GK"]["saves"])), 0, shots_faced)
        gc = clamp(shots_faced - saves, 0, shots_faced)
        cs = 1 if gc==0 else 0
        save_pct = round(100*saves/max(1,shots_faced),1)
        pen_faced = np.random.binomial(1, 0.05)
        pen_saved = np.random.binomial(pen_faced, 0.25)
    else:
        shots_faced = saves = gc = cs = 0
        save_pct = 0.0; pen_faced = pen_saved = 0

    yc = int(yc_override); rc = int(rc_override)
    assists = np.random.binomial(1, 0.3 if goals>0 else 0.1)
    rating = round(np.random.uniform(5.5, 7.5) + 0.3*goals + 0.1*assists - 0.5*rc - 0.2*min(yc,2),1)
    rating = clamp(rating, 3.0, 10.0)

    return {
        "minutes_played": minutes_played, "rating": rating,
        "shots_total": shots_total, "shots_on_target": shots_on, "shots_off_target": shots_off, "shots_blocked": shots_blocked,
        "goals": goals, "expected_goals": xg, "big_chances_created": key_passes//2, "big_chances_missed": max(0, goals-1),
        "penalties_won": np.random.binomial(1, 0.03 if mac in ("MID","FWD") else 0.01),
        "penalties_taken": np.random.binomial(1, 0.08 if mac in ("FWD","MID") else 0.01),
        "penalties_scored": 0, "penalties_missed": 0,
        "dribbles_attempted": dr_a, "dribbles_successful": dr_s, "dribble_success_rate": round(100*dr_s/max(1,dr_a),1),
        "total_passes": total_passes, "passes_completed": passes_completed, "pass_accuracy_pct": pass_acc,
        "key_passes": key_passes, "assists": assists, "expected_assists": round(abs(np.random.normal(0.18 + 0.2*(key_passes>0), 0.18)),2),
        "progressive_passes": prog, "through_balls": through, "crosses_attempted": cross_a, "crosses_completed": cross_c,
        "cross_accuracy_pct": cross_rate, "long_passes_attempted": long_a, "long_passes_completed": long_c,
        "forward_passes": fwd_p, "backward_passes": back_p, "tackles_attempted": tack_a, "tackles_won": tack_w,
        "tackle_success_rate": tack_rate, "interceptions": inter, "clearances": clear, "blocks": blocks,
        "duels_won": duels_w, "duels_lost": duels_l, "duel_success_rate": duel_rate,
        "aerial_duels_won": aer_w, "aerial_duels_lost": aer_l, "aerial_success_rate": aer_rate,
        "pressures": press, "ball_recoveries": recov, "errors_leading_to_shot": err_shot,
        "saves": saves, "shots_faced": shots_faced, "goals_conceded": gc, "clean_sheet": cs, "save_pct": save_pct,
        "penalties_faced": pen_faced, "penalties_saved": pen_saved,
        "yellow_cards": yc, "red_cards": rc,
        "fouls_committed": max(0, int(np.random.normal(1.2 if mac!="GK" else 0.2, 0.8))),
        "fouls_suffered":  max(0, int(np.random.normal(1.0 if mac!="GK" else 0.2, 0.7))),
        "offsides": max(0, int(np.random.normal(0.8 if mac=="FWD" else 0.2, 0.6))),
        "handballs": np.random.binomial(1, 0.01),
        "own_goals": np.random.binomial(1, 0.005 if mac in ("DEF","GK") else 0.002),
        "distance_covered_km": round((10.6 if mac in ("MID","FWD") else 9.8 if mac=="DEF" else 5.2) * (minutes_played/95.0) * np.random.uniform(0.9,1.1),2),
        "sprints": max(0, int(np.random.normal(22 if mac in ("MID","FWD") else 14, 6) * (minutes_played/95.0))),
        "accelerations": max(0, int(np.random.normal(34 if mac in ("MID","FWD") else 18, 9) * (minutes_played/95.0))),
        "top_speed_kmh": round(np.random.uniform(28,36) if mac!="GK" else np.random.uniform(23,29),1),
        "average_speed_kmh": clamp(round((9.0 if mac!="GK" else 7.0),2),5.0,11.5)
    }

# -------------------- Génération + Rapport --------------------
rows=[]; favori_prob=0.5; match_idx=1
report_rows=[]

for Y in range(START_YEAR, END_YEAR+1):
    fr_fix = schedule_friendlies(Y)
    l1_fix = schedule_season(Y)
    matches = [(d,"Amical",opp,wh) for d,opp,wh in fr_fix] + [(d,"Ligue 1",opp,wh) for d,opp,wh in l1_fix]
    matches.sort(key=lambda x:x[0])
    # Lissage ≤ 4 jours
    for i in range(1,len(matches)):
        if (matches[i][0]-matches[i-1][0]).days>4:
            matches[i]=(matches[i-1][0]+timedelta(days=np.random.randint(3,5)),matches[i][1],matches[i][2],matches[i][3])

    for md, comp, opponent, where in matches:
        is_home=(where=="HOME"); is_fav=(favori_prob>0.5)
        # (Outcome "intentionnel" : on l'utilise comme signal, mais on recalera ensuite au score)
        _ = compute_outcome(is_home,is_fav)

        contracted=players_under_contract(md)
        XI=pick_lineup(contracted)
        bench=contracted[~contracted["player_id"].isin(XI["player_id"])]

        xi_macros = XI["macro"].value_counts().to_dict()
        formation_ok = (xi_macros.get("GK",0)==1 and xi_macros.get("DEF",0)>=4 and xi_macros.get("MID",0)>=4 and xi_macros.get("FWD",0)>=2)

        minutes_rows, duration, subs_used, subs_min_time = assign_minutes_and_cards(comp, XI, bench)

        meteo = meteo_by_month(md.month)
        match_player_rows = []
        reds_minutes_loss = 0
        yellows=0; reds=0
        n_sub_entries=0

        for r, start_m, end_m, dur, yc, rc, is_sub in minutes_rows:
            minutes_played=end_m-start_m
            mac=r["macro"]
            stats=gen_stats_row(comp, mac, minutes_played, "?", yc, rc)
            match_player_rows.append({
                "match_id": f"M{match_idx:06d}",
                "date": md.strftime("%Y-%m-%d"),
                "lieux": "Domicile" if is_home else "Exterieur",
                "competition": comp,
                # placeholders score/result ajoutés après calcul
                "result": "",
                "goals_for": 0,
                "goals_against": 0,
                "score": "",
                "meteo": meteo,
                "adversaire": opponent,
                "favoris ?": "Oui" if is_fav else "Non",
                "player_id": r["player_id"],
                "nom joueur": r["nom joueur"],
                "prénom joueur": r["prénom joueur"],
                "position joueur": r["poste"],
                **stats
            })
            yellows += int(yc); reds += int(rc)
            if rc==1: reds_minutes_loss += (duration - end_m)
            if is_sub: n_sub_entries += 1

        # ---- Score du match ----
        goals_for = int(sum(r["goals"] for r in match_player_rows))
        base_lambda = 1.1 if is_home else 1.4
        base_lambda = (max(0.6, base_lambda - 0.15) if (favori_prob>0.5) else min(2.0, base_lambda + 0.10))
        goals_against = int(np.random.poisson(base_lambda))
        if goals_for > goals_against: final_outcome = "V"
        elif goals_for == goals_against: final_outcome = "N"
        else: final_outcome = "D"
        # mise à jour de la proba "favori" sur base du résultat réel
        if final_outcome=="V": favori_prob=min(0.75,favori_prob+0.05)
        elif final_outcome=="D": favori_prob=max(0.25,favori_prob-0.05)

        # Ajuster GK
        gk_idx = [i for i,r in enumerate(match_player_rows) if r["position joueur"]=="GK"]
        if gk_idx:
            gk_minutes = np.array([match_player_rows[i]["minutes_played"] for i in gk_idx], dtype=int)
            weights = (np.ones_like(gk_minutes)/max(1,len(gk_minutes))) if gk_minutes.sum()==0 else (gk_minutes/gk_minutes.sum())
            alloc = np.floor(weights * goals_against).astype(int)
            remainder = goals_against - alloc.sum()
            for k in range(remainder):
                alloc[k % len(alloc)] += 1
            extra_total = int(np.random.poisson(3))
            extra = np.floor(weights * extra_total).astype(int)
            for k in range(extra_total - extra.sum()):
                extra[k % len(extra)] += 1
            for j, idx in enumerate(gk_idx):
                conceded = int(alloc[j])
                shots_faced = int(max(conceded, conceded + extra[j]))
                saves = int(max(0, shots_faced - conceded))
                save_pct = round(100 * saves / max(1, shots_faced), 1)
                clean_sheet = 1 if goals_against == 0 else 0
                match_player_rows[idx]["goals_conceded"] = conceded
                match_player_rows[idx]["shots_faced"]     = shots_faced
                match_player_rows[idx]["saves"]           = saves
                match_player_rows[idx]["save_pct"]        = save_pct
                match_player_rows[idx]["clean_sheet"]     = clean_sheet

        score_str = f"{TEAM_NAME} {goals_for} - {goals_against} {opponent}"
        for rline in match_player_rows:
            rline["result"] = final_outcome
            rline["goals_for"] = goals_for
            rline["goals_against"] = goals_against
            rline["score"] = score_str

        rows.extend(match_player_rows)

        total_minutes = sum(x["minutes_played"] for x in match_player_rows)
        expected_minutes = 11*duration - reds_minutes_loss
        report_rows.append({
            "match_id": f"M{match_idx:06d}",
            "date": md.strftime("%Y-%m-%d"),
            "competition": comp,
            "lieux": "Domicile" if is_home else "Exterieur",
            "adversaire": opponent,
            "favoris ?": "Oui" if is_fav else "Non",
            "duration_min": duration,
            "players_rows": len(match_player_rows),
            "subs_used": subs_used,
            "min_sub_minute": subs_min_time if subs_min_time!="" else "",
            "yellow_cards": yellows,
            "red_cards": reds,
            "total_minutes": total_minutes,
            "expected_minutes": expected_minutes,
            "minutes_ok": bool(total_minutes == expected_minutes),
            "formation_4-4-2_ok": bool(formation_ok),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "result": final_outcome,
            "score": score_str
        })

        match_idx+=1

# ---- Export CSV (ordre des colonnes incluant score/résultat)
cols=[
"match_id","date","lieux","competition","result","goals_for","goals_against","score",
"meteo","adversaire","favoris ?","player_id","nom joueur","prénom joueur","position joueur",
"minutes_played","rating","shots_total","shots_on_target","shots_off_target","shots_blocked","goals","expected_goals","big_chances_created","big_chances_missed",
"penalties_won","penalties_taken","penalties_scored","penalties_missed","dribbles_attempted","dribbles_successful","dribble_success_rate",
"total_passes","passes_completed","pass_accuracy_pct","key_passes","assists","expected_assists","progressive_passes","through_balls","crosses_attempted",
"crosses_completed","cross_accuracy_pct","long_passes_attempted","long_passes_completed","forward_passes","backward_passes","tackles_attempted","tackles_won",
"tackle_success_rate","interceptions","clearances","blocks","duels_won","duels_lost","duel_success_rate","aerial_duels_won","aerial_duels_lost","aerial_success_rate",
"pressures","ball_recoveries","errors_leading_to_shot","saves","shots_faced","goals_conceded","clean_sheet","save_pct","penalties_faced","penalties_saved",
"yellow_cards","red_cards","fouls_committed","fouls_suffered","offsides","handballs","own_goals","distance_covered_km","sprints","accelerations","top_speed_kmh","average_speed_kmh"
]
df=pd.DataFrame(rows).reindex(columns=cols)
df.to_csv(OUT_PATH,index=False,encoding="utf-8")

# ---- Export rapport Excel
rep = pd.DataFrame(report_rows).sort_values(["date","match_id"])
with pd.ExcelWriter(RAPPORT_PATH, engine="openpyxl") as xw:
    rep.to_excel(xw, sheet_name="par_match", index=False)
    rep["year"]=rep["date"].str.slice(0,4).astype(int)
    synth = rep.groupby(["year","competition"], as_index=False).agg(
        matchs=("match_id","nunique"),
        minutes_ok=("minutes_ok","mean"),
        rouges=("red_cards","sum"),
        jaunes=("yellow_cards","sum"),
        subs_total=("subs_used","sum"),
        buts_pour=("goals_for","sum"),
        buts_contre=("goals_against","sum")
    )
    synth["minutes_ok_pct"] = (synth["minutes_ok"]*100).round(1)
    synth.drop(columns=["minutes_ok"], inplace=True)
    synth.to_excel(xw, sheet_name="synthese", index=False)

print(f"✅ Générés :\n- {OUT_PATH}\n- {RAPPORT_PATH}")
PY
