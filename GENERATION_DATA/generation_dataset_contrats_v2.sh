#!/usr/bin/env bash
# Générateur d'historiques de contrats joueurs (CSV)
# ✓ 30–40 joueurs actifs en simultané dès 1970 et ensuite (paramétrable)
# ✓ Paramètre --total-players : nombre total d'IDs uniques sur toute l'histoire
# ✓ Anti-fratrie (évite mêmes noms de famille en même temps, 98%)
# ✓ Transferts sortants + retours rares
# ✓ Cohérences: âge/naissance, poste voisin, nationalité & agent stables, pied fort, taille/poids, salaire multi-facteurs

set -euo pipefail

OUT=""
MIN_ROSTER=30
MAX_ROSTER=40
TOTAL_PLAYERS=0       # 0 => auto (pas de plafond), sinon plafond d'IDs uniques
SEED=42
CLUB_DATE="1970-08-12"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out) OUT="$2"; shift 2;;
    --min-roster) MIN_ROSTER="$2"; shift 2;;
    --max-roster) MAX_ROSTER="$2"; shift 2;;
    --total-players) TOTAL_PLAYERS="$2"; shift 2;;
    -s|--seed) SEED="$2"; shift 2;;
    -c|--club-date) CLUB_DATE="$2"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0;;
    *) echo "Argument inconnu: $1" >&2; exit 1;;
  esac
done

if [[ -z "${OUT}" ]]; then
  echo "Erreur: spécifie un fichier de sortie avec -o/--out" >&2
  exit 1
fi

python3 - <<'PY' "$OUT" "$MIN_ROSTER" "$MAX_ROSTER" "$TOTAL_PLAYERS" "$SEED" "$CLUB_DATE"
import sys, random, math
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np

# -------- Args --------
OUT = sys.argv[1]
MIN_ROSTER = int(sys.argv[2])
MAX_ROSTER = int(sys.argv[3])
TOTAL_PLAYERS_LIMIT = int(sys.argv[4])  # 0 => no cap
SEED = int(sys.argv[5])
CLUB_CREATION_STR = sys.argv[6]

random.seed(SEED); np.random.seed(SEED)
CLUB_CREATION = datetime.strptime(CLUB_CREATION_STR, "%Y-%m-%d")
TODAY = date.today()

assert 20 <= MIN_ROSTER <= MAX_ROSTER <= 50, "bornes roster incohérentes"
if TOTAL_PLAYERS_LIMIT and TOTAL_PLAYERS_LIMIT < MIN_ROSTER:
    raise SystemExit("--total-players doit être >= --min-roster")

# --- Postes & voisinages ---
ALL_POS = ["AD","BU","AG","MC","MOC","MD","MG","MDC","DD","DG","DC","GK"]
NEIGHBORS = {
    "GK":["GK"],
    "DC":["DC","DD","DG","MDC"],
    "DD":["DD","DC","MD"],
    "DG":["DG","DC","MG"],
    "MC":["MC","MDC","MOC","MD","MG"],
    "MOC":["MOC","MC","BU"],
    "MDC":["MDC","MC","DC"],
    "MD":["MD","MC","DD","AD"],
    "MG":["MG","MC","DG","AG"],
    "BU":["BU","MOC","AD","AG"],
    "AD":["AD","BU","MD"],
    "AG":["AG","BU","MG"],
}

MACRO_MIN = {"GK":2,"DEF":8,"MID":8,"FWD":5}
SUB_BY_MACRO = {
    "GK":["GK"],
    "DEF":["DD","DG","DC"],
    "MID":["MC","MOC","MDC","MD","MG"],
    "FWD":["BU","AD","AG"]
}

# Nationalités + bonus salaire
NATS = [
    ("France",0.35),("Spain",0.08),("England",0.08),("Brazil",0.09),("Argentina",0.06),
    ("Germany",0.06),("Italy",0.06),("Portugal",0.06),("Netherlands",0.04),("Belgium",0.04),
    ("Morocco",0.03),("Algeria",0.03),("Senegal",0.02)
]
NAT_LABELS, NAT_PROBS = zip(*NATS)
NAT_BONUS = {"Brazil":1.05,"Argentina":1.03,"France":1.02,"England":1.06,"Germany":1.04,"Italy":1.04,
             "Portugal":1.03,"Spain":1.05,"Netherlands":1.02,"Belgium":1.01,"Morocco":1.01,"Algeria":1.01,"Senegal":1.01}

# Gros générateur de noms (pools + syllabes)
FIRST_BASE = [
 "Arthur","Hugo","Kylian","Karim","Antoine","Ousmane","Kingsley","Olivier","Benjamin","Raphaël","Nicolás","Pedro","João",
 "Luca","Marco","Andrea","Matteo","Sergio","Raúl","Iker","Álvaro","Tiago","Ruben","Diogo","Luis","Miguel","André",
 "Tom","Nathan","Yanis","Maxime","Baptiste","Clément","Alexandre","Paul","Enzo","Noah","Ethan","Sacha",
 "Jules","Aurélien","Adrien","Warren","Alexis","Jonathan","Yann","Loïs","Brice","Alban","Nabil","Youssef","Achraf",
 "Sadio","Kalidou","Ismaïla","Boulaye","Abdou","Amine","Riyad","Hakim","Romain","Quentin","Milan","Elyes","Mehdi"
]
LAST_BASE = [
 "Dupont","Martin","Bernard","Giroud","Pavard","Varane","Hernandez","Maignan","Camavinga","Konaté","Upamecano","Nkunku",
 "Thuram","Koundé","Tchouaméni","Rabiot","Sanchez","Diop","Lacazette","Gouiri","Fofana","Saliba","Lopez","Silva","Costa",
 "Pereira","Gonçalves","Fernandes","Mendes","Rossi","Esposito","Romano","Bianchi","Conti","Ricci","Lombardi",
 "García","Fernández","Martínez","Sánchez","Pérez","Gómez","Rodríguez","De Jong","Van Dijk","De Bruyne","Lukaku",
 "Mahrez","Ounahi","Boufal","Brahimi","Ziyech","Bentaleb","Mandi","Sarr","Mané","Koulibaly","Ndiaye","Diatta","Benrahma"
]
SYL_FIRST = ["Ma","Lu","Ka","An","O","Ki","Sa","Na","Theo","Ari","Yo","Mi","Ra","Di","Ju","Au","Ad","Lo","Ba","Quen","Wor","Al","Ri","Ta","Ni","El","Me","Mo"]
SYL_LAST  = ["-son","-sen","-ez","-ov","-ić","-ini","-etti","-aldi","-escu","-idis","-opoulos","-ian","-mano","-eiro","-eira","-azzi","-ovic","-sky","-ski"]

def make_random_name():
    if random.random() < 0.7:
        fn = random.choice(FIRST_BASE)
        ln = random.choice(LAST_BASE)
    else:
        fn = random.choice(SYL_FIRST) + random.choice(["","a","o","e"]) + random.choice(SYL_FIRST).lower()
        ln = random.choice(LAST_BASE) + random.choice(SYL_LAST)
        ln = ln.replace("--","-").replace("  "," ")
    return fn, ln

# Agents
AG_FIRST = ["Pierre","Jean","Marco","Miguel","Jorge","Mino","Pini","Sacha","Matthias","Federico","Hassan","Yusef","Andrea",
            "Farid","Daniel","David","Luis","Francesco","Karim","Samir","Rodolfo","Sofian","Ramos","Santos"]
AG_LAST  = ["De la Croix","Mendes","Raiola","Zahavi","Barnett","Sissoko","Antunes","Bozzo","Iaque","Bolarinwa","Ferreira","Rosell",
            "Campos","Ortega","Giacomo","Silva","Costa","Benchetrit","Morales","Zeroual","Brahimi","Mendes Filho","Ouedraogo"]
AG_NAT   = ["France","Portugal","Italy","Israel","UK","France","Portugal","Italy","Italy","Nigeria","Brazil","Spain",
            "Portugal","Spain","Italy","Portugal","Portugal","France","Spain","Morocco","Algeria","Portugal","Burkina Faso"]
AGENTS = [{"nom_agent": AG_LAST[i%len(AG_LAST)], "prenom_agent": AG_FIRST[i%len(AG_FIRST)], "nationalite_agent": AG_NAT[i%len(AG_NAT)]} for i in range(50)]

def height_by_macro(m): return int(np.random.normal({"GK":191,"DEF":184,"MID":178,"FWD":180}[m], 5))
def choose_macro_distribution(n):
    base = MACRO_MIN.copy()
    extra = n - sum(base.values())
    probs = [0.08,0.36,0.36,0.20]  # GK, DEF, MID, FWD
    for _ in range(extra):
        base[np.random.choice(["GK","DEF","MID","FWD"], p=probs)] += 1
    return base

def choose_role():
    r = np.random.rand()
    if r < 1/5: return "Titulaire indiscutable"
    if r < 1/5 + 1/3: return "Titulaire"
    if r < 1/5 + 1/3 + 1/3: return "Remplaçant"
    return "Reserviste"

def salary_model(age, tenure_years, nat, role, geste, mauvais, duration):
    age_peak = 28
    age_score = math.exp(-((age - age_peak)**2) / (2*6.0**2))
    tenure_score = math.log1p(max(0, tenure_years)) / math.log(11)
    role_mult = {"Titulaire indiscutable":1.25,"Titulaire":1.10,"Remplaçant":0.90,"Reserviste":0.75}[role]
    skill_score = 0.6*(geste/5) + 0.4*(mauvais/5)
    dur_mult = {1:0.95,2:1.00,3:1.05,4:1.08,5:1.12}[duration]
    nat_mult = NAT_BONUS.get(nat,1.0)
    base, top = 100_000, 3_000_000
    score = 0.55*age_score + 0.25*tenure_score + 0.20*skill_score
    amount = (base + score*(top-base)*0.85) * role_mult * dur_mult * nat_mult
    return int(max(base, min(top, round(amount, -3))))

def evolve_skill(x): return int(np.clip(x + np.random.choice([-1,0,1], p=[0.03,0.89,0.08]), 1, 5))
def next_position(curr): return np.random.choice(NEIGHBORS[curr]) if np.random.rand() < 0.12 else curr

# Anti-fratrie helpers
def active_lastnames_on(day, rows):
    ln = set()
    for c in rows:
        s = datetime.strptime(c["Date de signature"], "%Y-%m-%d").date()
        e = datetime.strptime(c["Date de fin"], "%Y-%m-%d").date()
        if s <= day <= e: ln.add(c["Nom Joueur"])
    return ln

def active_count_on(day, rows):
    return sum(1 for c in rows if datetime.strptime(c["Date de signature"], "%Y-%m-%d").date() <= day <= datetime.strptime(c["Date de fin"], "%Y-%m-%d").date())

# Player factories
def make_player(pid, macro, start_year_for_age):
    dob_year = np.random.randint(start_year_for_age-33, start_year_for_age-16+1)
    dob = datetime(dob_year, np.random.randint(1,13), np.random.randint(1,28))
    nat = np.random.choice(NAT_LABELS, p=NAT_PROBS)
    pied = np.random.choice(["D","G"], p=[0.78,0.22])
    h = height_by_macro(macro)
    bmi = np.random.normal(22.5,1.5)
    w = int(round((h/100)**2 * bmi))
    sub = np.random.choice(SUB_BY_MACRO[macro])
    geste = np.random.randint(2,5); mauvais = np.random.randint(1,4)
    fn, ln = make_random_name()
    ag = random.choice(AGENTS)
    return {"player_id": f"P{pid:05d}","prenom": fn,"nom": ln,"nationalite": nat,"pied": pied,
            "taille_cm": h,"poids_kg": w,"dob": dob,"poste": sub,"macro": macro,
            "geste": geste,"mauvais": mauvais,"agent": ag}

def add_contract(rows, p, start, end, duration, tenure, pos, geste, mauvais, role):
    age = (start.date() - p["dob"].date()).days // 365
    salary = salary_model(age, tenure, p["nationalite"], role, geste, mauvais, duration)
    ag = p["agent"]
    rows.append({
        "Type de Contrat":"CDD Sportif",
        "Date de signature": start.strftime("%Y-%m-%d"),
        "Date de fin": end.strftime("%Y-%m-%d"),
        "Durée Contrat (ans)": duration,
        "Salaire annuel (€)": salary,
        "Nom Joueur": p["nom"], "Prénom Joueur": p["prenom"],
        "Date de naissance joueur": p["dob"].strftime("%Y-%m-%d"),
        "Age Joueur (à la signature)": age,
        "Nationalité Joueur": p["nationalite"], "Poste Joueur": pos, "Pied Fort": p["pied"],
        "Rôle dans l'équipe": role,
        "Niveau geste technique (1-5)": geste, "Niveau mauvais pied (1-5)": mauvais,
        "Taille Joueur (cm)": p["taille_cm"], "Poids Joueur (kg)": p["poids_kg"],
        "Nom Agent": ag["nom_agent"], "Prénom Agent": ag["prenom_agent"], "Nationalité Agent": ag["nationalite_agent"],
        "Ancienneté au club (ans)": round(tenure,1), "Player ID": p["player_id"]
    })

# Carrière au club (avec départs, retours rares)
def build_career(rows, p, start_first=None):
    career_end = min(p["dob"] + timedelta(days=int(365*np.random.uniform(34,40))),
                     datetime(TODAY.year, TODAY.month, TODAY.day))
    start = start_first or (CLUB_CREATION + timedelta(days=np.random.randint(0,180)))
    duration = int(np.random.choice([1,2,3,4,5], p=[0.25,0.25,0.22,0.18,0.10]))
    end = start + timedelta(days=365*duration + np.random.randint(-20,20))
    tenure = 0.0
    pos, geste, mauvais = p["poste"], p["geste"], p["mauvais"]
    while start < career_end:
        role = choose_role()
        add_contract(rows, p, start, end, duration, tenure, pos, geste, mauvais, role)
        tenure += duration
        # départ ?
        base_leave = 0.30
        age_next = ((end.date() - p["dob"].date()).days // 365) + 1
        base_leave += 0.10 if age_next >= 30 else -0.05
        base_leave += -0.08 if role=="Titulaire indiscutable" else 0.05 if role=="Reserviste" else 0.0
        base_leave += -0.08 if tenure >= 6 else 0.0
        if np.random.rand() < np.clip(base_leave, 0.05, 0.75):
            # rare retour
            if np.random.rand() < 0.10:
                start = end + timedelta(days=np.random.randint(2*365, 5*365+1))
                duration = int(np.random.choice([1,2,3,4,5], p=[0.25,0.25,0.22,0.18,0.10]))
                end = start + timedelta(days=365*duration + np.random.randint(-20,20))
                pos = next_position(pos); geste = evolve_skill(geste); mauvais = evolve_skill(mauvais)
                continue
            else:
                break
        # prolongation
        start = end + timedelta(days=np.random.randint(0,60))
        duration = int(np.random.choice([1,2,3,4,5], p=[0.25,0.25,0.22,0.18,0.10]))
        end = start + timedelta(days=365*duration + np.random.randint(-20,20))
        pos = next_position(pos); geste = evolve_skill(geste); mauvais = evolve_skill(mauvais)
        if end > career_end: end = career_end

# --- Moteur principal avec plafond d'IDs uniques ---
rows = []
pid_counter = 1
unique_created = 0
alumni = []  # ex-joueurs qu'on peut faire revenir (même Player ID)

# 1) Roster initial (entre MIN et MAX) en 1970, anti-fratrie
initial_size = np.random.randint(MIN_ROSTER, MAX_ROSTER+1)
macro = MACRO_MIN.copy()
extra = initial_size - sum(macro.values())
for _ in range(extra):
    macro[np.random.choice(["GK","DEF","MID","FWD"], p=[0.08,0.36,0.36,0.20])] += 1

active_ln = set()
for macro_role, cnt in macro.items():
    for _ in range(cnt):
        # respecter plafond d'IDs uniques
        if TOTAL_PLAYERS_LIMIT and unique_created >= TOTAL_PLAYERS_LIMIT:
            break
        tries = 0
        while True:
            p = make_player(pid_counter, macro_role, 1970)
            tries += 1
            if p["nom"] not in active_ln or np.random.rand() < 0.02 or tries > 30:
                break
        build_career(rows, p)  # construit ses périodes au club
        alumni.append(p)
        pid_counter += 1
        unique_created += 1
        active_ln.add(p["nom"])

# 2) Garde-fou annuel : si < MIN_ROSTER, recruter
def active_count_on(day):
    return sum(1 for c in rows if datetime.strptime(c["Date de signature"], "%Y-%m-%d").date() <= day <= datetime.strptime(c["Date de fin"], "%Y-%m-%d").date())

def active_lastnames_on(day):
    lns = set()
    for c in rows:
        s = datetime.strptime(c["Date de signature"], "%Y-%m-%d").date()
        e = datetime.strptime(c["Date de fin"], "%Y-%m-%d").date()
        if s <= day <= e: lns.add(c["Nom Joueur"])
    return lns

for year in range(1970, TODAY.year+1):
    day = date(year,1,1)
    cur = active_count_on(day)
    if cur >= MIN_ROSTER:
        continue
    need = MIN_ROSTER - cur
    # 2.1 : créer de NOUVEAUX joueurs (si on n'a pas atteint le plafond TOTAL_PLAYERS)
    can_create = 10**9 if TOTAL_PLAYERS_LIMIT==0 else max(0, TOTAL_PLAYERS_LIMIT - unique_created)
    to_create = min(need, can_create)
    for _ in range(to_create):
        macro_pick = np.random.choice(["GK","DEF","MID","FWD"], p=[0.08,0.36,0.36,0.20])
        act_ln = active_lastnames_on(day)
        tries = 0
        while True:
            p = make_player(pid_counter, macro_pick, year)
            tries += 1
            if p["nom"] not in act_ln or np.random.rand() < 0.02 or tries > 30:
                break
        start_first = datetime(year,1,1) + timedelta(days=np.random.randint(0,120))
        build_career(rows, p, start_first=start_first)
        alumni.append(p)
        pid_counter += 1
        unique_created += 1
        need -= 1
        if need <= 0:
            break
    if need <= 0:
        continue
    # 2.2 : plus d'IDs disponibles → faire REVENIR des alumni (pas de nouvel ID)
    # on pioche des anciens au hasard en respectant (autant que possible) l'anti-fratrie
    act_ln = active_lastnames_on(day)
    # mélanger alumni pour diversifier
    pool = alumni[:]
    random.shuffle(pool)
    for p in pool:
        if need <= 0:
            break
        # si son dernier contrat connu finit avant 'day', on peut tenter un retour
        # (simple test : autoriser retour à partir de 'day' avec 0-120j de délai)
        start_first = datetime(year,1,1) + timedelta(days=np.random.randint(0,120))
        # éviter fratrie à 98%
        if p["nom"] in act_ln and np.random.rand() >= 0.02:
            continue
        build_career(rows, p, start_first=start_first)
        act_ln.add(p["nom"])
        need -= 1

# Export
df = (pd.DataFrame(rows)
      .sort_values(["Nom Joueur","Prénom Joueur","Date de signature"])
      .reset_index(drop=True))
df.to_csv(OUT, index=False, encoding="utf-8")

# Log de contrôle
distinct_ids = df["Player ID"].nunique()
# --- Statistiques de contrôle ---
distinct_ids = df["Player ID"].nunique()

# Effectif actif aujourd'hui
today_count = sum(
    1 for c in df.to_dict("records")
    if datetime.strptime(c["Date de signature"], "%Y-%m-%d").date()
       <= TODAY <= datetime.strptime(c["Date de fin"], "%Y-%m-%d").date()
)

# Effectif par année
roster_stats = []
for y in range(CLUB_CREATION.year, TODAY.year+1):
    d = date(y,1,1)
    active = sum(
        1 for c in df.to_dict("records")
        if datetime.strptime(c["Date de signature"], "%Y-%m-%d").date() <= d <= datetime.strptime(c["Date de fin"], "%Y-%m-%d").date()
    )
    roster_stats.append({"year":y,"active_players":active})

df_stats = pd.DataFrame(roster_stats)
min_roster = df_stats["active_players"].min()
max_roster = df_stats["active_players"].max()
mean_roster = df_stats["active_players"].mean()

# Sauvegarde du CSV des stats
df_stats.to_csv("roster_stats.csv", index=False)

# --- Logs console ---
print(f"✅ Dataset généré : {OUT}")
print(f"   {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"   IDs uniques : {distinct_ids}")
print(f"   Actifs aujourd'hui : {today_count}")
print(f"   Roster annuel : min={min_roster}, max={max_roster}, moyenne={mean_roster:.1f}")
print("   ➜ Stats détaillées sauvegardées dans roster_stats.csv")
if TOTAL_PLAYERS_LIMIT:
    print(f"   (Plafond fixé --total-players = {TOTAL_PLAYERS_LIMIT})")



print(f"OK:{OUT}:{df.shape[0]} lignes / {df.shape[1]} colonnes | IDs uniques = {distinct_ids}")
if TOTAL_PLAYERS_LIMIT:
    print(f"(Cible --total-players = {TOTAL_PLAYERS_LIMIT})")
PY

