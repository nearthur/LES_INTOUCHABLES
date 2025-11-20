# Installation et exécution (PowerShell)

1. Créer un environnement virtuel (optionnel mais recommandé)

```powershell
python -m venv .venv
```

2. Activer l'environnement (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Installer les dépendances (le fichier `requirements.txt` est vide par défaut)

```powershell
pip install -r requirements.txt
```

4. (Optionnel) Installer les dépendances de développement

```powershell
pip install -r dev-requirements.txt
```

4. Exécuter le script pour générer le CSV

```powershell
python .\DATA\IMAGES\image_names_to_csv.py
```

Options utiles:

- Spécifier un dossier d'images différent:

```powershell
python .\DATA\IMAGES\image_names_to_csv.py -d ".\DATA\IMAGES\Visages générés"
```

- Spécifier un emplacement de sortie pour le CSV:

```powershell
python .\DATA\IMAGES\image_names_to_csv.py -o ".\liste_personnes.csv"
```

Remarques:
 - Le script utilise uniquement la librairie standard de Python; `requirements.txt` est donc vide par défaut.
 - Le CSV de sortie par défaut est `liste_personnes.csv` placé à la racine du dépôt.
 - Pour formatter le code et exécuter les tests, installez les dépendances de développement via `dev-requirements.txt`.
