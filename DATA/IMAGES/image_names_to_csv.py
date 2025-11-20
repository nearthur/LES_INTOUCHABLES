import os
import csv
import re
import argparse
import os.path

# Chemin vers le dossier des visages (calculé relativement à ce script)
DOSSIER_VISAGE = os.path.abspath(os.path.join(os.path.dirname(__file__), "Visages générés"))

def extraire_noms_prenoms(nom_fichier):
    """
    Extrait le prénom et le nom d'un fichier image.
    Gère différents formats de noms de fichiers.
    """
    # Enlever l'extension
    nom_sans_ext = os.path.splitext(nom_fichier)[0]
    
    # Séparer par espace ou tiret
    parties = re.split(r'[\s-]+', nom_sans_ext)
    
    if len(parties) >= 2:
        prenom = parties[0]
        nom = ' '.join(parties[1:])
        return prenom, nom
    elif len(parties) == 1:
        return parties[0], ""
    else:
        return "", ""

def creer_csv_depuis_images(dossier_images, fichier_csv_sortie):
    """
    Parcourt un dossier d'images et crée un fichier CSV avec prénom, nom et nom de fichier.
    """
    # Extensions d'images acceptées
    extensions_images = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')
    
    # Liste pour stocker les données
    donnees = []
    
    # Parcourir les fichiers du dossier
    dossier_images = os.path.abspath(dossier_images)
    if os.path.exists(dossier_images):
        for entry in sorted(os.scandir(dossier_images), key=lambda e: e.name):
            if entry.is_file():
                _, ext = os.path.splitext(entry.name)
                if ext.lower() in extensions_images:
                    prenom, nom = extraire_noms_prenoms(entry.name)
                    donnees.append({
                        'Prénom': prenom,
                        'Nom': nom,
                        'Nom du fichier': entry.name
                    })
    
    # Créer le fichier CSV
    # S'assurer que le dossier de sortie existe
    sortie_dir = os.path.dirname(os.path.abspath(fichier_csv_sortie))
    if sortie_dir and not os.path.exists(sortie_dir):
        os.makedirs(sortie_dir, exist_ok=True)

    with open(fichier_csv_sortie, 'w', newline='', encoding='utf-8-sig') as csvfile:
        colonnes = ['Prénom', 'Nom', 'Nom du fichier']
        writer = csv.DictWriter(csvfile, fieldnames=colonnes)
        
        writer.writeheader()
        writer.writerows(donnees)
    
    print(f"✓ Fichier CSV créé : {fichier_csv_sortie}")
    print(f"✓ Nombre d'images traitées : {len(donnees)}")

# Exemple d'utilisation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un CSV à partir des noms d'images d'un dossier")
    parser.add_argument('-d', '--dossier', default=DOSSIER_VISAGE,
                        help='Chemin vers le dossier d\'images (défaut: Visages générés à côté du script)')
    # Défaut demandé: enregistrer le CSV dans DATA/CSV du dépôt
    default_out = os.path.normpath(r"C:\Users\defaultuser0.LOIC\.vscode\LES_INTOUCHABLES\DATA\CSV\liste_personnes.csv")
    parser.add_argument('-o', '--output', default=default_out,
                        help='Chemin du fichier CSV de sortie (défaut: liste_personnes.csv à la racine du repo)')

    args = parser.parse_args()

    print(args.dossier)
    creer_csv_depuis_images(args.dossier, args.output)
