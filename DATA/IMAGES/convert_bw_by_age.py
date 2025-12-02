import os
import csv
import argparse
import unicodedata
from datetime import datetime
from PIL import Image


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    # remove diacritics
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return s


def load_liste_personnes(path):
    # returns mapping filename -> (prenom, nom)
    mapping = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('Nom du fichier') or row.get('Nom du fichier')
            prenom = row.get('Prénom') or ''
            nom = row.get('Nom') or ''
            if fname:
                mapping[fname] = (prenom, nom)
    return mapping


def load_birthdates_from_histo(path):
    # returns mapping normalized (prenom, nom) -> birth_year (int)
    m = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prenom = row.get('Prénom Joueur') or row.get('Prénom') or ''
            nom = row.get('Nom Joueur') or row.get('Nom') or ''
            dob = row.get('Date de naissance joueur') or row.get('Date de naissance') or ''
            if not dob:
                continue
            # try parse year
            year = None
            try:
                # expect YYYY-MM-DD
                year = int(dob.split('-')[0])
            except Exception:
                try:
                    dt = datetime.strptime(dob, '%d/%m/%Y')
                    year = dt.year
                except Exception:
                    continue

            key = (normalize_text(prenom), normalize_text(nom))
            # keep earliest available mapping if multiple rows exist
            if key not in m:
                m[key] = year
    return m


def convert_to_bw(path_in, path_out=None, overwrite=True):
    if path_out is None:
        path_out = path_in
    try:
        with Image.open(path_in) as im:
            has_alpha = im.mode in ('RGBA', 'LA') or ('transparency' in im.info)
            if has_alpha:
                # split alpha, convert RGB to L and merge with alpha
                if im.mode != 'RGBA':
                    im = im.convert('RGBA')
                rgb, alpha = im.convert('RGB'), im.split()[-1]
                l = rgb.convert('L')
                la = Image.merge('LA', (l, alpha))
                la.save(path_out)
            else:
                l = im.convert('L')
                l.save(path_out)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Convert images to B/W when player's birth year < 1990")
    default_images = os.path.join(os.path.dirname(__file__), 'Visages_standard')
    default_csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CSV'))
    parser.add_argument('-i', '--images', default=default_images, help='Folder with standardized images')
    parser.add_argument('-c', '--csvdir', default=default_csv_dir, help='Folder containing CSV files')
    parser.add_argument('--dry-run', action='store_true', help='Do not write files, just report')
    parser.add_argument('--before-year', type=int, default=1990, help='Year threshold (born before -> convert)')
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images)
    csv_dir = os.path.abspath(args.csvdir)

    liste_path = os.path.join(csv_dir, 'liste_personnes.csv')
    histo_path = os.path.join(csv_dir, 'histo_contrats.csv')

    if not os.path.exists(images_dir):
        print(f'Images folder not found: {images_dir}')
        return
    if not os.path.exists(liste_path):
        print(f'File not found: {liste_path}')
        return
    if not os.path.exists(histo_path):
        print(f'File not found: {histo_path}')
        return

    print('Loading mappings...')
    file_to_name = load_liste_personnes(liste_path)
    birth_map = load_birthdates_from_histo(histo_path)

    processed = 0
    converted = 0
    skipped_no_match = 0
    errors = 0

    for entry in sorted(os.listdir(images_dir)):
        path = os.path.join(images_dir, entry)
        if not os.path.isfile(path):
            continue
        # find in liste_personnes mapping
        if entry not in file_to_name:
            # try with normalized matching by filename
            found = False
            for fname in file_to_name:
                if normalize_text(fname) == normalize_text(entry):
                    prenom, nom = file_to_name[fname]
                    found = True
                    break
            if not found:
                print(f'No mapping for image: {entry} — skipping')
                skipped_no_match += 1
                continue
        else:
            prenom, nom = file_to_name[entry]

        key = (normalize_text(prenom), normalize_text(nom))
        birth_year = birth_map.get(key)
        processed += 1
        if birth_year is None:
            print(f'No birthdate for {prenom} {nom} ({entry}) — skipping')
            skipped_no_match += 1
            continue

        if birth_year < args.before_year:
            print(f'Converting to B/W: {entry} (born {birth_year})')
            if args.dry_run:
                converted += 1
                continue
            ok, err = convert_to_bw(path, path_out=path, overwrite=True)
            if ok:
                converted += 1
            else:
                errors += 1
                print(f'Error converting {entry}: {err}')
        else:
            print(f'Keep color: {entry} (born {birth_year})')

    print('\nSummary:')
    print(f'  Images processed: {processed}')
    print(f'  Converted to B/W: {converted}')
    print(f'  Skipped (no mapping/birth): {skipped_no_match}')
    print(f'  Errors: {errors}')


if __name__ == '__main__':
    main()
