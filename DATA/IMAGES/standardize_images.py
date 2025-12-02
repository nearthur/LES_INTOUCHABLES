import argparse
import os
from PIL import Image


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    return img.crop((left, top, left + min_side, top + min_side))


def process_image(path_in: str, path_out: str, size: int, crop: bool, quality: int, out_format: str):
    with Image.open(path_in) as im:
        # convert to RGB to avoid problems with palettes / alpha when saving as JPEG
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")

        if crop:
            im = center_crop_to_square(im)

        im = im.resize((size, size), Image.LANCZOS)

        save_kwargs = {}
        fmt = out_format or im.format or "PNG"
        if fmt.upper() in ("JPEG", "JPG"):
            save_kwargs["quality"] = quality
            # ensure mode is RGB for JPEG
            if im.mode == "RGBA":
                im = im.convert("RGB")

        # create parent dir if needed
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        im.save(path_out, format=fmt)


def standardize_folder(input_dir: str, output_dir: str, size: int, crop: bool, quality: int, out_format: str, overwrite: bool):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    supported_ext = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

    if not os.path.exists(input_dir):
        print(f"Input folder not found: {input_dir}")
        return

    files = [f for f in sorted(os.listdir(input_dir))]
    count = 0
    for name in files:
        src = os.path.join(input_dir, name)
        if not os.path.isfile(src):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() not in supported_ext:
            continue

        base_name = os.path.splitext(name)[0]
        out_ext = "." + (out_format.lower() if out_format else ext.lstrip("."))
        dst_name = base_name + out_ext
        dst = os.path.join(output_dir, dst_name)

        if os.path.exists(dst) and not overwrite:
            print(f"Skipping (exists): {dst}")
            continue

        try:
            process_image(src, dst, size, crop, quality, out_format)
            count += 1
            print(f"Processed: {src} -> {dst}")
        except Exception as e:
            print(f"Failed: {src} ({e})")

    print(f"\nDone. Processed {count} images. Output folder: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Standardize images: crop/resize and save to output folder")
    parser.add_argument("-d", "--dossier", default=os.path.join(os.path.dirname(__file__), "Visages générés"),
                        help="Dossier source d'images (défaut: Visages générés à côté du script)")
    parser.add_argument("-o", "--output", default=os.path.join(os.path.dirname(__file__), "Visages_standard"),
                        help="Dossier de sortie (défaut: DATA/IMAGES/Visages_standard)")
    parser.add_argument("-s", "--size", type=int, default=256, help="Taille (pixels) des images de sortie (carrées)")
    parser.add_argument("--no-crop", dest="crop", action="store_false", help="Ne pas recadrer au centre avant redimensionnement")
    parser.add_argument("-q", "--quality", type=int, default=90, help="Qualité pour JPEG (1-100)")
    parser.add_argument("-f", "--format", default=None, help="Format de sortie (JPEG, PNG, ...). Par défaut conserve l'extension")
    parser.add_argument("--overwrite", action="store_true", help="Écrase les fichiers existants dans le dossier de sortie")

    args = parser.parse_args()

    standardize_folder(args.dossier, args.output, args.size, args.crop, args.quality, args.format, args.overwrite)


if __name__ == "__main__":
    main()
