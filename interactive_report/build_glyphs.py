"""Glyph data for the interactive report (AI-generated helper).

The report (`index.html`) embeds two things from Omniglot: 6 + 6 PNG data URIs (the examples
shown) of 40x40 glyphs for display (`IMG`), and 12 + 12 arrays of 12x12 ink values for the toy
network (`PIX`). This script derives both from the dataset the main code downloads to
`./data` (run `uv run src/run_omniglot.py max_n_tasks=1 n_epochs_progress=1
n_epochs_compress=1 hydra.run.dir=outputs/tmp` once from the repository root).

Task A = the first 6 characters of Oriya, task B = the first 6 of Keble, 2 samples each.
Display glyphs: resized to 40x40 (Lanczos), PNG, base64. Network inputs: the glyph
resized to 12x12 with a box filter, inverted so that ink is high, each image scaled so
that its largest value is 255, written as 144 hex pairs.

The page does not read these files at run time; the builder of the page inlined the
same data (its 12x12 values differ from this script's by a few grey levels because of
the resampling kernel). Use this script to rebuild the data for other alphabets.

Usage, from the repository root:
    uv run interactive_report/build_glyphs.py --out interactive_report/drafts
"""

import argparse
import base64
import io
import json
import os

from PIL import Image

ROOTS = ["data/omniglot-py/images_background", "data/omniglot-py/images_evaluation"]
TASKS = {"A": "Oriya", "B": "Keble"}


def find_alphabet(name: str) -> str:
    for root in ROOTS:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    raise SystemExit(f"{name} not found under {ROOTS}; download Omniglot first")


def glyph_files(alphabet: str, n_chars: int, n_samples: int) -> list[str]:
    files = []
    for ch in sorted(os.listdir(alphabet))[:n_chars]:
        for f in sorted(os.listdir(os.path.join(alphabet, ch)))[:n_samples]:
            files.append(os.path.join(alphabet, ch, f))
    return files


def to_data_uri(im: Image.Image, size: int) -> str:
    buf = io.BytesIO()
    im.resize((size, size), Image.LANCZOS).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_hex_pixels(im: Image.Image, size: int) -> str:
    small = im.resize((size, size), Image.BOX)
    ink = [255 - p for p in small.getdata()]  # [size * size], ink is high
    peak = max(ink) or 1
    return "".join(f"{round(v * 255 / peak):02x}" for v in ink)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="interactive_report/drafts")
    parser.add_argument("--n-chars", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--display-size", type=int, default=40)
    parser.add_argument("--input-size", type=int, default=12)
    args = parser.parse_args()

    img, pix = {}, {}
    for key, name in TASKS.items():
        files = glyph_files(find_alphabet(name), args.n_chars, args.n_samples)
        images = [Image.open(f).convert("L") for f in files]
        img[key] = [to_data_uri(im, args.display_size) for im in images]
        pix[key] = [to_hex_pixels(im, args.input_size) for im in images]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "glyphs_display.json"), "w") as f:
        json.dump(img, f)
    with open(os.path.join(args.out, "glyphs_12x12.json"), "w") as f:
        json.dump(pix, f)
    print(
        os.path.join(args.out, "glyphs_display.json"),
        {k: len(v) for k, v in img.items()},
    )
    print(
        os.path.join(args.out, "glyphs_12x12.json"), {k: len(v) for k, v in pix.items()}
    )


if __name__ == "__main__":
    main()
