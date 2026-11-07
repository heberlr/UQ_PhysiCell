"""Build ex10_rule_inactivation.gif from the PhysiCell movies in ./videos.

Usage (from this folder):  python make_rule_inactivation_gif.py
Requires: ffmpeg on PATH, Pillow.

Columns are ordered by the true Tumor:CAF seeding ratio (1:10 -> 10:1).
Note the IC files are mislabeled in the model construction (see ex10 notebook):
cells_1_to_2.csv is Tumor:CAF 2:1 and cells_2_to_1.csv is Tumor:CAF 1:2.

Top row: nominal model (all rules active), movie_s{S}r0.mp4.
Bottom row (added only if those movies exist in ./videos): the top-ranked
inactive rule for that IC.
"""
import glob
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(HERE, 'videos')
OUT = os.path.join(HERE, 'ex10_rule_inactivation.gif')

# (label, nominal SampleID, top-ranked inactive SampleID or None, inactive rule name)
COLUMNS = [
    ('1:10', 3, 73, 'Slow down TA-MC'),   # cells_1_to_10.csv
    ('1:5', 2, 86, 'MET Tumor cell'),     # cells_1_to_5.csv
    ('1:2', 4, 67, 'EMT Tumor cell'),     # cells_2_to_1.csv (mislabeled file)
    ('1:1', 0, 56, 'Epi. Tumor CIP'),     # cells_1_to_1.csv
    ('2:1', 1, 57, 'Epi. Tumor CIP'),     # cells_1_to_2.csv (mislabeled file)
    ('5:1', 5, 47, 'Slow down CAF'),      # cells_5_to_1.csv
    ('10:1', 6, 48, 'Slow down CAF'),     # cells_10_to_1.csv
]

CROP = (318, 112, 896, 768)  # title + spatial plot of the 1280x960 frames; drops colorbar/legend
TILE_W = 220
TILE_H = int(TILE_W * (CROP[3] - CROP[1]) / (CROP[2] - CROP[0]))
LABEL_W, HEAD_H, LEGEND_H = 100, 24, 30
FRAME_MS, LAST_FRAME_MS = 400, 1500

font = ImageFont.load_default(size=14)
bold = ImageFont.load_default(size=16)


def movie_path(sample_id):
    return os.path.join(VIDEOS, f'movie_s{sample_id}r0.mp4')


def load_frames(sample_id):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(['ffmpeg', '-v', 'error', '-i', movie_path(sample_id), f'{d}/%04d.png'], check=True)
        return [Image.open(f).convert('RGB').crop(CROP).resize((TILE_W, TILE_H), Image.LANCZOS)
                for f in sorted(glob.glob(f'{d}/*.png'))]


def main():
    inactive_ids = [c[2] for c in COLUMNS if c[2] is not None]
    two_rows = any(os.path.exists(movie_path(s)) for s in inactive_ids)
    needed = [c[1] for c in COLUMNS] + ([s for s in inactive_ids if os.path.exists(movie_path(s))] if two_rows else [])
    missing = [s for s in [c[1] for c in COLUMNS] if not os.path.exists(movie_path(s))]
    if missing:
        raise FileNotFoundError(f'Missing nominal movies for samples {missing} in {VIDEOS}')
    if not two_rows:
        print('No inactive-rule movies found in videos/ -> building the nominal row only.')

    movies = {s: load_frames(s) for s in needed}
    n_frames = min(len(v) for v in movies.values())
    n_rows = 2 if two_rows else 1
    W = LABEL_W + TILE_W * len(COLUMNS)
    H = n_rows * (HEAD_H + TILE_H) + LEGEND_H

    out = []
    for k in range(n_frames):
        im = Image.new('RGB', (W, H), 'white')
        dr = ImageDraw.Draw(im)
        for j, (ic, nom, top, rule) in enumerate(COLUMNS):
            x = LABEL_W + j * TILE_W
            dr.text((x + TILE_W / 2, HEAD_H / 2), f'Tumor:CAF {ic}', fill='black', font=bold, anchor='mm')
            im.paste(movies[nom][k], (x, HEAD_H))
            if two_rows:
                y0 = HEAD_H + TILE_H
                if top in movies:
                    dr.text((x + TILE_W / 2, y0 + HEAD_H / 2), f'Inactive: {rule}', fill='black', font=font, anchor='mm')
                    im.paste(movies[top][k], (x, y0 + HEAD_H))
                else:
                    dr.text((x + TILE_W / 2, y0 + HEAD_H + TILE_H / 2), 'not available', fill='gray', font=font, anchor='mm')
        dr.text((LABEL_W / 2, HEAD_H + TILE_H / 2), 'Nominal\n(all rules)', fill='black', font=font, anchor='mm', align='center')
        if two_rows:
            dr.text((LABEL_W / 2, 2 * HEAD_H + 1.5 * TILE_H), 'Top-ranked\nrule\ninactive', fill='black', font=font, anchor='mm', align='center')

        x, y = LABEL_W + 20, H - LEGEND_H / 2
        for name, c in [('fibroblast', 'yellow'), ('epithelial tumor', 'green'), ('mesenchymal tumor', 'blue')]:
            dr.ellipse((x, y - 6, x + 12, y + 6), fill=c, outline='black')
            dr.text((x + 18, y), name, fill='black', font=font, anchor='lm')
            x += 180
        dr.text((x, y), 'background: ECM density (0-10)', fill='black', font=font, anchor='lm')
        out.append(im)

    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in out]
    pal[0].save(OUT, save_all=True, append_images=pal[1:], loop=0, optimize=True,
                duration=[FRAME_MS] * (len(pal) - 1) + [LAST_FRAME_MS])
    print(f'{OUT}: {len(pal)} frames, {os.path.getsize(OUT) // 1024} KB')


if __name__ == '__main__':
    main()
