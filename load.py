import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_curves(csv_dir, min_pts=20, max_pts=250):
    """
    Same preprocessing as in DBSCAN.ipynb.
    """
    curves_data = []
    files = os.listdir(csv_dir)
    for fname in tqdm(files, desc="Processing curves"):
        oid = fname.replace(".csv", "")
        df = pd.read_csv(os.path.join(csv_dir, fname))
        g = df[df['fid'] == 1]
        r = df[df['fid'] == 2].sort_values('mjd')

        if len(r) < min_pts or len(g) < min_pts or len(r) > max_pts:
            continue

        color_offset = g['magpsf'].median() - r['magpsf'].median()
        mag = r['magpsf'].values + color_offset
        t = r['mjd'].values

        if np.any(np.isnan(mag)) or np.any(np.isnan(t)):
            continue

        # Normalize
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)

        curves_data.append((oid, t_norm, mag_norm, t, mag, df['ra'][0], df['dec'][0]))

    return curves_data
