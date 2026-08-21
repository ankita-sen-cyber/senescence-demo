"""Download and decompress the GSE63577 count matrix from GEO (no credentials needed).

Run from the repo root:  python scripts/download_data.py
This downloads the .gz file and decompresses it automatically — no manual gunzip step.
"""
import os
import gzip
import shutil
import urllib.request

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63577/suppl/"
FILE = "GSE63577_counts_rpkm_exvivo_jenage_data.xls.gz"

os.makedirs("data", exist_ok=True)
gz_path = os.path.join("data", FILE)
xls_path = gz_path[:-3]  # strip the .gz suffix

if os.path.exists(xls_path):
    print(f"already present: {xls_path}")
else:
    if not os.path.exists(gz_path):
        print(f"downloading {BASE + FILE}")
        urllib.request.urlretrieve(BASE + FILE, gz_path)
    print(f"decompressing {gz_path}")
    with gzip.open(gz_path, "rb") as f_in, open(xls_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"-> {xls_path}")

print("done.")
