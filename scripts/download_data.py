"""Download the GSE63577 count matrix from GEO (no credentials needed)."""
import os
import urllib.request

BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63577/suppl/"
FILES = [
    "GSE63577_counts_rpkm_exvivo_jenage_data.xls.gz",
]

os.makedirs("data", exist_ok=True)
for f in FILES:
    dest = os.path.join("data", f)
    if os.path.exists(dest):
        print(f"exists: {dest}")
        continue
    url = BASE + f
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest}")

print("done. gunzip with: gunzip -kf data/*.gz")
