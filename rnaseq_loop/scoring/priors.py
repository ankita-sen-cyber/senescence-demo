"""External prior sources for target scoring.

- Open Targets Platform (GraphQL): target-disease association scores.
- DepMap (bi-annual CRISPR essentiality): CSV downloads.

Both are cached to disk under `data/priors/`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger(__name__)

OT_GQL = "https://api.platform.opentargets.org/api/v4/graphql"


def _ot_graphql(query: str, variables: dict) -> dict:
    r = requests.post(OT_GQL, json={"query": query, "variables": variables}, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    return payload["data"]


def _resolve_ot_disease_id(disease_efo: str) -> str:
    """Return a live Open Targets disease id, searching if the given EFO is stale."""
    probe = """
    query DiseaseExists($efoId: String!) {
      disease(efoId: $efoId) { id name }
    }
    """
    data = _ot_graphql(probe, {"efoId": disease_efo})
    if data.get("disease"):
        return data["disease"]["id"]

    search = """
    query DiseaseSearch($q: String!) {
      search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 5}) {
        hits { id name entity }
      }
    }
    """
    hits = _ot_graphql(search, {"q": disease_efo}).get("search", {}).get("hits") or []
    diseases = [h for h in hits if h.get("entity") == "disease" and h.get("id")]
    if not diseases:
        raise ValueError(
            f"No disease found for EFO id {disease_efo}. "
            "Open Targets IDs change; try searching https://platform.opentargets.org/"
        )
    chosen = diseases[0]["id"]
    log.warning(
        f"Open Targets has no disease {disease_efo}; using search hit "
        f"{chosen} ({diseases[0].get('name')})"
    )
    return chosen


def opentargets_association(
    disease_efo: str,
    top_n: int = 500,
    cache_dir: str | Path = "data/priors/opentargets",
) -> pd.DataFrame:
    """
    Fetch target-disease association scores for a disease EFO id
    (e.g. 'EFO_0022597' for aging, 'MONDO_0007254' for breast cancer).
    """
    cache_dir = ensure_dir(cache_dir)
    cache_file = cache_dir / f"{disease_efo}_top{top_n}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    disease_efo = _resolve_ot_disease_id(disease_efo)
    cache_file = cache_dir / f"{disease_efo}_top{top_n}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    query = """
    query DiseaseAssociations($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: 0, size: $size}) {
          count
          rows {
            score
            target { id approvedSymbol biotype }
            datatypeScores { id score }
          }
        }
      }
    }
    """
    data = _ot_graphql(query, {"efoId": disease_efo, "size": top_n})["disease"]
    if data is None:
        raise ValueError(f"No disease found for EFO id {disease_efo}")

    rows = []
    for row in data["associatedTargets"]["rows"]:
        record = {
            "ensembl_id": row["target"]["id"],
            "symbol": row["target"]["approvedSymbol"],
            "biotype": row["target"]["biotype"],
            "ot_overall_score": row["score"],
        }
        for dt in row["datatypeScores"]:
            record[f"ot_{dt['id']}"] = dt["score"]
        rows.append(record)

    df = pd.DataFrame(rows)
    df.to_parquet(cache_file)
    log.info(f"OT: {len(df)} targets for {data['name']} ({disease_efo}) cached at {cache_file}")
    return df


def load_depmap_essentiality(
    csv_path: str | Path,
    subset_cell_lines: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load DepMap gene effect (Chronos or CERES) scores.

    Expects the standard DepMap release CSV with cell lines as rows and
    genes as `SYMBOL (ENSG...)` columns. Download from https://depmap.org/portal/data_page/.

    Returns a DataFrame indexed by gene symbol with per-cell-line effect scores.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # Normalize column names: "TP53 (ENSG00000141510)" → "TP53"
    df.columns = [c.split(" ")[0] for c in df.columns]
    if subset_cell_lines is not None:
        df = df.loc[df.index.intersection(subset_cell_lines)]
    # Transpose so rows are genes.
    gene_df = df.T
    log.info(f"DepMap: {gene_df.shape[0]} genes × {gene_df.shape[1]} cell lines")
    return gene_df


def depmap_essentiality_summary(
    gene_df: pd.DataFrame,
    threshold: float = -0.5,
) -> pd.DataFrame:
    """
    Per-gene summary: mean effect, fraction of cell lines strongly dependent
    (effect ≤ threshold, following DepMap convention).
    """
    out = pd.DataFrame({
        "depmap_mean_effect": gene_df.mean(axis=1),
        "depmap_frac_essential": (gene_df <= threshold).mean(axis=1),
        "depmap_n_lines": gene_df.notna().sum(axis=1),
    })
    return out
