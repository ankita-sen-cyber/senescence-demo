# Results viewer

A static, dependency-free frontend for browsing the pipeline outputs in `results/` —
built for biological review of the senescence target-discovery demo.

## What it shows

- Overview KPIs (known-marker recovery, ranked targets, significant genes, held-out accuracy)
- Failure analysis: cell-line vs. condition F-statistics (why iteration 1 exists)
- Interactive volcano plot with an iteration 0 / iteration 1 toggle and labeled literature markers
- Ranked intervention targets, filterable by inhibit / activate direction
- Pathway enrichment (Hallmark GSEA) and transcription-factor regulon activity
- Per-marker validation grid comparing both iterations
- Leave-one-cell-line-out generalization

## Run locally

No build step. Serve the folder with any static server (opening `index.html`
directly also works in most browsers, since all data is inlined in `data.js`):

```bash
cd viewer
python3 -m http.server 8080
# open http://localhost:8080
```

## Regenerating the data

`data.js` is a snapshot generated from the CSV/JSON files in `../results/`
(volcano points are downsampled to ~6,000 per iteration for rendering speed;
all significant genes are kept, non-significant ones are subsampled).
If you re-run the pipeline, regenerate `data.js` from the new `results/` files
before reviewing.

Charts are rendered with Chart.js (loaded from CDN); everything else is plain
HTML/CSS/JS.
