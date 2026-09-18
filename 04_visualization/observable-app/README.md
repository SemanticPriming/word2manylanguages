# Word2ManyLanguages Explorer (Observable Framework)

Static, client-side port of the original Shiny app (`../app.R`) built with
[Observable Framework](https://observablehq.com/framework/) so it can be
hosted for free on GitHub Pages instead of requiring a Shiny server.

## What changed vs. the Shiny app

- Data cleaning/recoding (language names, algo names, R-squared clamping)
  now happens once at build time in [`docs/data/combined.json.js`](docs/data/combined.json.js),
  a Framework data loader that reads the three CSVs from `src-data/` and
  emits a single combined `data/combined.json`.
- All filtering, the best-models table, CSV downloads, and the heatmap
  (Observable Plot) run entirely in the browser — no server/backend needed.
- The Window/Dimension range sliders became multi-select checkboxes (there's
  no built-in dual-handle range input in Observable Inputs); default is all
  values selected, which is equivalent to the original full-range default.
- The Best Models table gains a **Citation** column, linking each row's
  norm dataset back to its source paper (see below). This wasn't in the
  Shiny app.

## Citations

Each row of the Replication/Extension evals is tied to a `dataset` file
(e.g. `Khwaileh2018.csv`, `de-grandy-2020.tsv`). The data loader matches
that filename against the [SemanticPrimeR](https://github.com/SemanticPriming/semanticprimeR)
package's model-card YAMLs — copied into `src-data/citations/` — to attach
an "Author et al., Year" label and a `https://doi.org/...` link. Count evals
(corpus frequency counts, not literature norms) instead show the corpus
name (Subtitles/Wikipedia) with no link.

Matching is filename-based (`Khwaileh2018` / normalized `de-grandy-2020` →
`Grandy2020`) against whatever model cards existed in the local copy of
semanticprimeR at the time `src-data/citations/` was populated. Coverage:
~98% of Extension datasets, ~54% of Replication datasets (the rest fall
back to showing the bare filename with no link — usually because the
package's card uses a different year/suffix than this project's filename,
e.g. `Pexman2017.yaml` vs. `en-pexman-2019.tsv`). To improve coverage,
refresh `src-data/citations/` from a newer semanticprimeR checkout's
`inst/extdata/model_cards/`, or add/rename YAMLs by hand to match the
`dataset` filenames used here.

## Develop

```sh
npm install
npm run dev
```

Then open the printed local URL. Edits to `docs/index.md` hot-reload.

## Build

```sh
npm run build
```

`output` in [`observablehq.config.js`](observablehq.config.js) is set to
`../../docs`, so this writes the static site straight into the **repo-root**
`docs/` folder (`words2many_backup/docs/`) — a different folder than this
app's own `docs/` source directory (`observable-app/docs/`, holding
`index.md` and the data loader).

## Deploy to GitHub Pages

No GitHub Actions workflow needed — classic branch-based Pages serves the
committed `docs/` folder directly:

1. Run `npm run build` from this folder whenever you change `docs/index.md`,
   `docs/data/combined.json.js`, or the files in `src-data/`.
2. `git add` and commit the regenerated repo-root `docs/` folder along with
   your source changes, and push to `master`.
3. In the GitHub repo, go to **Settings → Pages**, set **Source** to
   **Deploy from a branch**, and pick branch `master`, folder `/docs`.
4. The site publishes at `https://semanticpriming.github.io/word2manylanguages/`.

The repo-root `docs/` folder is a build artifact (regenerate, don't hand-edit
it) but it must stay committed to git for Pages to serve it — it is not
gitignored.

If you fork or rename the repo, update `base` in
[`observablehq.config.js`](observablehq.config.js) to match the new path
(or set it to `"/"` if deploying to a custom domain or a `<user>.github.io`
root repo).

## Updating the data

Replace the CSVs in `src-data/` (same filenames/columns as the Shiny app
expects) and rerun `npm run build` — the loader recombines and recleans
everything automatically.
