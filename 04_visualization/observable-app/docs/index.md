---
title: Word2ManyLanguages Explorer
---

# Word2ManyLanguages Explorer

Interactive exploration of model performance across languages, corpora, and hyperparameter configurations. Ported from the original [Shiny app](https://shiny.rstudio.com/) to [Observable Framework](https://observablehq.com/framework/).

```js
const raw = FileAttachment("data/combined.json").json();
```

```js
const sources = [...new Set(raw.map((d) => d.source))].sort();
```

<div class="tip">
Use the filters below to narrow the dataset before viewing the table or heatmap. Results update automatically.
</div>

## Filters

```js
const sourceInput = view(
  Inputs.select(["All", ...sources], {label: "Source", value: "All"})
);
```

```js
const bySource = sourceInput === "All" ? raw : raw.filter((d) => d.source === sourceInput);
```

```js
const languages = [...new Set(bySource.map((d) => d.language))].sort();
```

```js
const languageInput = view(
  Inputs.select(["All", ...languages], {label: "Language", value: "All"})
);
```

```js
const byLanguage = languageInput === "All" ? bySource : bySource.filter((d) => d.language === languageInput);
```

```js
const vars = [...new Set(byLanguage.map((d) => d.var))].sort();
```

```js
const varInput = view(
  Inputs.select(["All", ...vars], {label: "Variable", value: "All"})
);
```

```js
const byVar = varInput === "All" ? byLanguage : byLanguage.filter((d) => d.var === varInput);
```

```js
const algoInput = view(
  Inputs.select(["All", "Continuous Bag of Words", "Skip-gram"], {label: "Algorithm", value: "All"})
);
```

```js
const byAlgo = algoInput === "All" ? byVar : byVar.filter((d) => d.algo === algoInput);
```

```js
const windowChoices = [...new Set(byAlgo.map((d) => d.window).filter((d) => d != null))].sort((a, b) => a - b);
```

```js
const windowInput = view(
  Inputs.checkbox(windowChoices, {label: "Window Size", value: windowChoices})
);
```

```js
const dimChoices = [...new Set(byAlgo.map((d) => d.dim).filter((d) => d != null))].sort((a, b) => a - b);
```

```js
const dimInput = view(
  Inputs.checkbox(dimChoices, {label: "Dimensions", value: dimChoices})
);
```

```js
const rankingMetric = view(
  Inputs.radio(
    new Map([
      ["Adjusted R", "adjusted_r"],
      ["Adjusted R Squared", "adjusted_r_squared"],
      ["R Squared", "r_squared"],
      ["R", "r"]
    ]),
    {label: "Ranking Metric", value: "adjusted_r"}
  )
);
```

```js
const windowSet = new Set(windowInput);
const dimSet = new Set(dimInput);
const filtered = byAlgo.filter((d) => windowSet.has(d.window) && dimSet.has(d.dim));
```

## Best Models Table

```js
const bestModels = filtered
  .map((d) => ({
    var: d.var,
    language: d.language,
    algo: d.algo,
    source: d.source,
    window: d.window,
    dim: d.dim,
    value: d[rankingMetric] == null ? null : Math.round(d[rankingMetric] * 1000) / 1000,
    citation: d.citation ?? null,
    citation_doi: d.citation_doi ?? null,
    citation_title: d.citation_title ?? null
  }))
  .sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
```

```js
// The table needs the citation rendered as a clickable link; the CSV
// download needs plain text, so keep both views over the same rows.
const bestModelsDisplay = bestModels.map((d) => ({
  ...d,
  citation:
    d.citation == null
      ? null
      : d.citation_doi
      ? htl.html`<a href=${d.citation_doi} target="_blank" rel="noopener noreferrer" title=${d.citation_title ?? ""}>${d.citation}</a>`
      : d.citation
}));
```

```js
bestModelsDisplay.length > 0
  ? Inputs.table(bestModelsDisplay, {
      columns: ["var", "language", "algo", "source", "window", "dim", "value", "citation"],
      header: {
        var: "Variable",
        language: "Language",
        algo: "Algorithm",
        source: "Source",
        window: "Window",
        dim: "Dim",
        value: "Value",
        citation: "Citation"
      },
      format: {citation: (c) => c ?? ""},
      rows: 12
    })
  : htl.html`<p><em>No rows match the current filters. Try widening the selection or choosing "All".</em></p>`
```

```js
function downloadCsv(rows, filename) {
  const header = Object.keys(rows[0] ?? {});
  const csv = [
    header.join(","),
    ...rows.map((row) => header.map((key) => JSON.stringify(row[key] ?? "")).join(","))
  ].join("\n");
  const blob = new Blob([csv], {type: "text/csv"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

```js
Inputs.button("Download Best Models CSV", {
  reduce: () => downloadCsv(bestModels, "best_models.csv")
})
```

## Heatmap

```js
const heatmapSummary = (() => {
  const groups = d3.rollup(
    filtered,
    (v) => d3.mean(v, (d) => d[rankingMetric]),
    (d) => d.algo,
    (d) => d.window,
    (d) => d.dim
  );
  const windowLevels = windowChoices;
  const dimLevels = dimChoices;
  const algoLevels = [...new Set(filtered.map((d) => d.algo))];
  const out = [];
  for (const algo of algoLevels) {
    for (const window of windowLevels) {
      for (const dim of dimLevels) {
        const avg = groups.get(algo)?.get(window)?.get(dim);
        out.push({
          algo_panel: algo === "Continuous Bag of Words" ? "CBOW" : algo === "Skip-gram" ? "SKIP-GRAM" : algo,
          window,
          dim,
          avg_metric: avg == null || Number.isNaN(avg) ? null : Math.round(avg * 1000) / 1000
        });
      }
    }
  }
  return out;
})();
```

```js
heatmapSummary.length > 0
  ? Plot.plot({
      title: `Average ${rankingMetric} by Window × Dimension`,
      subtitle: "Red outline indicates original model dimensions and windows (window = 5, dim = 300)",
      marginLeft: 60,
      facet: {data: heatmapSummary, x: "algo_panel"},
      x: {axis: null},
      y: {label: "Window", domain: windowChoices, tickFormat: (d) => d},
      color: {type: "sequential", scheme: "viridis", label: "Avg (rounded)", legend: true},
      fx: {label: null},
      marks: [
        Plot.cell(heatmapSummary, {
          x: "dim",
          y: "window",
          fill: "avg_metric",
          fx: "algo_panel",
          stroke: "#ccc"
        }),
        Plot.text(heatmapSummary, {
          x: "dim",
          y: "window",
          fx: "algo_panel",
          text: (d) => (d.avg_metric == null ? "" : d.avg_metric.toFixed(3)),
          fill: "black",
          fontWeight: "bold"
        }),
        Plot.frame(
          heatmapSummary.filter((d) => d.window === 5 && d.dim === 300),
          {x: "dim", y: "window", fx: "algo_panel", stroke: "red", strokeWidth: 2.5}
        )
      ],
      height: 420,
      width: 900
    })
  : htl.html`<p><em>No rows match the current filters.</em></p>`
```

```js
Inputs.button("Download Heatmap Data CSV", {
  reduce: () => downloadCsv(heatmapSummary, "heatmap_data.csv")
})
```

## Citations

References used in this project will appear here.

<style>
.tip {
  background: color-mix(in srgb, var(--theme-foreground-focus) 12%, transparent);
  border-left: 4px solid var(--theme-foreground-focus);
  padding: 0.5rem 1rem;
  border-radius: 4px;
  margin: 1rem 0;
}
</style>
