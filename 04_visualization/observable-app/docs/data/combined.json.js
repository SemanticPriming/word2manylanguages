// Data loader: combines the three evaluation CSVs, cleans and recodes them
// the same way 04_visualization/app.R does, joins each row against the
// SemanticPrimeR model-card YAMLs for a citation + DOI link, and emits one
// JSON array the client-side app loads via FileAttachment("data/combined.json").
import {csvParse} from "d3-dsv";
import {readFile, readdir} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import path from "node:path";
import yaml from "js-yaml";

const here = path.dirname(fileURLToPath(import.meta.url));
const srcDataDir = path.join(here, "..", "..", "src-data");
const citationsDir = path.join(srcDataDir, "citations");

const LANGUAGE_NAMES = {
  af: "Afrikaans", ar: "Arabic", bg: "Bulgarian", bn: "Bengali",
  br: "Breton", bs: "Bosnian", ca: "Catalan", cs: "Czech",
  da: "Danish", de: "German", el: "Greek", en: "English",
  eo: "Esperanto", es: "Spanish", et: "Estonian", eu: "Basque",
  fa: "Persian", fi: "Finnish", fr: "French", gl: "Galician",
  he: "Hebrew", hi: "Hindi", hr: "Croatian", hu: "Hungarian",
  hy: "Armenian", id: "Indonesian", is: "Icelandic", it: "Italian",
  ja: "Japanese", ka: "Georgian", kk: "Kazakh", ko: "Korean",
  lt: "Lithuanian", lv: "Latvian", mk: "Macedonian", ml: "Malayalam",
  ms: "Malay", nl: "Dutch", no: "Norwegian", pl: "Polish",
  pt: "Portuguese", ro: "Romanian", ru: "Russian", si: "Sinhala",
  sk: "Slovak", sl: "Slovenian", sq: "Albanian", sr: "Serbian",
  sv: "Swedish", ta: "Tamil", te: "Telugu", tl: "Tagalog",
  tr: "Turkish", uk: "Ukrainian", ur: "Urdu", vi: "Vietnamese",
  zh: "Chinese"
};

const ALGO_NAMES = {
  cbow: "Continuous Bag of Words",
  sg: "Skip-gram"
};

function clamp01(x) {
  if (x === null || x === undefined || x === "" || Number.isNaN(+x)) return null;
  return Math.min(Math.max(+x, 0), 1);
}

function toNumber(x) {
  if (x === null || x === undefined || x === "") return null;
  const n = +x;
  return Number.isNaN(n) ? null : n;
}

// ---- Citations: load every SemanticPrimeR model-card YAML, keyed by its
// bibtex id (the filename without extension, e.g. "Khwaileh2018"). --------
// Loosely normalized (lowercase, alphanumeric-only) so e.g. "Stadthagen-
// Gonzalez2017" and a derived candidate "stadthagengonzalez2017" still match.
const loosen = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");

async function loadCitations() {
  const files = await readdir(citationsDir);
  const byKey = new Map();
  for (const file of files) {
    if (!file.endsWith(".yaml")) continue;
    const key = file.slice(0, -5);
    const doc = yaml.load(await readFile(path.join(citationsDir, file), "utf8"));
    const c = doc?.citation;
    if (!c?.author || !c?.year) continue;
    const firstAuthorSurname = c.author.split(",")[0].trim().split(" ").pop();
    const label = c.author.includes(",")
      ? `${firstAuthorSurname} et al., ${c.year}`
      : `${firstAuthorSurname}, ${c.year}`;
    const entry = {
      citation: label,
      citation_title: c.title ?? null,
      citation_doi: c.doi ? `https://doi.org/${c.doi}` : null
    };
    byKey.set(key, entry);
    byKey.set(loosen(key), entry);
  }
  return byKey;
}

// eval_inputs/*_evals dataset filenames come in two shapes:
//   "Khwaileh2018.csv"        (extension evals: bibtex key + extension)
//   "de-grandy-2020.tsv"      (replication evals: lang-author-year, dashed)
// Both normalize to a bibtex key like "Khwaileh2018" / "Grandy2020".
function datasetToBibtexKey(dataset) {
  if (!dataset) return null;
  const base = dataset.replace(/\.(csv|tsv)$/i, "");
  const m = base.match(/^[a-z]{2}-(.+)-(\d{4}[a-z]?)$/i);
  if (m) {
    const author = m[1].replace(/-/g, "");
    return author.charAt(0).toUpperCase() + author.slice(1) + m[2];
  }
  return base;
}

function citationFor(dataset, dataSource, citations) {
  if (dataSource) {
    // Count evals are keyed off corpus files (subtitles/Wikipedia dumps),
    // not literature norms, so there's no DOI to link — just label the corpus.
    return {citation: dataSource, citation_title: null, citation_doi: null};
  }
  const key = datasetToBibtexKey(dataset);
  return (
    citations.get(key) ??
    citations.get(loosen(key ?? "")) ?? {
      citation: dataset ?? null,
      citation_title: null,
      citation_doi: null
    }
  );
}

async function loadCsv(filename, source, citations) {
  const text = await readFile(path.join(srcDataDir, filename), "utf8");
  const rows = csvParse(text);
  return rows.map((d) => ({
    var: d.var,
    adjusted_r: clamp01(d.adjusted_r),
    adjusted_r_squared: clamp01(d.adjusted_r_squared),
    r_squared: clamp01(d.r_squared),
    r: clamp01(d.r),
    dataset: d.dataset ?? null,
    language: LANGUAGE_NAMES[d.language] ?? d.language,
    dim: toNumber(d.dim),
    window: toNumber(d.window),
    algo: ALGO_NAMES[d.algo] ?? d.algo,
    source,
    ...citationFor(d.dataset, d.data_source, citations)
  }));
}

const citations = await loadCitations();

const [rep, ext, count] = await Promise.all([
  loadCsv("rep_evals_formatted_new.csv", "Replication", citations),
  loadCsv("extension_evals_formatted_new.csv", "Extension", citations),
  loadCsv("count_evals_formatted_new.csv", "Count", citations)
]);

process.stdout.write(JSON.stringify([...rep, ...ext, ...count]));
