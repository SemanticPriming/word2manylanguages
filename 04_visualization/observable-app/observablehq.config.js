export default {
  title: "Word2ManyLanguages Explorer",
  pages: [],
  root: "docs",
  // Build straight into the repo-root /docs folder so GitHub Pages can serve
  // it via Settings → Pages → Deploy from a branch → /docs, with no Actions
  // workflow needed. This is a *different* docs/ than `root` above (that one
  // is this app's own source folder, two levels down).
  output: "../../docs",
  // GitHub Pages serves the site from a subpath (the repo name), so set base
  // to "/<repo-name>/" when deploying, or leave "/" if using a custom domain
  // or a user/organization root page (username.github.io).
  base: "/word2manylanguages/",
  toc: false,
  pager: false,
  footer: "Word2ManyLanguages Explorer — built with Observable Framework."
};
