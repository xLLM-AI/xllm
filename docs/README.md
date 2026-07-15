# xLLM Documentation

[English](./README.md) | [简体中文](./README.zh-CN.md)

This repository contains the Astro + Starlight documentation site for
[xLLM](https://github.com/xLLM-AI/xllm), an LLM inference framework for
high-performance serving on domestic AI accelerators.

The site is built with Starlight and `starlight-theme-rapide`. It includes a
custom header, bilingual navigation, and a page-level `Copy page` action for
copying documentation content as Markdown.

## Documentation Structure

The documentation is maintained in two parallel language trees:

- English: `src/content/docs/en`
- Simplified Chinese: `src/content/docs/zh`

The root path redirects to the English documentation:

- `/` redirects to `/en/`
- `/en/` serves the English documentation
- `/zh/` serves the Simplified Chinese documentation

When adding or moving pages, keep matching relative paths in both language
trees so Starlight can switch between languages for the same topic.

## Project Layout

```text
.
├── astro.config.mjs          # Starlight, locale, sidebar, and component config
├── package.json              # npm scripts and dependencies
├── src/
│   ├── assets/               # Site-level assets such as the logo
│   ├── components/           # Starlight component overrides
│   ├── content/
│   │   └── docs/
│   │       ├── en/           # English documentation
│   │       ├── zh/           # Simplified Chinese documentation
│   │       └── assets/       # Documentation images and diagrams
│   ├── pages/index.astro     # Redirect from / to /en/
│   └── styles/theme.css      # Project theme customizations
└── public/                   # Static public assets
```

## Local Development

Install dependencies:

```sh
npm install
```

Start the local development server:

```sh
npm run dev
```

Build the production site:

```sh
npm run build
```

Preview the production build:

```sh
npm run preview
```

## Editing Documentation

- Put user-facing content under `src/content/docs/en` and
  `src/content/docs/zh`.
- Keep English and Chinese files aligned by path when a page exists in both
  languages.
- Store shared documentation images in `src/content/docs/assets`.
- Update the `sidebar` section in `astro.config.mjs` when adding new sections
  that should appear in navigation.
- Run `npm run build` before submitting changes to catch broken routes,
  frontmatter errors, and Starlight content issues.

## Related Repository

- xLLM source code: <https://github.com/xLLM-AI/xllm>
