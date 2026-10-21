# GhostReel website

The production website is built from `site/content/`, `site/layouts/`, and
`site/assets/` using Zine 0.14.0. The homepage is `site/layouts/home.shtml`;
the guide is `site/content/guide.smd`.

From the repository root, start Zine's live development server:

```sh
zine
```

Open http://localhost:1990/GhostReel/. Zine watches the site files and reloads
the page after edits. To use another port, run `zine --port 1991`.

To build the static site locally, run:

```sh
zine release --force --output site-public
```

The output goes to `site-public/` (gitignored), not Zine's default `public/`:
`public/` is the desktop app's Vite static folder.

The download buttons read `site/assets/release.json`, a snapshot of the latest
GitHub release. The Pages and Cloudflare workflows refresh it on every build,
and the Release workflow rebuilds both sites after publishing a release.
`scripts/site-for-cloudflare.sh` rewrites the config for
https://ghostreel.foreachdeveloper.com/ in CI; don't commit its output.
