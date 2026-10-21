# GhostReel website

The production website is built from `site/content/`, `site/layouts/`, and
`site/assets/` using Zine 0.14.0. The homepage is `site/layouts/home.shtml`;
the guide is `site/content/guide.smd`.

From the repository root, start Zine's live development server:

```sh
zine
```

Open http://localhost:1990/. Zine watches the site files and reloads
the page after edits. To use another port, run `zine --port 1991`.

To build the static site locally, run:

```sh
zine release --force --output site-public
```

The output goes to `site-public/` (gitignored), not Zine's default `public/`:
`public/` is the desktop app's Vite static folder.

The site is served at https://ghostreel.foreachdeveloper.com/ from Cloudflare
Pages. Pushing to `master` doesn't deploy it: after publishing a release, the
Release workflow runs `.github/workflows/cloudflare.yml` on the release tag, so
the guide describes the app people can download. To publish a site-only fix
between releases, run it by hand:

```sh
gh workflow run cloudflare.yml --ref master
```

The download buttons read `site/assets/release.json`, a snapshot of the latest
GitHub release, which the Cloudflare workflow refreshes on every build.
