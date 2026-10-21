// Resolve the latest release assets into direct file links.
// The Pages build writes the latest release to release.json, so visitors don't
// hit GitHub's API rate limit; the live API is only a fallback.
(() => {
  const endpoint = 'https://api.github.com/repos/highercomve/GhostReel/releases/latest';
  const cached = document.querySelector('[data-release-json]')?.dataset.releaseJson;
  const version = document.querySelector('[data-release-version]');
  const primary = document.querySelector('[data-primary-download]');
  const cards = [...document.querySelectorAll('[data-download-os]')];
  const os = /Win/i.test(navigator.userAgent) ? 'windows' : /Mac/i.test(navigator.userAgent) ? 'macos' : 'linux';

  // What an asset is, from the file names release.yml publishes:
  // GhostReel_<v>_amd64.AppImage, GhostReel_<v>_x64-setup.exe, GhostReel_<v>_aarch64.dmg / _x64.dmg.
  // Updater payloads (.sig, .app.tar.gz, latest.json) are not offered.
  function describe(name) {
    if (/\.AppImage$/i.test(name)) return { os:'linux', label:'AppImage · x86-64', specs:[['Package','AppImage'],['GPU','NVIDIA CUDA']], priority:0 };
    if (/-setup\.exe$/i.test(name)) return { os:'windows', label:'Windows installer (.exe)', specs:[['Package','Installer (.exe)'],['GPU','NVIDIA CUDA']], priority:0 };
    if (/aarch64\.dmg$/i.test(name)) return { os:'macos', arch:'arm', label:'macOS disk image · Apple Silicon', priority:0 };
    if (/x64\.dmg$/i.test(name)) return { os:'macos', arch:'intel', label:'macOS disk image · Intel', priority:1 };
    return null;
  }
  function size(bytes) {
    return bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + ' GB' : Math.max(1, Math.round(bytes / 1e6)) + ' MB';
  }
  const ARCHES = [['arm','Apple Silicon'],['intel','Intel']];
  function downloadLink(asset) {
    const link = document.createElement('a');
    const label = document.createElement('span');
    const meta = document.createElement('small');
    link.append(label, meta);
    link.set = next => {
      link.href = next.url;
      link.setAttribute('aria-label', 'Download ' + next.label + ', ' + size(next.bytes));
      label.textContent = '↓  Download';
      meta.textContent = size(next.bytes);
    };
    link.set(asset);
    return link;
  }
  function choice(title, name, options) {
    const set = document.createElement('fieldset');
    set.className = 'asset-choice';
    const legend = document.createElement('legend');
    legend.textContent = title;
    const row = document.createElement('div');
    row.className = 'asset-options';
    options.forEach(([value, text]) => {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = name;
      input.value = value;
      const span = document.createElement('span');
      span.textContent = text;
      label.append(input, span);
      row.append(label);
    });
    set.append(legend, row);
    return set;
  }
  function fixed(list, title, value) {
    const row = document.createElement('div');
    row.className = 'asset-choice';
    const legend = document.createElement('p');
    legend.className = 'asset-legend';
    legend.textContent = title;
    const box = document.createElement('div');
    box.className = 'asset-fixed';
    box.textContent = value;
    row.append(legend, box);
    list.append(row);
  }
  function macPicker(list, items) {
    const arches = choice('Mac', 'mac-arch', ARCHES.filter(([a]) => items.some(i => i.arch === a)));
    const link = downloadLink(items[0]);
    list.append(arches, link);
    const first = list.querySelector(`input[name="mac-arch"][value="${items[0].arch}"]`);
    if (first) first.checked = true;
    list.addEventListener('change', () => {
      const arch = list.querySelector('input[name="mac-arch"]:checked').value;
      link.set(items.find(i => i.arch === arch) || items[0]);
    });
  }
  function retry(message) {
    version.textContent = message;
    cards.forEach(card => {
      const list = card.querySelector('[data-assets]');
      list.replaceChildren();
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'asset-retry';
      button.textContent = 'Retry downloads';
      button.addEventListener('click', load);
      list.append(button);
    });
  }
  async function load() {
    version.textContent = 'Finding the latest version…';
    cards.forEach(card => { card.querySelector('[data-assets]').textContent = 'Loading downloads…'; });
    try {
      let release = null;
      if (cached) {
        try {
          const response = await fetch(cached, { cache:'no-cache' });
          if (response.ok) release = await response.json();
        } catch(error) { console.warn('Could not read the cached release:', error); }
      }
      if (!release?.assets?.length) {
        const response = await fetch(endpoint, { headers:{ Accept:'application/vnd.github+json' } });
        if (!response.ok) throw new Error('Release API ' + response.status);
        release = await response.json();
      }
      const assets = { linux:[], windows:[], macos:[] };
      (release.assets || []).forEach(asset => {
        const type = describe(asset.name);
        if (type && /^https:\/\/github\.com\/highercomve\/GhostReel\/releases\/download\//i.test(asset.browser_download_url)) {
          assets[type.os].push({ ...type, url:asset.browser_download_url, bytes:asset.size });
        }
      });
      Object.values(assets).forEach(items => items.sort((a,b) => a.priority - b.priority));
      if (!Object.values(assets).some(items => items.length)) throw new Error('No installers found');
      version.textContent = release.tag_name ? 'Latest release · ' + release.tag_name : 'Latest release';
      cards.forEach(card => {
        const target = card.dataset.downloadOs;
        const list = card.querySelector('[data-assets]');
        list.replaceChildren();
        if (!assets[target].length) {
          const empty = document.createElement('span');
          empty.className = 'asset-loading';
          empty.textContent = 'No download for this platform in the latest release.';
          list.append(empty);
          return;
        }
        if (target === 'macos' && assets.macos.length > 1) return macPicker(list, assets.macos);
        const asset = assets[target][0];
        (asset.specs || [['Package', asset.label]]).forEach(([title, value]) => fixed(list, title, value));
        list.append(downloadLink(asset));
      });
      const preferred = assets[os][0];
      if (preferred && primary) {
        // A Mac's chip can't be read reliably from the browser: on a Mac with two
        // builds, the button goes to the card where the visitor picks one.
        const pick = os === 'macos' && assets.macos.length > 1;
        primary.href = pick ? '#download' : preferred.url;
        primary.firstChild.textContent = 'Download for ' + ({linux:'Linux',windows:'Windows',macos:'macOS'}[os]) + ' ';
        primary.title = pick ? 'Apple Silicon or Intel' : preferred.label + ' · ' + size(preferred.bytes);
      }
    } catch(error) {
      console.warn('Could not load GhostReel release assets:', error);
      if (primary) primary.href = '#download';
      retry('Downloads are temporarily unavailable');
    }
  }
  load();
})();
