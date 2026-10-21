// The mobile menu, and the latest version badge from the cached release.json
// (refreshed by CI on every build), with GitHub's API as a fallback.
(function () {
  var toggle = document.querySelector(".menu-toggle");
  var nav = document.querySelector(".topnav");
  if (toggle && nav) {
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  var badge = document.querySelector("[data-latest-version]");
  if (!badge) return;
  function show(release) {
    var version = release && release.tag_name;
    if (!version) throw new Error("no tag");
    badge.textContent = version;
    badge.hidden = false;
  }
  function json(r) { if (!r.ok) throw new Error(r.status); return r.json(); }
  fetch(badge.getAttribute("data-release-json"), { cache: "no-cache" })
    .then(json)
    .then(show)
    .catch(function () {
      return fetch("https://api.github.com/repos/highercomve/GhostReel/releases/latest", { headers: { Accept: "application/vnd.github+json" } })
        .then(json)
        .then(show);
    })
    .catch(function () { badge.hidden = true; });
})();
