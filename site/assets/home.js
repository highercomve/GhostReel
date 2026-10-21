// The hero walkthrough: four illustrative steps with fixed sample footage, not a real index.
(() => {
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const demo = document.querySelector('.demo');
  if (!demo) return;
  const stepButtons = [...document.querySelectorAll('.step-button')];
  const playButton = document.querySelector('.demo-play');
  const motionButton = document.querySelector('.motion-toggle');
  const panels = [...demo.querySelectorAll('[data-panel]')];
  const status = demo.querySelector('[data-local-status]');
  const messages = [
    'Add a folder: GhostReel finds every video in it.',
    'Speech, keyframes and vectors are processed on your GPU.',
    'Search by what was said, shown or written on screen.',
    'Draft the cut, preview it, and export. Your footage stayed here.'
  ];
  const durations = [2600, 3200, 3400, 4200];
  let step = 0;
  let playing = !reducedMotion.matches;
  let motionPaused = reducedMotion.matches;
  let timer;

  function syncMotion() {
    document.body.classList.toggle('motion-paused', motionPaused);
    motionButton.textContent = motionPaused ? '▷ Play motion' : 'Ⅱ Pause motion';
    motionButton.setAttribute('aria-pressed', String(motionPaused));
  }
  function schedule() {
    clearTimeout(timer);
    if (playing && !motionPaused && !document.hidden) timer = setTimeout(() => setStep((step + 1) % 4), durations[step]);
  }
  function syncPlay() {
    const active = playing && !motionPaused;
    playButton.textContent = active ? 'Ⅱ' : '▷';
    playButton.setAttribute('aria-label', active ? 'Pause walkthrough' : 'Play walkthrough');
  }
  function setStep(next) {
    step = next;
    demo.dataset.step = String(step);
    if (status) status.textContent = messages[step];
    // Hidden panels stay out of the accessibility tree.
    panels.forEach(panel => panel.setAttribute('aria-hidden', String(panel.dataset.panel !== String(step))));
    stepButtons.forEach((button, index) => {
      if (index === step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    syncPlay();
    schedule();
  }
  stepButtons.forEach((button, index) => button.addEventListener('click', () => {
    playing = false;
    setStep(index);
  }));
  playButton.addEventListener('click', () => {
    if (motionPaused) { motionPaused = false; playing = true; syncMotion(); }
    else playing = !playing;
    syncPlay();
    schedule();
  });
  motionButton.addEventListener('click', () => {
    motionPaused = !motionPaused;
    syncMotion();
    syncPlay();
    schedule();
  });
  reducedMotion.addEventListener('change', event => {
    motionPaused = event.matches;
    syncMotion();
    syncPlay();
    schedule();
  });
  document.addEventListener('visibilitychange', schedule);
  syncMotion();
  setStep(reducedMotion.matches ? 2 : 0);
})();
