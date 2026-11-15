/**
 * A clip's volume in the live viewer. A media element's own `volume` can only make a clip softer;
 * a speaker far from the mic needs louder, so the element's sound is routed through Web Audio
 * (a gain node) the first time a clip asks for a change. Elements nobody turns up or down are
 * never touched. The element must load with `crossOrigin = "anonymous"` (set before its src) —
 * the desktop media server allows it — or the browser feeds the graph silence.
 */
let ctx: AudioContext | null = null;
const gains = new WeakMap<HTMLMediaElement, GainNode>();

/** Play `el` at `db` (0 = as recorded); `muted` silences it whatever the gain. */
export function setLiveGain(el: HTMLMediaElement | null, db: number, muted: boolean) {
  if (!el) return;
  let g = gains.get(el);
  if (!g) {
    if (!db) return;
    try {
      ctx ??= new AudioContext();
      const src = ctx.createMediaElementSource(el);
      g = ctx.createGain();
      src.connect(g).connect(ctx.destination);
      gains.set(el, g);
    } catch {
      return; // no Web Audio here: the clip plays as recorded
    }
  }
  if (ctx?.state === "suspended") void ctx.resume();
  g.gain.value = muted ? 0 : Math.pow(10, db / 20);
}
