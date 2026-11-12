/**
 * Something went into or came out of the trash in this window (a delete, a restore from a run's
 * link or from Recently deleted). Lists of cuts and clip sets, the trash itself and the run log
 * reload on it.
 */
const EVENT = "ghostreel:trash-changed";

export function trashChanged(): void {
  window.dispatchEvent(new Event(EVENT));
}

/** Subscribe; returns the unsubscribe. */
export function onTrashChanged(cb: () => void): () => void {
  window.addEventListener(EVENT, cb);
  return () => window.removeEventListener(EVENT, cb);
}
