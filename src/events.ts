import { emit, listen } from "@tauri-apps/api/event";
import { isDesktop } from "./api";

let source: EventSource | null = null;
let broadcastChannel: BroadcastChannel | null = null;

const getBroadcastChannel = () => {
  if (typeof BroadcastChannel === "undefined") return null;
  return (broadcastChannel ??= new BroadcastChannel("ghostreel"));
};

/** One stream for the page: the server sends every event to every connection, so one is enough. */
const events = () => (source ??= new EventSource("/api/events"));

/**
 * Emit an event across windows / tabs.
 */
export async function emitEvent<T>(name: string, payload: T): Promise<void> {
  if (isDesktop) {
    await emit(name, payload).catch(() => {});
  } else {
    try {
      getBroadcastChannel()?.postMessage({ name, payload });
    } catch {}
  }
}

/**
 * Subscribe to a backend event — a Tauri event in the desktop app, a named SSE event in a
 * browser. Also listens to BroadcastChannel for cross-tab messages. Returns the unsubscribe,
 * which is safe to call before the subscription is live.
 */
export function onEvent<T>(name: string, cb: (payload: T) => void): () => void {
  if (isDesktop) {
    let stop: (() => void) | null = null;
    let cancelled = false;
    listen<T>(name, (e) => cb(e.payload)).then((un) => {
      if (cancelled) un();
      else stop = un;
    });
    return () => {
      cancelled = true;
      stop?.();
    };
  }
  const handler = (e: Event) => {
    try {
      cb(JSON.parse((e as MessageEvent<string>).data) as T);
    } catch {
      // A payload we can't read is worth less than the stream that carries the next one.
    }
  };
  const src = events();
  src.addEventListener(name, handler);

  const bc = getBroadcastChannel();
  const bcHandler = (e: MessageEvent) => {
    if (e.data?.name === name) {
      cb(e.data.payload);
    }
  };
  bc?.addEventListener("message", bcHandler);

  return () => {
    src.removeEventListener(name, handler);
    bc?.removeEventListener("message", bcHandler);
  };
}
