import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import SearchWindow from "./SearchWindow";
import "./styles.css";

function isSearchWindow(): boolean {
  if (typeof window === "undefined") return false;
  const searchParams = new URLSearchParams(window.location.search);
  if (searchParams.get("window") === "search") return true;
  if (window.location.hash === "#search") return true;
  try {
    const label = (window as unknown as { __TAURI_INTERNALS__?: { metadata?: { currentWindow?: { label?: string } } } })
      ?.__TAURI_INTERNALS__?.metadata?.currentWindow?.label;
    if (label === "search") return true;
  } catch {}
  return false;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {isSearchWindow() ? <SearchWindow /> : <App />}
  </React.StrictMode>,
);
