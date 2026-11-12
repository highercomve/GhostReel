import type { CSSProperties } from "react";
import type { FrameView } from "../api";
import MediaImage from "../MediaImage";

type SrcRect = { x: number; y: number; w: number; h: number };

/**
 * Where a source picture of `srcW`×`srcH` goes so that the source rectangle `r` (source pixels)
 * exactly fills its box: size and offset as percentages of the box. The live player places its
 * videos with this, and a still (a keyframe thumbnail) is placed the same way.
 */
export function cropBox(srcW: number, srcH: number, r: SrcRect) {
  return {
    width: `${(srcW / r.w) * 100}%`,
    height: `${(srcH / r.h) * 100}%`,
    left: `${(-r.x / r.w) * 100}%`,
    top: `${(-r.y / r.h) * 100}%`,
  };
}

/** The rectangle that shows the whole `content` picture, as big as fits, centred in the frame. */
export function fitRect(view: Pick<FrameView, "out_w" | "out_h" | "content">): SrcRect {
  const c = view.content;
  const k = Math.min(view.out_w / c.w, view.out_h / c.h);
  const w = view.out_w / k;
  const h = view.out_h / k;
  return { x: c.x - (w - c.w) / 2, y: c.y - (h - c.h) / 2, w, h };
}

/**
 * A still of the source (a keyframe thumbnail at any size) framed as a clip's plan frames it: in
 * the output frame's shape, each crop filling its band, or the whole picture fitted.
 */
export default function FramedImage({ path, view, alt = "" }: { path: string; view: FrameView; alt?: string }) {
  const bands = view.crops.length > 0 ? view.crops : [fitRect(view)];
  const frame: CSSProperties = { aspectRatio: `${view.out_w} / ${view.out_h}` };
  return (
    <div className="framed-image" style={frame} role="img" aria-label={alt || undefined}>
      {bands.map((r, i) => (
        <div key={i} className="framed-image-band">
          <MediaImage path={path} alt="" loading="lazy" style={{ position: "absolute", maxWidth: "none", objectFit: "fill", ...cropBox(view.src_w, view.src_h, r) }} />
        </div>
      ))}
    </div>
  );
}
