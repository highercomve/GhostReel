import { useEffect, useState, type ImgHTMLAttributes } from "react";
import { mediaUrl } from "./api";

/** Load local images through the same token-protected HTTP server as video playback.
 * WebKit blocks asset:// image URLs from the HTTP origin used by the development UI.
 */
export default function MediaImage({ path, ...props }: ImgHTMLAttributes<HTMLImageElement> & { path: string }) {
  const [image, setImage] = useState<{ path: string; src: string | null } | null>(null);
  useEffect(() => {
    let cancelled = false;
    const url = path.startsWith("data:") ? Promise.resolve(path) : mediaUrl(path);
    url.then((src) => { if (!cancelled) setImage({ path, src }); })
      .catch(() => { if (!cancelled) setImage({ path, src: null }); });
    return () => { cancelled = true; };
  }, [path]);
  if (image?.path !== path || !image.src) {
    const label = image?.path === path ? "Thumbnail unavailable" : "Loading thumbnail…";
    return <span className="media-image-placeholder" role="img" aria-label={label}>{label}</span>;
  }
  return <img {...props} src={image.src} onError={(event) => {
    setImage({ path, src: null });
    props.onError?.(event);
  }} />;
}
