export function windowToTransform(
  sourceW: number,
  sourceH: number,
  outW: number,
  outH: number,
  cx: number,
  cy: number,
  zoom: number
) {
  const outRatio = outW / outH;
  const srcRatio = sourceW / sourceH;
  
  let rectW, rectH;
  if (srcRatio > outRatio) {
    rectW = sourceH * outRatio;
    rectH = sourceH;
  } else {
    rectW = sourceW;
    rectH = sourceW / outRatio;
  }
  
  const w = rectW / zoom;
  const h = rectH / zoom;
  
  const px = cx * sourceW;
  const py = cy * sourceH;
  
  const left = px - w / 2;
  const top = py - h / 2;
  
  // Transform to apply to the image to map [left, top, w, h] to [0, 0, outW, outH]
  const scale = outW / w;
  const translateX = -left;
  const translateY = -top;
  
  return { scale, translateX, translateY };
}

export function smoothstep(x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x * x * (3 - 2 * x);
}

export function fadeOpacity(f: number, D_k: number): number {
  if (D_k <= 0) return 1;
  if (f >= D_k) return 1;
  if (f < 0) return 0;
  return smoothstep((f + 1) / (D_k + 1));
}
