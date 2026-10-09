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
