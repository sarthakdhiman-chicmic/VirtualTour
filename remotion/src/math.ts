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

export function easeLinear(t: number): number {
  return t;
}

export function easeIn(t: number): number {
  return t * t;
}

export function easeOut(t: number): number {
  return 1 - (1 - t) * (1 - t);
}

export function easeInOut(t: number): number {
  return t * t * (3 - 2 * t);
}

// Zoom is perceived multiplicatively: interpolating it linearly makes a push-in visibly
// decelerate. Interpolating in log space keeps the apparent speed constant.
export function interpolateZoom(start: number, end: number, t: number): number {
  if (start <= 0 || end <= 0) return start + (end - start) * t;
  return start * Math.pow(end / start, t);
}

export function interpolateEasing(t: number, easing: string = 'ease_in_out'): number {
  switch (easing) {
    case 'linear':
      return easeLinear(t);
    case 'ease_in':
      return easeIn(t);
    case 'ease_out':
      return easeOut(t);
    case 'ease_in_out':
    default:
      return easeInOut(t);
  }
}

// Extra zoom that fits a frame rolled by `roll` radians inside its unrolled window, so a
// rolled shot never shows anything outside the window the planner validated.
export function rollFitScale(roll: number, outW: number, outH: number): number {
  const a = Math.abs(roll);
  return Math.cos(a) + (Math.max(outW, outH) / Math.min(outW, outH)) * Math.sin(a);
}

export type CameraPoint = { cx: number; cy: number; zoom: number; roll_deg?: number };

// Camera state at eased progress `p` (0..1).
export function cameraAt(start: CameraPoint, end: CameraPoint, p: number) {
  const r0 = start.roll_deg ?? 0;
  const r1 = end.roll_deg ?? 0;
  return {
    cx: start.cx + (end.cx - start.cx) * p,
    cy: start.cy + (end.cy - start.cy) * p,
    zoom: interpolateZoom(start.zoom, end.zoom, p),
    roll: ((r0 + (r1 - r0) * p) * Math.PI) / 180,
  };
}

// Parallax offsets for the depth shader, in source uv units (top-left origin).
// The offset grows linearly through the shot from an anchor time: the middle when both ends
// have spare image around the window, otherwise the lower-zoom end (where there is none).
export function parallaxAt(
  start: CameraPoint,
  end: CameraPoint,
  p: number,
  sourceW: number,
  sourceH: number,
  strength: number
) {
  const anchored = Math.min(start.zoom, end.zoom) >= 1.06 ? p - 0.5 : start.zoom <= end.zoom ? p : p - 1;
  const dx = end.cx - start.cx;
  const dy = (end.cy - start.cy) * (sourceH / sourceW); // in source-width units
  const len = Math.hypot(dx, dy);
  const lateral = len > 1e-6
    ? [strength * (dx / len) * anchored, strength * (dy / len) * anchored * (sourceW / sourceH)]
    : [0, 0];
  const radial = strength * 2.5 * Math.sign(Math.log(end.zoom / start.zoom)) * anchored;
  return { lateral, radial };
}
