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

// ---------------------------------------------------------------------------------------------
// Plan 1.3 keyframe_path. Mirrors modules/camera_path.py (tests/test_camera_path.py checks parity).
// ---------------------------------------------------------------------------------------------
export const PROFILE_S0 = 1.6;
export const PROFILE_S1 = 0.8;

// Global time profile: cubic Hermite, p(0)=0, p(1)=1, start slope 1.6, end slope 0.8.
export function hermiteProfile(t: number, s0: number = PROFILE_S0, s1: number = PROFILE_S1): number {
  const x = Math.min(1, Math.max(0, t));
  const x2 = x * x;
  const x3 = x2 * x;
  return (x3 - 2 * x2 + x) * s0 + (-2 * x3 + 3 * x2) + (x3 - x2) * s1;
}

// Fritsch-Carlson monotone cubic tangents.
export function monotoneSlopes(ts: number[], ys: number[]): number[] {
  const n = ts.length;
  if (n === 2) {
    const d = (ys[1] - ys[0]) / (ts[1] - ts[0]);
    return [d, d];
  }
  const delta: number[] = [];
  for (let k = 0; k < n - 1; k++) delta.push((ys[k + 1] - ys[k]) / (ts[k + 1] - ts[k]));
  const m: number[] = new Array(n).fill(0);
  m[0] = delta[0];
  m[n - 1] = delta[n - 2];
  for (let k = 1; k < n - 1; k++) {
    m[k] = delta[k - 1] * delta[k] <= 0 ? 0 : (delta[k - 1] + delta[k]) / 2;
  }
  for (let k = 0; k < n - 1; k++) {
    if (delta[k] === 0) {
      m[k] = 0;
      m[k + 1] = 0;
      continue;
    }
    let a = m[k] / delta[k];
    let b = m[k + 1] / delta[k];
    if (a < 0) { m[k] = 0; a = 0; }
    if (b < 0) { m[k + 1] = 0; b = 0; }
    const s = a * a + b * b;
    if (s > 9) {
      const tau = 3 / Math.sqrt(s);
      m[k] = tau * a * delta[k];
      m[k + 1] = tau * b * delta[k];
    }
  }
  return m;
}

export function monotoneEval(ts: number[], ys: number[], ms: number[], x: number): number {
  if (x <= ts[0]) return ys[0];
  if (x >= ts[ts.length - 1]) return ys[ys.length - 1];
  let k = 0;
  while (k < ts.length - 2 && x > ts[k + 1]) k++;
  const h = ts[k + 1] - ts[k];
  const s = (x - ts[k]) / h;
  const s2 = s * s;
  const s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * ys[k] + (s3 - 2 * s2 + s) * h * ms[k]
    + (-2 * s3 + 3 * s2) * ys[k + 1] + (s3 - s2) * h * ms[k + 1];
}

export type Keyframe = { t: number; cx: number; cy: number; zoom: number };

// Camera at raw shot progress t (0..1): time profile first, then per-channel monotone cubic.
export function keyframePathAt(keyframes: Keyframe[], t: number) {
  const u = hermiteProfile(t);
  const ts = keyframes.map((k) => k.t);
  const channel = (c: 'cx' | 'cy' | 'zoom') => {
    const ys = keyframes.map((k) => k[c]);
    return monotoneEval(ts, ys, monotoneSlopes(ts, ys), u);
  };
  return { cx: channel('cx'), cy: channel('cy'), zoom: channel('zoom'), roll: 0 };
}

// ---------------------------------------------------------------------------------------------
// Projection "rotate" (addendum 1.3.1). Mirrors rotate_homography in modules/camera_path.py: the
// camera turns in place to point at photo pixel (cx, cy); a pure rotation is exactly a homography
// of one photo, so straight lines stay straight. Row-major 3x3, OUTPUT px -> SOURCE px.
// ---------------------------------------------------------------------------------------------
export const DEFAULT_HFOV_DEG = 75;
export type Mat3 = number[][];

const mul3 = (a: Mat3, b: Mat3): Mat3 =>
  [0, 1, 2].map((i) => [0, 1, 2].map((j) => a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]));

export function baseWindow(sourceW: number, sourceH: number, outW: number, outH: number): [number, number] {
  const aSrc = sourceW / sourceH;
  const aOut = outW / outH;
  return aSrc >= aOut ? [aOut / aSrc, 1] : [1, aSrc / aOut];
}

export function rotateHomography(
  cx: number, cy: number, zoom: number, sourceW: number, sourceH: number, outW: number, outH: number,
  hfovDeg: number = DEFAULT_HFOV_DEG
): Mat3 {
  const f = (sourceW / 2) / Math.tan((hfovDeg * Math.PI) / 180 / 2);
  const [bw] = baseWindow(sourceW, sourceH, outW, outH);
  const fv = (f * zoom * outW) / (bw * sourceW);
  const x0 = cx * sourceW - sourceW / 2;
  const y0 = cy * sourceH - sourceH / 2;
  const yaw = Math.atan2(x0, f);
  const pitch = Math.atan2(y0, Math.hypot(x0, f));
  const cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const r: Mat3 = [[cyw, -syw * sp, syw * cp], [0, cp, sp], [-syw, -cyw * sp, cyw * cp]];
  const kvInv: Mat3 = [[1 / fv, 0, -outW / 2 / fv], [0, 1 / fv, -outH / 2 / fv], [0, 0, 1]];
  const ks: Mat3 = [[f, 0, sourceW / 2], [0, f, sourceH / 2], [0, 0, 1]];
  return mul3(ks, mul3(r, kvInv));
}

export function invert3(m: Mat3): Mat3 {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

// CSS matrix3d (column-major) for a 2D homography h mapping element px -> output px.
export function homographyToMatrix3d(h: Mat3): string {
  const n = h.map((row) => row.map((v) => v / h[2][2]));
  return `matrix3d(${[n[0][0], n[1][0], 0, n[2][0], n[0][1], n[1][1], 0, n[2][1], 0, 0, 1, 0, n[0][2], n[1][2], 0, n[2][2]].join(',')})`;
}
