import { ThreeCanvas } from '@remotion/three';
import { useCurrentFrame, staticFile, delayRender, continueRender } from 'remotion';
import * as THREE from 'three';
import React, { useEffect, useMemo, useState } from 'react';
import { interpolateEasing, cameraAt, rollFitScale, parallaxAt, CameraPoint } from './math';

// Depth parallax on a single photo: the same window/roll framing as KenBurnsShot, plus a
// per-pixel shift scaled by depth around a focus depth, so near objects move against far ones.
// A full-screen shader (no mesh), so nothing tears; near-object edges stretch smoothly instead.
// Raising strength reads more strongly but bends straight lines next to near objects more.
export const PARALLAX_STRENGTH = 0.035;
// Motion blur: every frame integrates the camera over this fraction of a frame interval
// (0.5 = a 180-degree shutter, the film standard). Without it, sharp frames moving ~3 px
// per frame step visibly over fine detail, which reads as judder.
export const SHUTTER = 0.5;
const BLUR_TAPS = 8;

const vertexShader = `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const fragmentShader = `
uniform sampler2D colorMap;
uniform sampler2D depthMap;
uniform vec2 outSize;
uniform float focus;
// camera state when the shutter opened (0) and at this frame (1)
uniform vec2 winCenter[2];
uniform vec2 winSize[2];
uniform float roll[2];
uniform float rollScale[2];
uniform vec2 lateral[2];
uniform float radial[2];
varying vec2 vUv;

// textures are flipped on upload; the maths below uses a top-left origin like the plan
vec4 sampleTop(sampler2D m, vec2 s) { return texture2D(m, vec2(s.x, 1.0 - s.y)); }

// source uv shown at output pixel p (centred, in pixels) for camera state i
vec2 sourceFor(vec2 p, int i) {
  // inverse of CSS rotate(roll) scale(rollScale) about the frame centre
  float c = cos(roll[i]), s = sin(roll[i]);
  vec2 q = vec2(c * p.x + s * p.y, -s * p.x + c * p.y) / rollScale[i];
  vec2 base = winCenter[i] + (q / outSize) * winSize[i];
  // inverse warp: find the source pixel whose depth-shifted position lands here
  vec2 src = base;
  for (int k = 0; k < 5; k++) {
    float d = sampleTop(depthMap, src).r - focus;
    src = base + d * lateral[i] - d * radial[i] * (base - winCenter[i]);
  }
  return src;
}

void main() {
  vec2 p = (vec2(vUv.x, 1.0 - vUv.y) - 0.5) * outSize;
  vec2 a = sourceFor(p, 0);
  vec2 b = sourceFor(p, 1);
  vec4 acc = vec4(0.0);
  for (int k = 0; k < ${BLUR_TAPS}; k++) {
    acc += sampleTop(colorMap, clamp(mix(a, b, (float(k) + 0.5) / float(${BLUR_TAPS})), 0.0, 1.0));
  }
  gl_FragColor = acc / float(${BLUR_TAPS});
}`;

function useTexture(src: string) {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const [handle] = useState(() => delayRender(`texture ${src}`));
  useEffect(() => {
    new THREE.TextureLoader().load(
      src,
      (t) => {
        t.minFilter = THREE.LinearFilter;
        t.generateMipmaps = false;
        t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
        setTexture(t);
        continueRender(handle);
      },
      undefined,
      (err) => { throw new Error(`failed to load ${src}: ${err}`); }
    );
  }, [src, handle]);
  return texture;
}

export const ParallaxShot: React.FC<{
  durationFrames: number;
  image: string;
  depth: { image: string; focus: number };
  source: { width: number; height: number };
  camera: { start: CameraPoint; end: CameraPoint; easing?: string };
  width: number;
  height: number;
}> = ({ durationFrames, image, depth, source, camera, width, height }) => {
  const frame = useCurrentFrame();
  const colorMap = useTexture(staticFile(image));
  const depthMap = useTexture(staticFile(depth.image));

  // window size at zoom 1, as a fraction of the source (same as windowToTransform)
  const outA = width / height;
  const srcA = source.width / source.height;
  const [baseW, baseH] = srcA > outA ? [outA / srcA, 1] : [1, srcA / outA];

  const stateAt = (f: number) => {
    const t = Math.max(0, Math.min(1, f / Math.max(1, durationFrames - 1)));
    const p = interpolateEasing(t, camera.easing ?? 'ease_in_out');
    const { cx, cy, zoom, roll } = cameraAt(camera.start, camera.end, p);
    const { lateral, radial } = parallaxAt(camera.start, camera.end, p, source.width, source.height, PARALLAX_STRENGTH);
    return { cx, cy, zoom, roll, lateral, radial };
  };
  const states = [stateAt(frame - SHUTTER), stateAt(frame)];

  const uniforms = useMemo(() => ({
    colorMap: { value: null as THREE.Texture | null },
    depthMap: { value: null as THREE.Texture | null },
    outSize: { value: new THREE.Vector2(width, height) },
    focus: { value: depth.focus },
    winCenter: { value: [new THREE.Vector2(), new THREE.Vector2()] },
    winSize: { value: [new THREE.Vector2(), new THREE.Vector2()] },
    roll: { value: [0, 0] },
    rollScale: { value: [1, 1] },
    lateral: { value: [new THREE.Vector2(), new THREE.Vector2()] },
    radial: { value: [0, 0] },
  }), [width, height, depth.focus]);
  uniforms.colorMap.value = colorMap;
  uniforms.depthMap.value = depthMap;
  states.forEach((st, i) => {
    uniforms.winCenter.value[i].set(st.cx, st.cy);
    uniforms.winSize.value[i].set(baseW / st.zoom, baseH / st.zoom);
    uniforms.roll.value[i] = st.roll;
    uniforms.rollScale.value[i] = rollFitScale(st.roll, width, height);
    uniforms.lateral.value[i].set(st.lateral[0], st.lateral[1]);
    uniforms.radial.value[i] = st.radial;
  });

  return (
    <ThreeCanvas width={width} height={height} orthographic camera={{ position: [0, 0, 1] }}>
      {colorMap && depthMap ? (
        <mesh frustumCulled={false}>
          <planeGeometry args={[2, 2]} />
          <shaderMaterial vertexShader={vertexShader} fragmentShader={fragmentShader} uniforms={uniforms} />
        </mesh>
      ) : null}
    </ThreeCanvas>
  );
};
