import { AbsoluteFill, Img, useCurrentFrame, staticFile } from 'remotion';
import React, { useMemo } from 'react';
import { windowToTransform, interpolateEasing, cameraAt, rollFitScale, keyframePathAt, CameraPoint, Keyframe,
  rotateHomography, invert3, homographyToMatrix3d, DEFAULT_HFOV_DEG,
  CompoundPath, CompoundCamera } from './math';

export const KenBurnsShot: React.FC<{
  durationFrames: number;
  image: string;
  source: { width: number; height: number };
  camera: {
    type?: string;
    // ken_burns
    start?: CameraPoint;
    end?: CameraPoint;
    easing?: string;
    // keyframe_path (plan 1.3)
    move?: string;
    keyframes?: Keyframe[];
    projection?: string; // 'crop' (default) | 'rotate' (addendum 1.3.1)
    hfov_deg?: number;
    // compound (addendum 1.3.2): start + segments + timing
    segments?: CompoundCamera['segments'];
    timing?: CompoundCamera['timing'];
  };
  width: number;
  height: number;
}> = ({ durationFrames, image, source, camera, width, height }) => {
  const frame = useCurrentFrame();
  const rawT = frame / Math.max(1, durationFrames - 1);
  const t = Math.max(0, Math.min(1, rawT));
  const compound = useMemo(
    () => (camera.type === 'compound'
      ? new CompoundPath(camera as unknown as CompoundCamera, source.width, source.height, width, height)
      : null),
    [camera, source.width, source.height, width, height]
  );
  const { cx, cy, zoom, roll } = compound
    ? compound.at(t)
    : camera.type === 'keyframe_path'
    ? keyframePathAt(camera.keyframes as Keyframe[], t)
    : cameraAt(camera.start as CameraPoint, camera.end as CameraPoint,
        interpolateEasing(t, camera.easing ?? 'ease_in_out'));

  if ((camera.type === 'keyframe_path' || camera.type === 'compound') && camera.projection === 'rotate') {
    // camera turning in place: the photo reprojected onto a camera aimed at (cx, cy).
    // NOT deterministic: two full renders differed in ~300 of 2100 frames (one render chunk,
    // mean abs diff 1-2.4 levels); will-change: transform did not fix it (out/tour_v3/report.md)
    const toSource = rotateHomography(cx, cy, zoom, source.width, source.height, width, height,
      camera.hfov_deg ?? DEFAULT_HFOV_DEG);
    return (
      <AbsoluteFill style={{ overflow: 'hidden' }}>
        <div style={{
          position: 'absolute',
          width: source.width,
          height: source.height,
          transformOrigin: '0 0',
          transform: homographyToMatrix3d(invert3(toSource)),
        }}>
          <Img src={staticFile(image)} style={{ width: '100%', height: '100%' }} />
        </div>
      </AbsoluteFill>
    );
  }

  const { scale, translateX, translateY } = windowToTransform(
    source.width, source.height, width, height, cx, cy, zoom
  );

  return (
    <AbsoluteFill style={{ overflow: 'hidden' }}>
      {/* roll about the frame centre, scaled up so the rolled frame stays inside the window */}
      <AbsoluteFill style={{ transform: `rotate(${roll}rad) scale(${rollFitScale(roll, width, height)})` }}>
        <div style={{
          position: 'absolute',
          width: source.width,
          height: source.height,
          transformOrigin: '0 0',
          transform: `scale(${scale}) translate(${translateX}px, ${translateY}px)`
        }}>
          <Img src={staticFile(image)} style={{ width: '100%', height: '100%' }} />
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
