import { AbsoluteFill, Img, useCurrentFrame, staticFile } from 'remotion';
import React from 'react';
import { windowToTransform, interpolateEasing, cameraAt, rollFitScale, keyframePathAt, CameraPoint, Keyframe } from './math';

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
  };
  width: number;
  height: number;
}> = ({ durationFrames, image, source, camera, width, height }) => {
  const frame = useCurrentFrame();
  const rawT = frame / Math.max(1, durationFrames - 1);
  const t = Math.max(0, Math.min(1, rawT));
  const { cx, cy, zoom, roll } = camera.type === 'keyframe_path'
    ? keyframePathAt(camera.keyframes as Keyframe[], t)
    : cameraAt(camera.start as CameraPoint, camera.end as CameraPoint,
        interpolateEasing(t, camera.easing ?? 'ease_in_out'));

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
