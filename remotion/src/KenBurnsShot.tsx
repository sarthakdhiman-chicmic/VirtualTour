import { AbsoluteFill, Img, useCurrentFrame, staticFile } from 'remotion';
import React from 'react';
import { windowToTransform, interpolateEasing, cameraAt, rollFitScale, CameraPoint } from './math';

export const KenBurnsShot: React.FC<{
  durationFrames: number;
  image: string;
  source: { width: number; height: number };
  camera: {
    start: CameraPoint;
    end: CameraPoint;
    easing?: string;
  };
  width: number;
  height: number;
}> = ({ durationFrames, image, source, camera, width, height }) => {
  const frame = useCurrentFrame();
  const rawT = frame / Math.max(1, durationFrames - 1);
  const t = Math.max(0, Math.min(1, rawT));
  const smoothT = interpolateEasing(t, camera.easing ?? 'ease_in_out');

  const { cx, cy, zoom, roll } = cameraAt(camera.start, camera.end, smoothT);

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
