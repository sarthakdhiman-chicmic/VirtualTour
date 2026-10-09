import { AbsoluteFill, Img, useCurrentFrame, staticFile } from 'remotion';
import React from 'react';
import { windowToTransform } from './math';

export const KenBurnsShot: React.FC<{
  durationFrames: number;
  image: string;
  source: { width: number; height: number };
  camera: {
    start: { cx: number; cy: number; zoom: number };
    end: { cx: number; cy: number; zoom: number };
  };
  width: number;
  height: number;
}> = ({ durationFrames, image, source, camera, width, height }) => {
  const frame = useCurrentFrame();
  const rawT = frame / Math.max(1, durationFrames - 1);
  const t = Math.max(0, Math.min(1, rawT));
  const smoothT = t * t * (3 - 2 * t);

  const cx = camera.start.cx + (camera.end.cx - camera.start.cx) * smoothT;
  const cy = camera.start.cy + (camera.end.cy - camera.start.cy) * smoothT;
  const zoom = camera.start.zoom + (camera.end.zoom - camera.start.zoom) * smoothT;

  const { scale, translateX, translateY } = windowToTransform(
    source.width, source.height, width, height, cx, cy, zoom
  );

  return (
    <AbsoluteFill style={{ overflow: 'hidden' }}>
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
  );
};
