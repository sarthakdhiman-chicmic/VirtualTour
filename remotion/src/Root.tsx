import { Composition, getInputProps } from 'remotion';
import React from 'react';
import { TourComposition } from './TourComposition';

export const Root: React.FC = () => {
  const props = getInputProps();
  const fps = (props.fps as number) || 30;
  const width = (props.width as number) || 1920;
  const height = (props.height as number) || 1080;
  const duration = props.shots ? 
    (props.shots as any[]).reduce((a, b) => a + b.durationFrames, 0) : 30;

  return (
    <Composition
      id="TourComposition"
      component={TourComposition}
      durationInFrames={Math.max(1, duration)}
      fps={fps}
      width={width}
      height={height}
      defaultProps={props}
    />
  );
};
