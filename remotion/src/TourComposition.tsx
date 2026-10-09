import { Composition, Sequence, getInputProps } from 'remotion';
import React from 'react';
import { KenBurnsShot } from './KenBurnsShot';

export const TourComposition: React.FC = () => {
  const props = getInputProps();
  if (!props.shots) return null; // Default render check
  
  const shots = props.shots as any[];
  
  let currentFrame = 0;
  return (
    <>
      {shots.map((shot, idx) => {
        const start = currentFrame;
        currentFrame += shot.durationFrames;
        return (
          <Sequence key={shot.id} from={start} durationInFrames={shot.durationFrames}>
            <KenBurnsShot 
              durationFrames={shot.durationFrames}
              image={shot.image}
              source={shot.source}
              camera={shot.camera}
              width={props.width as number}
              height={props.height as number}
            />
          </Sequence>
        );
      })}
    </>
  );
};
