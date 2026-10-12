import { Composition, Sequence, getInputProps, AbsoluteFill, useCurrentFrame } from 'remotion';
import React from 'react';
import { KenBurnsShot } from './KenBurnsShot';
import { fadeOpacity } from './math';

export const TourComposition: React.FC = () => {
  const props = getInputProps();
  if (!props.shots || !props.timeline) return null; // Default render check
  
  const shots = props.shots as any[];
  const timeline = props.timeline as any[];
  const frame = useCurrentFrame();
  
  return (
    <AbsoluteFill style={{ backgroundColor: 'black' }}>
      {shots.map((shot, idx) => {
        const t = timeline[idx];
        const D_k = t.fade_frames;
        
        // The shot starts at t.render_from. So relative frame is:
        const f = frame - t.render_from;
        const opacity = fadeOpacity(f, D_k);
        
        return (
          <Sequence key={shot.id} from={t.render_from} durationInFrames={t.render_frames}>
            <div style={{ opacity, width: '100%', height: '100%', position: 'absolute' }}>
              <KenBurnsShot 
                durationFrames={t.render_frames}
                image={shot.image}
                source={shot.source}
                camera={shot.camera}
                width={props.width as number}
                height={props.height as number}
              />
            </div>
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
