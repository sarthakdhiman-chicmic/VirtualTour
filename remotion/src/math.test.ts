import { windowToTransform } from './math';

describe('windowToTransform', () => {
  it('correctly maps 16:9 source to 16:9 output', () => {
    // Zoom 1.0, center
    const res = windowToTransform(1920, 1080, 1920, 1080, 0.5, 0.5, 1.0);
    expect(res.scale).toBeCloseTo(1.0);
    expect(res.translateX).toBeCloseTo(0);
    expect(res.translateY).toBeCloseTo(0);
  });

  it('correctly maps 4:3 source to 16:9 output (pillarboxed source)', () => {
    // 4:3 source (1440x1080) to 16:9 output (1920x1080)
    // The source should be zoomed to fill the width if we zoom = 1.0 (Wait, 1.0 zoom means max inscribed box matching output aspect ratio).
    // For 4:3 source into 16:9, the output ratio is wider. 
    // The inscribed 16:9 rect inside 1440x1080 has width 1440, height 1440 / (16/9) = 810.
    // scale = outW / rectW = 1920 / 1440 = 1.333
    const res = windowToTransform(1440, 1080, 1920, 1080, 0.5, 0.5, 1.0);
    expect(res.scale).toBeCloseTo(1920 / 1440);
    expect(res.translateX).toBeCloseTo(0); // Center: left is 0 for 1440, top is (1080 - 810) / 2 = 135
    expect(res.translateY).toBeCloseTo(-135);
  });
  
  it('correctly handles zoom', () => {
    const res = windowToTransform(1920, 1080, 1920, 1080, 0.5, 0.5, 2.0);
    // w is 1920 / 2 = 960. scale is 1920 / 960 = 2.0
    expect(res.scale).toBeCloseTo(2.0);
    // left is 960 - 480 = 480. translateX is -480
    expect(res.translateX).toBeCloseTo(-480);
    expect(res.translateY).toBeCloseTo(-270);
  });
});
