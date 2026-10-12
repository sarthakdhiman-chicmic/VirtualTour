import { windowToTransform, smoothstep, fadeOpacity, easeLinear, easeIn, easeOut, easeInOut, interpolateEasing, interpolateZoom, rollFitScale, parallaxAt } from './math';
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

describe('opacity functions', () => {
  it('smoothstep values', () => {
    expect(smoothstep(0)).toBeCloseTo(0);
    expect(smoothstep(0.5)).toBeCloseTo(0.5);
    expect(smoothstep(1)).toBeCloseTo(1);
    expect(smoothstep(-0.5)).toBeCloseTo(0);
    expect(smoothstep(1.5)).toBeCloseTo(1);
  });
  
  it('fadeOpacity bounds', () => {
    const D_k = 10;
    // first overlap frame strictly between 0 and 1
    const first = fadeOpacity(0, D_k);
    expect(first).toBeGreaterThan(0);
    expect(first).toBeLessThan(1);
    
    // last overlap frame strictly below 1
    const last = fadeOpacity(D_k - 1, D_k);
    expect(last).toBeGreaterThan(0);
    expect(last).toBeLessThan(1);
    
    // frame D_k is exactly 1
    expect(fadeOpacity(D_k, D_k)).toBe(1);
    
    // negative frames are exactly 0
    expect(fadeOpacity(-1, D_k)).toBe(0);
    expect(fadeOpacity(-5, D_k)).toBe(0);
  });
});

describe('easing functions', () => {
  it('linear values at 0, 0.25, 0.5, 0.75, 1', () => {
    expect(easeLinear(0)).toBeCloseTo(0);
    expect(easeLinear(0.25)).toBeCloseTo(0.25);
    expect(easeLinear(0.5)).toBeCloseTo(0.5);
    expect(easeLinear(0.75)).toBeCloseTo(0.75);
    expect(easeLinear(1)).toBeCloseTo(1);
  });

  it('ease_in values at 0, 0.25, 0.5, 0.75, 1', () => {
    expect(easeIn(0)).toBeCloseTo(0);
    expect(easeIn(0.25)).toBeCloseTo(0.0625);
    expect(easeIn(0.5)).toBeCloseTo(0.25);
    expect(easeIn(0.75)).toBeCloseTo(0.5625);
    expect(easeIn(1)).toBeCloseTo(1);
  });

  it('ease_out values at 0, 0.25, 0.5, 0.75, 1', () => {
    expect(easeOut(0)).toBeCloseTo(0);
    expect(easeOut(0.25)).toBeCloseTo(0.4375);
    expect(easeOut(0.5)).toBeCloseTo(0.75);
    expect(easeOut(0.75)).toBeCloseTo(0.9375);
    expect(easeOut(1)).toBeCloseTo(1);
  });

  it('ease_in_out values at 0, 0.25, 0.5, 0.75, 1', () => {
    expect(easeInOut(0)).toBeCloseTo(0);
    expect(easeInOut(0.25)).toBeCloseTo(0.15625);
    expect(easeInOut(0.5)).toBeCloseTo(0.5);
    expect(easeInOut(0.75)).toBeCloseTo(0.84375);
    expect(easeInOut(1)).toBeCloseTo(1);
  });

  it('interpolateEasing delegates properly', () => {
    expect(interpolateEasing(0.25, 'linear')).toBeCloseTo(0.25);
    expect(interpolateEasing(0.25, 'ease_in')).toBeCloseTo(0.0625);
    expect(interpolateEasing(0.25, 'ease_out')).toBeCloseTo(0.4375);
    expect(interpolateEasing(0.25, 'ease_in_out')).toBeCloseTo(0.15625);
    expect(interpolateEasing(0.25)).toBeCloseTo(0.15625); // default ease_in_out
  });
});

describe('interpolateZoom', () => {
  it('hits the endpoints', () => {
    expect(interpolateZoom(1.0, 1.22, 0)).toBeCloseTo(1.0);
    expect(interpolateZoom(1.0, 1.22, 1)).toBeCloseTo(1.22);
  });
  it('interpolates in log space (constant ratio per step)', () => {
    expect(interpolateZoom(1.0, 1.21, 0.5)).toBeCloseTo(1.1);
    expect(interpolateZoom(1.2, 1.0, 0.5)).toBeCloseTo(Math.sqrt(1.2));
  });
});

describe('rollFitScale', () => {
  it('is 1 with no roll', () => {
    expect(rollFitScale(0, 1920, 1080)).toBeCloseTo(1);
  });
  it('covers the rolled 16:9 frame', () => {
    const r = (1.5 * Math.PI) / 180;
    // rolled frame's bounding box must fit in the window scaled by k
    const k = rollFitScale(r, 1920, 1080);
    const bw = 1920 * Math.cos(r) + 1080 * Math.sin(r);
    const bh = 1920 * Math.sin(r) + 1080 * Math.cos(r);
    expect(bw / k).toBeLessThanOrEqual(1920 + 1e-6);
    expect(bh / k).toBeLessThanOrEqual(1080 + 1e-6);
  });
});

describe('parallaxAt', () => {
  const a = { cx: 0.4, cy: 0.5, zoom: 1.25 };
  const b = { cx: 0.57, cy: 0.5, zoom: 1.15 };
  it('is centred on the shot when both ends have spare image', () => {
    expect(parallaxAt(a, b, 0.5, 1920, 1280, 0.025).lateral[0]).toBeCloseTo(0);
    expect(parallaxAt(a, b, 1, 1920, 1280, 0.025).lateral[0]).toBeCloseTo(0.0125);
  });
  it('starts from zero at a zoom-1.0 end', () => {
    const z = { cx: 0.5, cy: 0.5, zoom: 1.0 };
    const e = { cx: 0.55, cy: 0.5, zoom: 1.2 };
    const r = parallaxAt(z, e, 0, 1920, 1280, 0.025);
    expect(r.lateral[0]).toBeCloseTo(0);
    expect(r.radial).toBeCloseTo(0);
  });
});
