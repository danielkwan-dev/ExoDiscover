import { describe, expect, it } from "vitest";

import {
  identity,
  multiply,
  perspective,
  rotationX,
  rotationY,
  transformPoint,
  translation,
} from "./mat4";

describe("identity", () => {
  it("leaves a point where it is", () => {
    const p = transformPoint(identity(), [3, -4, 12, 1]);
    expect(p.slice(0, 3)).toEqual([3, -4, 12]);
  });

  it("is neutral under multiplication", () => {
    const m = rotationY(0.7);
    expect(Array.from(multiply(m, identity()))).toEqual(Array.from(m));
    expect(Array.from(multiply(identity(), m))).toEqual(Array.from(m));
  });
});

describe("translation", () => {
  it("moves a point by the offset", () => {
    const p = transformPoint(translation(1, 2, 3), [10, 20, 30, 1]);
    expect(p.slice(0, 3)).toEqual([11, 22, 33]);
  });

  it("does not move a direction, only a position", () => {
    // w = 0 marks a direction; translating one would be wrong.
    const d = transformPoint(translation(1, 2, 3), [10, 20, 30, 0]);
    expect(d.slice(0, 3)).toEqual([10, 20, 30]);
  });
});

describe("rotation", () => {
  it("a quarter turn about y takes +x to -z", () => {
    const p = transformPoint(rotationY(Math.PI / 2), [1, 0, 0, 1]);
    expect(p[0]).toBeCloseTo(0);
    expect(p[2]).toBeCloseTo(-1);
  });

  it("a quarter turn about x takes +y to +z", () => {
    const p = transformPoint(rotationX(Math.PI / 2), [0, 1, 0, 1]);
    expect(p[1]).toBeCloseTo(0);
    expect(p[2]).toBeCloseTo(1);
  });

  it("preserves length", () => {
    const m = multiply(rotationY(1.1), rotationX(-0.4));
    const p = transformPoint(m, [3, -4, 12, 1]);
    expect(Math.hypot(p[0], p[1], p[2])).toBeCloseTo(13);
  });
});

describe("perspective", () => {
  const m = perspective(Math.PI / 3, 16 / 9, 0.1, 1000);

  it("puts a point on the view axis at the centre after the divide", () => {
    const p = transformPoint(m, [0, 0, -10, 1]);
    expect(p[0] / p[3]).toBeCloseTo(0);
    expect(p[1] / p[3]).toBeCloseTo(0);
  });

  it("shrinks a fixed offset as it recedes", () => {
    const near = transformPoint(m, [1, 0, -5, 1]);
    const far = transformPoint(m, [1, 0, -50, 1]);
    expect(Math.abs(near[0] / near[3])).toBeGreaterThan(Math.abs(far[0] / far[3]));
  });

  it("keeps a point in front of the camera inside the clip volume", () => {
    const p = transformPoint(m, [0, 0, -10, 1]);
    const ndcZ = p[2] / p[3];
    expect(ndcZ).toBeGreaterThan(-1);
    expect(ndcZ).toBeLessThan(1);
  });
});

describe("multiply", () => {
  it("applies the right-hand matrix first", () => {
    // Rotate, then translate: the translation must not be rotated.
    const m = multiply(translation(10, 0, 0), rotationY(Math.PI / 2));
    const p = transformPoint(m, [1, 0, 0, 1]);
    expect(p[0]).toBeCloseTo(10);
    expect(p[2]).toBeCloseTo(-1);
  });
});
