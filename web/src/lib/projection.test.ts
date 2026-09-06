import { describe, expect, it } from "vitest";

import { celestialToCartesian, project, rotate, type Vec3 } from "./projection";

const length = (v: Vec3) => Math.hypot(v.x, v.y, v.z);

describe("celestialToCartesian", () => {
  it("puts Earth at the origin", () => {
    const here = celestialToCartesian(0, 0, 0);
    expect(length(here)).toBeCloseTo(0);
  });

  it("places RA 0, Dec 0 along +x", () => {
    const v = celestialToCartesian(0, 0, 1);
    expect(v.x).toBeCloseTo(1);
    expect(v.y).toBeCloseTo(0);
    expect(v.z).toBeCloseTo(0);
  });

  it("places the north celestial pole along +z", () => {
    const v = celestialToCartesian(123, 90, 1);
    expect(v.z).toBeCloseTo(1);
    expect(v.x).toBeCloseTo(0);
    expect(v.y).toBeCloseTo(0);
  });

  it("keeps distance as the magnitude, whatever the direction", () => {
    for (const [ra, dec] of [
      [0, 0],
      [285, 44],
      [301, -12],
      [180, 89],
    ]) {
      expect(length(celestialToCartesian(ra, dec, 830))).toBeCloseTo(830, 6);
    }
  });

  it("separates two objects that share a direction but not a distance", () => {
    const near = celestialToCartesian(290, 45, 400);
    const far = celestialToCartesian(290, 45, 4000);
    expect(length(far) / length(near)).toBeCloseTo(10);
  });
});

describe("rotate", () => {
  const v: Vec3 = { x: 3, y: -4, z: 12 };

  it("is the identity at zero yaw and pitch", () => {
    const r = rotate(v, 0, 0);
    expect(r.x).toBeCloseTo(v.x);
    expect(r.y).toBeCloseTo(v.y);
    expect(r.z).toBeCloseTo(v.z);
  });

  it("preserves length, so rotating never moves a star closer", () => {
    for (const [yaw, pitch] of [
      [0.3, 0],
      [0, -0.9],
      [2.1, 1.2],
      [-1.7, 0.4],
    ]) {
      expect(length(rotate(v, yaw, pitch))).toBeCloseTo(length(v), 6);
    }
  });

  it("a quarter turn of yaw carries +x to +z", () => {
    const r = rotate({ x: 1, y: 0, z: 0 }, Math.PI / 2, 0);
    expect(r.x).toBeCloseTo(0);
    expect(r.z).toBeCloseTo(1);
  });
});

describe("project", () => {
  const view = { width: 800, height: 600, cameraDistance: 1000, focalLength: 900 };

  it("puts a point on the view axis at the centre of the canvas", () => {
    const p = project({ x: 0, y: 0, z: 0 }, view);
    expect(p).not.toBeNull();
    expect(p!.x).toBeCloseTo(400);
    expect(p!.y).toBeCloseTo(300);
  });

  it("draws nearer points larger", () => {
    // The camera looks from -z toward +z, so the smaller z is the nearer one.
    const near = project({ x: 0, y: 0, z: -400 }, view)!;
    const far = project({ x: 0, y: 0, z: 400 }, view)!;
    expect(near.scale).toBeGreaterThan(far.scale);
  });

  it("drops points behind the camera rather than mirroring them", () => {
    // Beyond the camera plane the perspective divide flips sign, which would
    // otherwise paint a star on the opposite side of the canvas.
    expect(project({ x: 10, y: 0, z: -1000 }, view)).toBeNull();
    expect(project({ x: 10, y: 0, z: -2000 }, view)).toBeNull();
  });

  it("moves a point right of the axis to the right of centre", () => {
    const p = project({ x: 100, y: 0, z: 0 }, view)!;
    expect(p.x).toBeGreaterThan(400);
  });

  it("treats +y as up on screen", () => {
    const p = project({ x: 0, y: 100, z: 0 }, view)!;
    expect(p.y).toBeLessThan(300);
  });
});
