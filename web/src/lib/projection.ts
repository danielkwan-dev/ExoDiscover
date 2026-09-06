/**
 * Placing catalogue objects in space, with Earth at the origin.
 *
 * Written by hand rather than pulled from a 3D engine: the whole job is a
 * spherical-to-Cartesian conversion, two rotations and a perspective divide.
 * A library would add several hundred kilobytes to the bundle and bring its own
 * recognisable house style, and there is nothing here worth outsourcing.
 *
 * Right-handed coordinates: +x toward RA 0h on the celestial equator, +z toward
 * the north celestial pole, distances in parsecs.
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface View {
  width: number;
  height: number;
  /**
   * How far back the camera sits. It looks from -z toward +z, so **smaller z
   * is nearer the viewer** and draws larger. Anything at or beyond
   * `z = -cameraDistance` is behind the camera.
   */
  cameraDistance: number;
  /** Larger values flatten the perspective. */
  focalLength: number;
}

export interface Projected {
  x: number;
  y: number;
  /** Perspective factor; use it to size a point so nearer reads as bigger. */
  scale: number;
}

const DEG = Math.PI / 180;

/** Right ascension and declination in degrees, distance in parsecs. */
export function celestialToCartesian(raDeg: number, decDeg: number, distance: number): Vec3 {
  const ra = raDeg * DEG;
  const dec = decDeg * DEG;
  const cosDec = Math.cos(dec);
  return {
    x: distance * cosDec * Math.cos(ra),
    y: distance * cosDec * Math.sin(ra),
    z: distance * Math.sin(dec),
  };
}

/**
 * Yaw about the vertical axis, then pitch about the horizontal one.
 *
 * Positive yaw carries +x toward +z.
 */
export function rotate(v: Vec3, yaw: number, pitch: number): Vec3 {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const x1 = v.x * cy - v.z * sy;
  const z1 = v.x * sy + v.z * cy;

  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const y2 = v.y * cp - z1 * sp;
  const z2 = v.y * sp + z1 * cp;

  return { x: x1, y: y2, z: z2 };
}

/**
 * Perspective projection onto the canvas, or null if the point sits at or
 * behind the camera.
 *
 * The null case matters: past the camera plane the divisor turns negative and
 * the point would be painted, mirrored, on the wrong side of the screen -- a
 * star apparently in front of Earth that is really behind the viewer.
 */
export function project(v: Vec3, view: View): Projected | null {
  const depth = view.cameraDistance + v.z;
  if (depth <= 1e-6) return null;

  const scale = view.focalLength / depth;
  return {
    x: view.width / 2 + v.x * scale,
    y: view.height / 2 - v.y * scale,
    scale,
  };
}
