import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api, type SkyObject } from "../lib/api";
import { celestialToCartesian, project, rotate, type Vec3 } from "../lib/projection";
import { Failed, Loading } from "../components/ApiState";

/**
 * Kepler's field, in three dimensions, with Earth at the origin.
 *
 * The canvas is dark, the stars bloom, and the survey turns. None of that is
 * decoration borrowed from a landing page: the sky is black, a point source
 * genuinely blooms in any real optic, and a still image of a 3D point cloud is
 * ambiguous about depth in a way that a slowly rotating one is not. The paper
 * ground and hairline rules of the rest of the site stop at the canvas edge.
 *
 * The shape drawn here is the honest one. Kepler stared at a single 22x16
 * degree window for four years, so the objects form a narrow beam punched
 * thousands of light years deep, not a sphere of neighbours.
 */

const LY_PER_PC = 3.261563777;

const COLOURS: Record<string, string> = {
  CONFIRMED: "#f2ece0",
  CANDIDATE: "#ff8b45",
  "FALSE POSITIVE": "#6b7487",
};

const ORDER = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"];

/** Radians per second of idle rotation: slow enough to read, not to distract. */
const DRIFT = 0.055;

interface Placed {
  o: SkyObject;
  v: Vec3;
  ly: number;
}

/**
 * A star, pre-rendered once per colour.
 *
 * The bloom is a radial gradient, which is what a point source does to an
 * optic. Building one gradient per star per frame would cost roughly nine
 * thousand allocations at sixty frames a second; drawing a cached sprite
 * scaled to size costs one blit.
 */
function makeStarSprite(colour: string): HTMLCanvasElement {
  // Close to the size it is actually drawn at (4-15 px). A 64 px source
  // resampled down for every star costs far more than it looks, and none of
  // the extra detail survives the downscale.
  const size = 20;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d")!;
  // Under additive blending thousands of these overlap, so each one has to
  // contribute very little light: a small bright core and a fast falloff.
  // A generous bloom here saturates the dense part of the beam to flat white
  // and the disposition colours stop being readable at all.
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, colour);
  g.addColorStop(0.06, `${colour}cc`);
  g.addColorStop(0.18, `${colour}33`);
  g.addColorStop(0.45, `${colour}0f`);
  g.addColorStop(1, `${colour}00`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return c;
}

const SkyMap = () => {
  const { data, isPending, error } = useQuery({
    queryKey: ["skymap"],
    queryFn: api.skymap,
    retry: false,
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Orientation lives in refs, not state: the animation loop reads it every
  // frame, and routing that through React would re-render the whole page 60
  // times a second to move two numbers.
  const spin = useRef(0);
  const pitch = useRef(0);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const [drifting, setDrifting] = useState(true);

  // 95% of the catalogue sits inside 6,000 ly. Opening on the full range lets
  // a handful of distant outliers stretch the frame until the bulk of the data
  // is a speck; the slider still reaches them.
  const [maxLy, setMaxLy] = useState(6000);
  const [kinds, setKinds] = useState<string[]>(ORDER);
  const [minProb, setMinProb] = useState(0);

  const placed = useMemo<Placed[]>(() => {
    if (!data) return [];
    return data.objects.map((o) => ({
      o,
      v: celestialToCartesian(o.ra, o.dec, o.dist_pc),
      ly: o.dist_pc * LY_PER_PC,
    }));
  }, [data]);

  const visible = useMemo(
    () =>
      placed.filter(
        (p) =>
          p.ly <= maxLy && kinds.includes(p.o.disposition) && p.o.probability >= minProb,
      ),
    [placed, maxLy, kinds, minProb],
  );

  /**
   * Turn the survey's own axis into the plane of the screen.
   *
   * Kepler pointed at one patch, so the cloud is a needle aimed from Earth
   * toward roughly RA 290, Dec 44. Viewed down that axis it is a dot; laid into
   * the screen plane it spans the frame. Yawing by atan2(-z, x) sends the axis
   * to z = 0, which is exactly that plane.
   */
  const baseYaw = useMemo(() => {
    if (!placed.length) return 0;
    const n = placed.length;
    const ax = placed.reduce((a, p) => a + p.v.x, 0) / n;
    const az = placed.reduce((a, p) => a + p.v.z, 0) / n;
    return Math.atan2(-az, ax);
  }, [placed]);

  const sprites = useMemo(() => {
    const out: Record<string, HTMLCanvasElement> = {};
    for (const [kind, colour] of Object.entries(COLOURS)) {
      out[kind] = makeStarSprite(colour);
    }
    return out;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !visible.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Geometry that only changes when the filters do, hoisted out of the frame
    // loop: centroid, the offsets around it, and the extent to fit.
    const n = visible.length;
    const cx = visible.reduce((a, p) => a + p.v.x, 0) / n;
    const cy = visible.reduce((a, p) => a + p.v.y, 0) / n;
    const cz = visible.reduce((a, p) => a + p.v.z, 0) / n;
    const rel = visible.map((p) => ({
      p,
      c: { x: p.v.x - cx, y: p.v.y - cy, z: p.v.z - cz },
    }));
    const earthRel = { x: -cx, y: -cy, z: -cz };
    const spread = Math.max(
      Math.hypot(earthRel.x, earthRel.y, earthRel.z),
      ...rel.map(({ c }) => Math.hypot(c.x, c.y, c.z)),
    );
    const cameraDistance = spread * 2.6;

    let frame = 0;
    let last = performance.now();

    const render = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (!drag.current && drifting) spin.current += DRIFT * dt;

      const dpr = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Deep space is not uniformly black: a faint wash toward the centre gives
      // the field somewhere to sit.
      const bg = ctx.createRadialGradient(
        width / 2,
        height / 2,
        0,
        width / 2,
        height / 2,
        Math.max(width, height) * 0.75,
      );
      bg.addColorStop(0, "#12161f");
      bg.addColorStop(1, "#05070b");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      const view = {
        width,
        height,
        cameraDistance,
        focalLength: Math.min(width, height) * 1.09,
      };
      const yaw = baseYaw + spin.current;

      // Additive blending: overlapping stars accumulate light the way they do
      // on a sensor, so the dense core of the beam reads as brighter.
      //
      // It also makes the painter's algorithm unnecessary. Addition commutes,
      // so the result is identical whatever order the sprites arrive in, and
      // sorting nine thousand points every frame was costing more than the
      // drawing did. Depth still reads, through size and brightness.
      ctx.globalCompositeOperation = "lighter";

      let alpha = -1;
      for (const { p, c } of rel) {
        const r = rotate(c, yaw, pitch.current);
        const s = project(r, view);
        if (!s) continue;
        const sprite = sprites[p.o.disposition];
        if (!sprite) continue;

        const depthFactor = cameraDistance / (cameraDistance + r.z);
        const radius = Math.max(
          0.7,
          Math.min(3.4, (p.o.koi_prad ?? 2) ** 0.3 * depthFactor),
        );
        // Quantised so the canvas state changes a handful of times per frame
        // rather than nine thousand.
        const a =
          Math.round((0.1 + 0.22 * Math.min(1, (depthFactor - 0.6) / 0.9)) * 50) / 50;
        if (a !== alpha) {
          ctx.globalAlpha = a;
          alpha = a;
        }
        // Whole pixels, which keeps the blit on the fast path.
        const d = Math.round(radius * 4.5);
        ctx.drawImage(sprite, s.x - d / 2, s.y - d / 2, d, d);
      }

      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;

      const origin = project(rotate(earthRel, yaw, pitch.current), view);
      if (origin) {
        ctx.strokeStyle = "#8fb6e0";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(origin.x, origin.y, 4, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = "#8fb6e0";
        ctx.font = "11px ui-monospace, Menlo, monospace";
        ctx.fillText("Earth", origin.x + 9, origin.y + 4);
      }

      frame = requestAnimationFrame(render);
    };

    frame = requestAnimationFrame(render);
    return () => cancelAnimationFrame(frame);
  }, [visible, baseYaw, sprites, drifting]);

  const toggle = (kind: string) =>
    setKinds((k) => (k.includes(kind) ? k.filter((x) => x !== kind) : [...k, kind]));

  return (
    <div className="container max-w-5xl px-4 py-12">
      <p className="label">Kepler field &middot; Earth at the origin</p>
      <h1 className="mt-2 text-3xl">The survey in three dimensions</h1>
      <p className="mt-3 max-w-3xl text-muted-foreground">
        Every object Kepler catalogued that can be placed in space, at its real right
        ascension, declination and distance. The catalogue carries no distance, so it is
        joined from the Kepler stellar table&rsquo;s Gaia-derived values &mdash; measured
        to about &plusmn;19%, and missing for 120 objects, which are not drawn rather
        than being put somewhere they are not.
      </p>

      {isPending && <Loading label="Loading sky map" />}
      {error && <Failed error={error as Error} />}

      {data && (
        <>
          <div className="mt-8 panel p-4">
            <canvas
              ref={canvasRef}
              className="block h-[520px] w-full cursor-grab rounded-sm active:cursor-grabbing"
              onPointerDown={(e) => {
                drag.current = { x: e.clientX, y: e.clientY };
                e.currentTarget.setPointerCapture(e.pointerId);
              }}
              onPointerMove={(e) => {
                if (!drag.current) return;
                spin.current += (e.clientX - drag.current.x) * 0.005;
                pitch.current = Math.max(
                  -1.4,
                  Math.min(1.4, pitch.current + (e.clientY - drag.current.y) * 0.005),
                );
                drag.current = { x: e.clientX, y: e.clientY };
              }}
              onPointerUp={() => (drag.current = null)}
            />
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <p className="text-xs text-muted-foreground">
                Drag to rotate. Showing{" "}
                <span className="figure">{visible.length.toLocaleString()}</span> of{" "}
                <span className="figure">{data.n.toLocaleString()}</span> objects.
              </p>
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={drifting}
                  onChange={(e) => setDrifting(e.target.checked)}
                  className="accent-accent"
                />
                drift
              </label>
            </div>
          </div>

          <div className="mt-6 grid gap-6 md:grid-cols-3">
            <div>
              <label className="label" htmlFor="dist">
                Within <span className="figure">{maxLy.toLocaleString()}</span> light years
              </label>
              <input
                id="dist"
                type="range"
                min={400}
                max={20000}
                step={100}
                value={maxLy}
                onChange={(e) => setMaxLy(Number(e.target.value))}
                className="mt-2 w-full accent-accent"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                The nearest is 391 ly and the median 2,728. Kepler observed a faint,
                distant field &mdash; nothing here is a neighbour.
              </p>
            </div>

            <div>
              <span className="label">Disposition</span>
              <ul className="mt-2 space-y-1">
                {ORDER.map((kind) => (
                  <li key={kind}>
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={kinds.includes(kind)}
                        onChange={() => toggle(kind)}
                        className="accent-accent"
                      />
                      <span
                        aria-hidden
                        className="inline-block h-2 w-2 rounded-full"
                        style={{ background: COLOURS[kind] }}
                      />
                      {kind.toLowerCase()}
                    </label>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <label className="label" htmlFor="prob">
                Model probability &ge;{" "}
                <span className="figure">{minProb.toFixed(2)}</span>
              </label>
              <input
                id="prob"
                type="range"
                min={0}
                max={0.99}
                step={0.01}
                value={minProb}
                onChange={(e) => setMinProb(Number(e.target.value))}
                className="mt-2 w-full accent-accent"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Raise it to keep only what the classifier considers planet-like,
                including among the unvetted candidates.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default SkyMap;
