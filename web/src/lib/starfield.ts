/**
 * A WebGL point-sprite star field.
 *
 * Canvas 2D managed about twenty frames a second with nine thousand sprites in
 * a small panel; this scene is full-screen, always moving, and carries nearly
 * seventeen thousand objects. That is routine for the GPU and out of reach for
 * a 2D context, so the renderer is WebGL -- but only just: one vertex shader,
 * one fragment shader, three buffers and a matrix. No engine, and the bundle
 * grows by nothing.
 *
 * Each star is a single GL_POINT. The vertex shader scales it with distance so
 * nearer objects are larger, and the fragment shader shades it as a soft disc
 * so it reads as a point of light rather than a square.
 */

import { identity, multiply, perspective, rotationX, rotationY, translation } from "./mat4";

const VERTEX = `
  attribute vec3 position;
  attribute vec3 colour;
  attribute float size;

  uniform mat4 mvp;
  uniform float pixelScale;

  varying vec3 vColour;
  varying float vFade;

  void main() {
    vec4 clip = mvp * vec4(position, 1.0);
    gl_Position = clip;

    // Perspective size: w is the view-space depth, so dividing by it makes a
    // star shrink with distance the way its apparent brightness does.
    float depth = max(clip.w, 0.0001);
    gl_PointSize = clamp(size * pixelScale / depth, 1.0, 26.0);

    // Fade the most distant objects rather than clipping them abruptly.
    vFade = clamp(1.6 - depth / 6000.0, 0.12, 1.0);
    vColour = colour;
  }
`;

const FRAGMENT = `
  precision mediump float;

  varying vec3 vColour;
  varying float vFade;

  void main() {
    // gl_PointCoord runs 0..1 across the point; shade it as a soft disc.
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d) * 2.0;
    if (r > 1.0) discard;

    float core = smoothstep(1.0, 0.0, r);
    float glow = pow(core, 3.0);
    gl_FragColor = vec4(vColour * (0.35 + 0.65 * glow), glow * vFade);
  }
`;

export interface Camera {
  /** Where the viewer is, in parsecs, in the same frame as the stars. */
  position: [number, number, number];
  yaw: number;
  pitch: number;
  fovY: number;
}

export interface Starfield {
  /** Upload a new set of stars. Positions are parsecs, colours 0..1. */
  setStars(positions: Float32Array, colours: Float32Array, sizes: Float32Array): void;
  draw(camera: Camera, width: number, height: number): void;
  /** The matrix used for the last draw, for hit-testing against the same view. */
  viewProjection(camera: Camera, aspect: number): Float32Array;
  destroy(): void;
}

function compile(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`shader failed to compile: ${log}`);
  }
  return shader;
}

/** Camera transform: translate the world by -eye, then yaw, then pitch. */
export function viewMatrix(camera: Camera) {
  const [x, y, z] = camera.position;
  return multiply(
    multiply(rotationX(camera.pitch), rotationY(camera.yaw)),
    translation(-x, -y, -z),
  );
}

export function createStarfield(canvas: HTMLCanvasElement): Starfield | null {
  const gl = canvas.getContext("webgl", {
    antialias: false,
    alpha: false,
    premultipliedAlpha: false,
  });
  if (!gl) return null;

  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`program failed to link: ${gl.getProgramInfoLog(program)}`);
  }
  gl.useProgram(program);

  const buffers = {
    position: gl.createBuffer()!,
    colour: gl.createBuffer()!,
    size: gl.createBuffer()!,
  };
  const attribs = {
    position: gl.getAttribLocation(program, "position"),
    colour: gl.getAttribLocation(program, "colour"),
    size: gl.getAttribLocation(program, "size"),
  };
  const uniforms = {
    mvp: gl.getUniformLocation(program, "mvp"),
    pixelScale: gl.getUniformLocation(program, "pixelScale"),
  };

  let count = 0;

  // Additive blending, so overlapping stars accumulate light rather than
  // occluding one another. It also removes any need to depth-sort: addition
  // commutes, so the result is the same in any order.
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

  const bind = (buffer: WebGLBuffer, location: number, componentsPerVertex: number) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, componentsPerVertex, gl.FLOAT, false, 0, 0);
  };

  const projectionFor = (camera: Camera, aspect: number) =>
    multiply(perspective(camera.fovY, aspect, 0.5, 200000), viewMatrix(camera));

  return {
    setStars(positions, colours, sizes) {
      count = positions.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.colour);
      gl.bufferData(gl.ARRAY_BUFFER, colours, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.size);
      gl.bufferData(gl.ARRAY_BUFFER, sizes, gl.STATIC_DRAW);
    },

    viewProjection(camera, aspect) {
      return projectionFor(camera, aspect);
    },

    draw(camera, width, height) {
      gl.viewport(0, 0, width, height);
      gl.clearColor(0.016, 0.02, 0.031, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (!count) return;

      gl.useProgram(program);
      gl.uniformMatrix4fv(uniforms.mvp, false, projectionFor(camera, width / height));
      gl.uniform1f(uniforms.pixelScale, height * 0.9);

      bind(buffers.position, attribs.position, 3);
      bind(buffers.colour, attribs.colour, 3);
      bind(buffers.size, attribs.size, 1);
      gl.drawArrays(gl.POINTS, 0, count);
    },

    destroy() {
      gl.deleteBuffer(buffers.position);
      gl.deleteBuffer(buffers.colour);
      gl.deleteBuffer(buffers.size);
      gl.deleteProgram(program);
    },
  };
}

export { identity };
