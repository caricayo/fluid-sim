"use strict";

import { getGL, createProgram, createFullscreenVAO, checkFloatSupport, makeFBO, makeDoubleFBO, setViewport, bindFBO } from "./gl-helpers.js";

// GLSL Shaders
const VS_SCREEN = `#version 300 es
layout (location = 0) in vec2 a_pos;
void main(){
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// Utility: read from sampler at pixel space via manual bilinear filter (works with NEAREST textures)
const GLSL_BILERP = `
vec4 bilerp(sampler2D tex, vec2 uv, vec2 resolution) {
  vec2 xy = uv * resolution - 0.5;
  vec2 ixy = floor(xy);
  vec2 f = xy - ixy;
  vec2 base = (ixy + 0.5) / resolution;
  vec2 texel = 1.0 / resolution;
  vec4 c00 = texture(tex, base);
  vec4 c10 = texture(tex, base + vec2(texel.x, 0.0));
  vec4 c01 = texture(tex, base + vec2(0.0, texel.y));
  vec4 c11 = texture(tex, base + texel);
  return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}`;

// Advection (semi-Lagrangian) with dissipation
const FS_ADVECT = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_source; // field to advect (e.g., velocity or dye)
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
uniform vec2 u_resolution;
uniform float u_dt;
uniform float u_dissipation;
${GLSL_BILERP}
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec2 vel = texture(u_velocity, uv).xy;
  vec2 coord = uv - u_dt * vel * vec2(u_resolution.y / u_resolution.x, 1.0); // aspect compensate
  vec4 src = bilerp(u_source, coord, u_resolution);
  frag = src * u_dissipation;
}`;

// Divergence of velocity field
const FS_DIVERGENCE = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec2 L = texture(u_velocity, uv - vec2(u_texelSize.x, 0)).xy;
  vec2 R = texture(u_velocity, uv + vec2(u_texelSize.x, 0)).xy;
  vec2 B = texture(u_velocity, uv - vec2(0, u_texelSize.y)).xy;
  vec2 T = texture(u_velocity, uv + vec2(0, u_texelSize.y)).xy;
  float div = 0.5 * ((R.x - L.x) + (T.y - B.y));
  frag = vec4(div, 0.0, 0.0, 1.0);
}`;

// Jacobi pressure solve
const FS_PRESSURE = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  float L = texture(u_pressure, uv - vec2(u_texelSize.x, 0)).x;
  float R = texture(u_pressure, uv + vec2(u_texelSize.x, 0)).x;
  float B = texture(u_pressure, uv - vec2(0, u_texelSize.y)).x;
  float T = texture(u_pressure, uv + vec2(0, u_texelSize.y)).x;
  float div = texture(u_divergence, uv).x;
  float p = (L + R + B + T - div) * 0.25;
  frag = vec4(p, 0.0, 0.0, 1.0);
}`;

// Subtract pressure gradient
const FS_GRADIENT = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_velocity;
uniform sampler2D u_pressure;
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  float L = texture(u_pressure, uv - vec2(u_texelSize.x, 0)).x;
  float R = texture(u_pressure, uv + vec2(u_texelSize.x, 0)).x;
  float B = texture(u_pressure, uv - vec2(0, u_texelSize.y)).x;
  float T = texture(u_pressure, uv + vec2(0, u_texelSize.y)).x;
  vec2 v = texture(u_velocity, uv).xy;
  vec2 grad = 0.5 * vec2(R - L, T - B);
  frag = vec4(v - grad, 0.0, 1.0);
}`;

// Curl (vorticity) and confinement
const FS_CURL = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_velocity;
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  float L = texture(u_velocity, uv - vec2(u_texelSize.x, 0)).y;
  float R = texture(u_velocity, uv + vec2(u_texelSize.x, 0)).y;
  float B = texture(u_velocity, uv - vec2(0, u_texelSize.y)).x;
  float T = texture(u_velocity, uv + vec2(0, u_texelSize.y)).x;
  float curl = R - L - (T - B);
  frag = vec4(curl, 0.0, 0.0, 1.0);
}`;

const FS_VORTICITY = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_velocity;
uniform sampler2D u_curl;
uniform vec2 u_texelSize;
uniform float u_dt;
uniform float u_eps; // vorticity strength
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  float L = abs(texture(u_curl, uv - vec2(u_texelSize.x, 0)).x);
  float R = abs(texture(u_curl, uv + vec2(u_texelSize.x, 0)).x);
  float B = abs(texture(u_curl, uv - vec2(0, u_texelSize.y)).x);
  float T = abs(texture(u_curl, uv + vec2(0, u_texelSize.y)).x);
  vec2 grad = vec2(R - L, T - B);
  float mag = length(grad) + 1e-5;
  vec2 N = grad / mag;
  float curl = texture(u_curl, uv).x;
  // Perpendicular to N in 2D for confinement force
  vec2 force = u_eps * vec2(N.y, -N.x) * curl;
  vec2 v = texture(u_velocity, uv).xy;
  v += force * u_dt;
  frag = vec4(v, 0.0, 1.0);
}`;

// Splat for dye or velocity
const FS_SPLAT = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_target;
uniform vec2 u_point; // in 0..1
uniform vec3 u_value; // rgb for dye; xy for velocity (z ignored)
uniform float u_radius; // in 0..1, roughly pixels proportion
uniform vec2 u_texelSize;
void main(){
  vec2 uv = gl_FragCoord.xy * u_texelSize;
  vec2 d = uv - u_point;
  float r = u_radius;
  float a = exp(-dot(d, d) / (r*r + 1e-6));
  vec4 base = texture(u_target, uv);
  frag = base + vec4(u_value * a, 1.0);
}`;

// Simple render with gamma correction
const FS_RENDER = `#version 300 es
precision highp float;
out vec4 frag;
uniform sampler2D u_dye;
uniform vec2 u_texelSize;
void main(){
  vec3 c = texture(u_dye, gl_FragCoord.xy * u_texelSize).rgb;
  // soft tonemap
  c = c / (1.0 + max(max(c.r, c.g), c.b));
  // gamma
  c = pow(c, vec3(1.0/2.2));
  frag = vec4(c, 1.0);
}`;

function uniformLoc(gl, prog, name) {
  const loc = gl.getUniformLocation(prog, name);
  if (loc === null) throw new Error(`Uniform not found: ${name}`);
  return loc;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

export class FluidSim {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    const gl = getGL(canvas);
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;

    if (!checkFloatSupport(gl)) {
      throw new Error("Required float framebuffer support missing (EXT_color_buffer_float)");
    }

    // Settings
    this.settings = {
      dyeDissipation: 0.995,
      velDissipation: 0.998,
      pressureIters: 18,
      vorticity: 1.0,
      splatRadius: 0.015,
      qualityScale: 0.75, // 0..1, sim resolution scalar
      maxDPR: 1.8,
    };
    Object.assign(this.settings, options);

    this._initGL();
    this._resize();
  }

  _initGL() {
    const gl = this.gl;
    this.vao = createFullscreenVAO(gl);
    this.programs = {
      advect: createProgram(gl, VS_SCREEN, FS_ADVECT),
      divergence: createProgram(gl, VS_SCREEN, FS_DIVERGENCE),
      pressure: createProgram(gl, VS_SCREEN, FS_PRESSURE),
      gradient: createProgram(gl, VS_SCREEN, FS_GRADIENT),
      curl: createProgram(gl, VS_SCREEN, FS_CURL),
      vorticity: createProgram(gl, VS_SCREEN, FS_VORTICITY),
      splat: createProgram(gl, VS_SCREEN, FS_SPLAT),
      render: createProgram(gl, VS_SCREEN, FS_RENDER),
    };

    // Uniform locations cached
    const U = (prog, name) => uniformLoc(gl, prog, name);
    this.u = {
      advect: {
        u_source: U(this.programs.advect, "u_source"),
        u_velocity: U(this.programs.advect, "u_velocity"),
        u_texelSize: U(this.programs.advect, "u_texelSize"),
        u_resolution: U(this.programs.advect, "u_resolution"),
        u_dt: U(this.programs.advect, "u_dt"),
        u_dissipation: U(this.programs.advect, "u_dissipation"),
      },
      divergence: {
        u_velocity: U(this.programs.divergence, "u_velocity"),
        u_texelSize: U(this.programs.divergence, "u_texelSize"),
      },
      pressure: {
        u_pressure: U(this.programs.pressure, "u_pressure"),
        u_divergence: U(this.programs.pressure, "u_divergence"),
        u_texelSize: U(this.programs.pressure, "u_texelSize"),
      },
      gradient: {
        u_velocity: U(this.programs.gradient, "u_velocity"),
        u_pressure: U(this.programs.gradient, "u_pressure"),
        u_texelSize: U(this.programs.gradient, "u_texelSize"),
      },
      curl: {
        u_velocity: U(this.programs.curl, "u_velocity"),
        u_texelSize: U(this.programs.curl, "u_texelSize"),
      },
      vorticity: {
        u_velocity: U(this.programs.vorticity, "u_velocity"),
        u_curl: U(this.programs.vorticity, "u_curl"),
        u_texelSize: U(this.programs.vorticity, "u_texelSize"),
        u_dt: U(this.programs.vorticity, "u_dt"),
        u_eps: U(this.programs.vorticity, "u_eps"),
      },
      splat: {
        u_target: U(this.programs.splat, "u_target"),
        u_point: U(this.programs.splat, "u_point"),
        u_value: U(this.programs.splat, "u_value"),
        u_radius: U(this.programs.splat, "u_radius"),
        u_texelSize: U(this.programs.splat, "u_texelSize"),
      },
      render: {
        u_dye: U(this.programs.render, "u_dye"),
        u_texelSize: U(this.programs.render, "u_texelSize"),
      },
    };

    // Geometry
    gl.bindVertexArray(this.vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.colorMask(true, true, true, true);
  }

  _allocate(width, height) {
    const gl = this.gl;
    const filter = gl.NEAREST; // manual bilerp in shader guarantees quality, avoids extension pitfalls
    const type = gl.HALF_FLOAT;
    // Choose sized internal formats
    const RG = gl.RG16F, RG_FMT = gl.RG, R = gl.R16F, R_FMT = gl.RED, RGBA = gl.RGBA16F, RGBA_FMT = gl.RGBA;

    // Velocity (RG), Pressure (R), Divergence (R), Curl (R), Dye (RGBA)
    this.velocity = makeDoubleFBO(gl, width, height, RG, RG_FMT, type, filter);
    this.pressure = makeDoubleFBO(gl, width, height, R, R_FMT, type, filter);
    this.divergence = makeFBO(gl, width, height, R, R_FMT, type, filter);
    this.curl = makeFBO(gl, width, height, R, R_FMT, type, filter);
    this.dye = makeDoubleFBO(gl, width, height, RGBA, RGBA_FMT, type, filter);
  }

  _dispose() {
    this.velocity?.dispose?.();
    this.pressure?.dispose?.();
    const gl = this.gl;
    if (this.divergence) { gl.deleteFramebuffer(this.divergence.fb); gl.deleteTexture(this.divergence.tex); }
    if (this.curl) { gl.deleteFramebuffer(this.curl.fb); gl.deleteTexture(this.curl.tex); }
    if (this.dye) this.dye.dispose();
  }

  _resize() {
    const gl = this.gl;
    const dpr = Math.min(window.devicePixelRatio || 1, this.settings.maxDPR);
    const scale = clamp(this.settings.qualityScale, 0.25, 1.0);
    const width = Math.max(16, Math.floor(this.canvas.clientWidth * dpr * scale));
    const height = Math.max(16, Math.floor(this.canvas.clientHeight * dpr * scale));
    if (gl.drawingBufferWidth === width && gl.drawingBufferHeight === height && this._w === width && this._h === height) {
      return;
    }
    this.canvas.width = Math.max(2, Math.floor(this.canvas.clientWidth * dpr));
    this.canvas.height = Math.max(2, Math.floor(this.canvas.clientHeight * dpr));
    this._dispose();
    this._allocate(width, height);
    this._w = width; this._h = height;
  }

  reset() {
    this._dispose();
    this._allocate(this._w, this._h);
  }

  _setCommonUniforms(program, texelW, texelH) {
    const gl = this.gl;
    gl.uniform2f(this.u[program].u_texelSize, 1 / texelW, 1 / texelH);
  }

  step(dt) {
    const gl = this.gl;
    const w = this.velocity.read.w, h = this.velocity.read.h;
    const texel = [1 / w, 1 / h];
    gl.bindVertexArray(this.vao);

    // Curl
    bindFBO(gl, this.curl);
    setViewport(gl, this.curl);
    gl.useProgram(this.programs.curl);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(this.u.curl.u_velocity, 0);
    gl.uniform2f(this.u.curl.u_texelSize, texel[0], texel[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Vorticity confinement
    if (this.settings.vorticity > 0.0) {
      bindFBO(gl, this.velocity.write);
      setViewport(gl, this.velocity.write);
      gl.useProgram(this.programs.vorticity);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
      gl.uniform1i(this.u.vorticity.u_velocity, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.curl.tex);
      gl.uniform1i(this.u.vorticity.u_curl, 1);
      gl.uniform2f(this.u.vorticity.u_texelSize, texel[0], texel[1]);
      gl.uniform1f(this.u.vorticity.u_dt, dt);
      gl.uniform1f(this.u.vorticity.u_eps, this.settings.vorticity);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.velocity.swap();
    }

    // Advect velocity
    bindFBO(gl, this.velocity.write);
    setViewport(gl, this.velocity.write);
    gl.useProgram(this.programs.advect);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(this.u.advect.u_source, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(this.u.advect.u_velocity, 1);
    gl.uniform2f(this.u.advect.u_texelSize, texel[0], texel[1]);
    gl.uniform2f(this.u.advect.u_resolution, w, h);
    gl.uniform1f(this.u.advect.u_dt, dt);
    gl.uniform1f(this.u.advect.u_dissipation, this.settings.velDissipation);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.velocity.swap();

    // Divergence
    bindFBO(gl, this.divergence);
    setViewport(gl, this.divergence);
    gl.useProgram(this.programs.divergence);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(this.u.divergence.u_velocity, 0);
    gl.uniform2f(this.u.divergence.u_texelSize, texel[0], texel[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pressure solve iterations
    gl.useProgram(this.programs.pressure);
    for (let i = 0; i < this.settings.pressureIters; i++) {
      bindFBO(gl, this.pressure.write);
      setViewport(gl, this.pressure.write);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.tex);
      gl.uniform1i(this.u.pressure.u_pressure, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.divergence.tex);
      gl.uniform1i(this.u.pressure.u_divergence, 1);
      gl.uniform2f(this.u.pressure.u_texelSize, texel[0], texel[1]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.pressure.swap();
    }

    // Subtract gradient
    bindFBO(gl, this.velocity.write);
    setViewport(gl, this.velocity.write);
    gl.useProgram(this.programs.gradient);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(this.u.gradient.u_velocity, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.tex);
    gl.uniform1i(this.u.gradient.u_pressure, 1);
    gl.uniform2f(this.u.gradient.u_texelSize, texel[0], texel[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.velocity.swap();

    // Advect dye
    bindFBO(gl, this.dye.write);
    setViewport(gl, this.dye.write);
    gl.useProgram(this.programs.advect);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.dye.read.tex);
    gl.uniform1i(this.u.advect.u_source, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.tex);
    gl.uniform1i(this.u.advect.u_velocity, 1);
    gl.uniform2f(this.u.advect.u_texelSize, texel[0], texel[1]);
    gl.uniform2f(this.u.advect.u_resolution, w, h);
    gl.uniform1f(this.u.advect.u_dt, dt);
    gl.uniform1f(this.u.advect.u_dissipation, this.settings.dyeDissipation);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.dye.swap();
  }

  renderToScreen() {
    const gl = this.gl;
    bindFBO(gl, null);
    setViewport(gl, null);
    gl.useProgram(this.programs.render);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.dye.read.tex);
    gl.uniform1i(this.u.render.u_dye, 0);
    gl.uniform2f(this.u.render.u_texelSize, 1 / this.dye.read.w, 1 / this.dye.read.h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  splat(point01, value3, radius01, target) {
    const gl = this.gl;
    const tgt = target === "velocity" ? this.velocity : this.dye;
    bindFBO(gl, tgt.write);
    setViewport(gl, tgt.write);
    gl.useProgram(this.programs.splat);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tgt.read.tex);
    gl.uniform1i(this.u.splat.u_target, 0);
    gl.uniform2f(this.u.splat.u_point, point01[0], point01[1]);
    gl.uniform3f(this.u.splat.u_value, value3[0], value3[1], value3[2] ?? 0.0);
    gl.uniform1f(this.u.splat.u_radius, radius01);
    gl.uniform2f(this.u.splat.u_texelSize, 1 / tgt.read.w, 1 / tgt.read.h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    tgt.swap();
  }
}
