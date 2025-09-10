# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a WebGL2-based fluid dynamics simulation that renders interactive smoke effects in real-time. The application demonstrates computational fluid dynamics using the incompressible Navier-Stokes equations implemented entirely in fragment shaders.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

- **`index.html`**: Entry point with canvas element and UI overlay
- **`styles.css`**: Modern CSS with CSS custom properties, glassmorphism UI effects, and responsive design
- **`src/app.js`**: Main application controller handling initialization, animation loop, user interaction, and performance management
- **`src/sim.js`**: Core fluid simulation engine implementing the full CFD pipeline with WebGL2 shaders
- **`src/gl-helpers.js`**: Low-level WebGL2 utilities for safe context creation, program compilation, and framebuffer management

## Core Fluid Simulation Pipeline

The simulation implements a standard CFD solver with these stages (all GPU-accelerated):
1. **Vorticity confinement** - Maintains fluid swirls and turbulence
2. **Velocity advection** - Self-advects the velocity field
3. **Divergence computation** - Calculates velocity field divergence
4. **Pressure projection** - Iterative Jacobi solver for pressure field
5. **Gradient subtraction** - Ensures incompressible flow
6. **Dye advection** - Advects visual dye particles through the velocity field

## Key Technical Features

- **Double framebuffers**: Ping-pong rendering for iterative algorithms
- **Manual bilinear filtering**: Custom shader interpolation for consistent quality across hardware
- **Dynamic quality scaling**: Automatic resolution adjustment based on frame timing
- **Multi-touch support**: Simultaneous pointer interaction with velocity injection
- **WebGL2 float textures**: Required for numerical precision in fluid calculations

## Development

This is a client-side only application with no build process. To run:

```bash
# Serve the directory with any HTTP server
python -m http.server 8000
# or
npx serve .
```

Open in browser at `http://localhost:8000`

## Interaction Controls

- **Click/drag**: Inject velocity and colored dye
- **P**: Pause/unpause simulation
- **Q**: Cycle quality settings (0.5x, 0.75x, 1.0x resolution)
- **V**: Toggle vorticity confinement on/off
- **R**: Reset simulation state

## Performance Considerations

- Simulation resolution scales with `qualityScale` setting (0.25-1.0)
- Automatic quality adjustment targets 60fps by monitoring frame times
- Uses `HALF_FLOAT` textures for memory/bandwidth optimization
- Pressure solver iterations configurable via `pressureIters` setting

## WebGL Requirements

- WebGL2 context required
- `EXT_color_buffer_float` extension required for rendering to float textures
- Graceful fallback with error messages for unsupported devices