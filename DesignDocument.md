# SpaceCharge.jl Design Document

## 1. Overview & Philosophy

SpaceCharge.jl is a modern, high-performance Julia package for 3D space-charge calculations, designed for scientific computing applications such as beam physics and accelerator modeling. The codebase is architected for clarity, modularity, extensibility, and performance, with seamless support for both CPU and GPU execution.

### Key Principles
- **Performance & Correctness:** Prioritize computational efficiency and numerical accuracy. Avoid type instabilities and unnecessary allocations.
- **CPU/GPU Agnosticism:** Core computational logic is written to run on both CPUs and GPUs, using `KernelAbstractions.jl` and array type dispatch.
- **Modularity:** The codebase is organized into logical modules (mesh, deposition, solvers, interpolation, etc.), each with clear responsibilities and interfaces.
- **Extensibility:** Adding new solvers, boundary conditions, or deposition schemes is straightforward via Julia's multiple dispatch and abstract types.
- **Testability:** Comprehensive unit and integration tests ensure correctness and facilitate safe refactoring.
- **Julian API:** The API is idiomatic, using keyword arguments, multiple dispatch, and clear docstrings.

---

## 2. Codebase Structure

```
SpaceCharge.jl/
├── src/
│   ├── SpaceCharge.jl         # Main module, exports, includes
│   ├── mesh.jl                # Mesh3D struct and constructors
│   ├── deposition.jl          # Particle deposition kernels
│   ├── interpolation.jl       # Field interpolation kernels
│   ├── green_functions.jl     # Green's function calculations
│   ├── solvers/
│   │   ├── free_space.jl      # Free-space and cathode image solver
│   │   └── rectangular_pipe.jl# Rectangular pipe solver
│   ├── utils.jl               # Physical constants, helpers
│   └── visualization.jl       # (Optional) Visualization utilities
├── test/                      # Unit and integration tests
├── examples/                  # Usage and comparison scripts
├── benchmark/                 # Performance benchmarks
├── README.md, TODO.md, etc.
```

---

## 3. Main Algorithmic Flow

### 3.1 Particle Deposition (deposit!)
- **Purpose:** Deposit the charge of a set of particles onto a 3D grid (mesh) using the Particle-in-Cell (PIC) method.
- **Implementation:**
  - The `deposit_kernel!` is written using `KernelAbstractions.jl` and can be launched on either CPU or GPU.
  - Each particle's charge is distributed to the 8 nearest grid points based on its position within the cell, using trilinear weights.
  - Atomic operations are used on the GPU for thread safety; vectorized loops are used on the CPU for performance.
  - The mesh is sized to contain all particles, so no bounds checking is needed in the kernel.
  - The main entry point is `deposit!(mesh, x, y, z, q)`, which dispatches to the appropriate backend.

### 3.2 Field Solve (solve!)
- **Purpose:** Compute the electric and magnetic fields on the grid from the deposited charge density, using FFT-based convolution with integrated Green's functions (IGF).
- **Implementation:**
  - The main solver is `solve!(mesh; kwargs...)`.
  - Pads the charge density array to double size for convolution.
  - Computes the integrated Green's function (potential and field components) using a kernel (`osc_get_cgrn_freespace!`).
  - Both charge and Green's function arrays are FFT'd (using `AbstractFFTs.jl` for CPU/GPU abstraction).
  - The convolution is performed in Fourier space (element-wise multiplication), then inverse FFT'd.
  - The result is extracted from the padded array and stored in `mesh.efield` and `mesh.phi`.
  - The B-field is computed from the E-field using relativistic formulas.
  - If `at_cathode=true`, an image charge is created by flipping and negating the charge density, and the solver is run again with an offset; the fields are superposed with correct sign handling.
 

### 3.3 Field Interpolation (interpolate_field)
- **Purpose:** Interpolate the computed electric and magnetic fields from the grid to arbitrary particle positions.
- **Implementation:**
  - The `interpolate_kernel!` is written using `KernelAbstractions.jl` for CPU/GPU support.
  - For each particle, the field is interpolated from the 8 nearest grid points using trilinear weights.
  - The main entry point is `interpolate_field(mesh, x, y, z)`, which returns arrays of field values at the given positions.

---

## 4. High-Level Design Choices

### 4.1 Data Structures
- **Mesh3D:** Central mutable struct holding grid geometry, physical parameters, and all field arrays. Supports both CPU (`Array`) and GPU (`CuArray`) storage via type parameters and `Adapt.jl`.
- **BoundaryCondition (Abstract Type):** All solvers dispatch on this, enabling extensibility for new boundary types.

### 4.2 Modularity & Extensibility
- Each major algorithmic component (deposition, interpolation, solvers) is in its own file/module.
- Green's function logic is abstracted for reuse and extension.

### 4.3 CPU/GPU Abstraction
- **KernelAbstractions.jl:** All performance-critical kernels (deposition, interpolation, Green's function generation) are written using this package, allowing a single code path for both CPU and GPU.
- **Array Types:** All data arrays in `Mesh3D` can be either `Array` or `CuArray`, with device selection handled by array type and `Adapt.jl`.
- **FFT Abstraction:** Uses `AbstractFFTs.jl` for device-agnostic FFTs (`FFTW.jl` for CPU, `CUDA.jl` for GPU).

### 4.4 API & User-Facing Abstractions
- **Main API Functions:**
  - `Mesh3D(...)` – Flexible constructors for mesh setup (auto or manual bounds).
  - `deposit!`, `clear_mesh!` – Particle-to-grid deposition and mesh reset.
  - `solve!(mesh; kwargs...)` – Field solver for free space.
  - `interpolate_field(mesh, x, y, z)` – Field interpolation at arbitrary points.
- **Keyword Arguments:** Used for all optional solver flags (e.g., `at_cathode`).

### 4.5 Testing & Validation
- **Test Organization:** Each major module has a corresponding test file (e.g., `test_deposition.jl`, `test_solvers.jl`).
- **Test Coverage:** Includes unit tests, integration tests, and analytical/benchmark comparisons.
- **GPU Testing:** Dedicated tests for GPU code paths (`test_gpu.jl`).
- **Continuous Integration:** (Planned) CI/CD and multi-version Julia testing.

---

## 5. Developer Guidelines

- **Follow Julian Style:** Use clear, type-stable code. Prefer multiple dispatch over conditionals. Avoid global state.
- **Documentation:** All public functions and types must have docstrings. Update `README.md` and `examples/` as needed.
- **Testing:** Add or update tests for all new features or bugfixes. Run the full test suite before submitting PRs.
- **Extending the Codebase:**
  - To add a new solver: create a new subtype of `BoundaryCondition` and implement `solve!` for it.
  - To add a new deposition/interpolation scheme: add a new kernel and dispatching function.
  - To add new physical models: extend the relevant modules and update tests/examples.
- **Performance:** Profile new code, especially for GPU. Avoid unnecessary allocations and scalar indexing.

---

## 6. Future Directions & TODOs

- Complete GPU support for all kernels and solvers.
- Add more boundary conditions and solver types.
- Improve parameter validation and error messages.
- Expand documentation and usage examples.
- Integrate with visualization and analysis tools.
- Support for distributed and multi-GPU computing.

---

## 7. References
- [Original Fortran OpenSpaceCharge](https://github.com/RobertRyne/OpenSpaceCharge)
- [Mayes, C., Ryne, R., Sagan, D. "3D Space Charge in Bmad." IPAC 2018](https://doi.org/10.18429/JACoW-IPAC2018-THPAK085)

---

*This document is maintained for developers. Please update it as the codebase evolves.*
