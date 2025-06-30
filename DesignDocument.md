## Design Document: OpenSpaceCharge.jl

### 1. High-Level Vision & Guiding Principles

The goal is to create a modern, high-performance, and user-friendly Julia package for 3D space-charge calculations that is functionally equivalent to the provided Fortran code. The package will be designed from the ground up to leverage Julia's strengths for parallel and GPU computing.

**Guiding Principles:**

1.  **Performance First:** The architecture will prioritize computational efficiency. We will use Julia's high-performance features, avoiding type instabilities and unnecessary allocations. The target is to match or exceed the performance of the original Fortran code.
2.  **CPU/GPU Agnostic Code:** The core computational logic (deposition, field calculation, interpolation) will be written using abstractions that allow the same code to run seamlessly on both multi-threaded CPUs and NVIDIA GPUs. This "write-once, run-anywhere" approach is a primary design goal.
3.  **Modularity and Clarity:** The package will be organized into logical modules. The code will be well-documented and written in a clear, "Julian" style, avoiding the legacy patterns (e.g., global module variables, cryptic flags) present in the Fortran code. The API will be intuitive, using keyword arguments and multiple dispatch.
4.  **Extensibility:** The design should make it straightforward to add new boundary conditions, particle deposition schemes, or field solvers in the future.
5.  **Correctness and Testability:** The package will be developed with a strong emphasis on testing. Each component will have unit tests, and the final results will be validated against known analytical solutions or benchmarked against the original Fortran code.

### 2. Core Technologies & Architecture

*   **Main Data Structure (`Grid` or `Mesh`):** A `mutable struct` will encapsulate all grid-related information, similar to `mesh3d_struct`. This will be the central object passed to most functions.
    ```julia
    mutable struct Mesh3D{T <: AbstractFloat, A <: AbstractArray{T}}
        # Grid indexing
        nlo::NTuple{3, Int}
        nhi::NTuple{3, Int}
        # Physical domain
        min_bounds::NTuple{3, T}
        max_bounds::NTuple{3, T}
        delta::NTuple{3, T}
        # Physics parameters
        gamma::T
        total_charge::T
        # Data arrays (CPU or GPU)
        rho::A
        phi::A
        efield::AbstractArray{T, 4} # Last dimension for component (x,y,z)
        bfield::AbstractArray{T, 4}
    end
    ```
*   **CPU/GPU Abstraction (`KernelAbstractions.jl`):** This is the cornerstone of our parallel strategy. We will write computational kernels (for deposition, interpolation, Green's function generation) using `KernelAbstractions.jl`. This allows us to write a single kernel that can be launched on a multi-threaded CPU backend or a CUDA GPU backend simply by changing the device object. This avoids code duplication entirely.
*   **FFT Abstraction (`AbstractFFTs.jl`):** We will use the `AbstractFFTs.jl` interface for all Fast Fourier Transforms. This allows the code to work transparently with `FFTW.jl` on the CPU and `CUDA.jl`'s `cuFFT` on the GPU. The correct FFT plan will be chosen at runtime based on the array type (`Array` or `CuArray`).
*   **Multiple Dispatch:** We will leverage Julia's multiple dispatch to handle different data types and boundary conditions elegantly. For example, the `solve!` function can have different methods for `solve!(mesh, ::FreeSpace)` and `solve!(mesh, ::RectangularPipe)`.
*   **Package Structure:**
    ```
    OpenSpaceCharge.jl/
    ├── Project.toml
    ├── Manifest.toml
    └── src/
        ├── OpenSpaceCharge.jl   # Main module, exports, includes
        ├── mesh.jl              # Mesh3D struct and constructors
        ├── deposition.jl        # Particle deposition kernels
        ├── interpolation.jl     # Field interpolation kernels
        ├── green_functions.jl   # IGF and other Green's function calculations
        ├── solvers/
        │   ├── free_space.jl    # Free-space and cathode image solver
        │   └── rect_pipe.jl     # Rectangular pipe solver
        └── utils.jl             # Helper functions
    └── test/
        └── runtests.jl
    ```

### 3. Step-by-Step Implementation Plan

This plan is broken down into phases, starting with core functionality on the CPU and progressively adding GPU support and more complex solvers.

---

#### **Phase 1: Project Setup and Core Data Structures**

1.  **Initialize Julia Project:** Create the `OpenSpaceCharge.jl` package structure. Define dependencies in `Project.toml`: `KernelAbstractions`, `CUDA`, `AbstractFFTs`, `FFTW`, `Test`.
2.  **Implement `Mesh3D` Struct:** Create the `mesh.jl` file. Define the `Mesh3D` struct as described above. Write convenient constructor functions, e.g., one that takes grid dimensions and physical size and calculates `delta` automatically.

---

#### **Phase 2: Particle and Grid Operations (CPU & GPU)**

3.  **Implement Particle Deposition (`deposit!`):**
    *   In `deposition.jl`, write a `deposit_kernel!` using `KernelAbstractions.jl`. This kernel will perform the trilinear (Cloud-in-Cell) deposition.
    *   The main `deposit!` function will take the mesh and particle coordinates (`x`, `y`, `z`, `q` arrays) and launch the kernel on the appropriate device (CPU or GPU, based on the array types).
    *   This function will replicate the logic in the Fortran `deposit_particles` subroutine.
4.  **Implement Field Interpolation (`interpolate_field`):**
    *   In `interpolation.jl`, write an `interpolate_kernel!` similar to the deposition kernel.
    *   The `interpolate_field` function will take the mesh and a set of coordinates and return the interpolated `E` and `B` fields at those points.

---

#### **Phase 3: The Free-Space Solver**

This phase replicates the modern `osc_freespace_solver2` logic.

5.  **Implement Integrated Green's Functions:**
    *   In `green_functions.jl`, translate the `lafun2` and `xlafun2` elemental functions from Fortran to Julia. Ensure they are numerically stable (e.g., handle divisions by zero, logs of non-positive numbers).
    *   Create a `KernelAbstractions.jl` kernel, `generate_igf_kernel!`, that computes the Green's function for potential (`phi`) or field components (`Ex`, `Ey`, `Ez`) on the padded grid. This kernel will call the translated `lafun2`/`xlafun2` and perform the 8-point differencing to get the integrated value for each cell, as seen in `osc_get_cgrn_freespace`.
6.  **Implement the Main Solver (`solve_freespace!`):**
    *   In `solvers/free_space.jl`, create the main `solve!` function dispatched for a `FreeSpace` boundary condition type.
    *   **Steps:**
        a.  Pad the `rho` array into a double-sized complex array for convolution.
        b.  Create an FFT plan using `plan_fft`.
        c.  Perform the forward FFT on the padded `rho`.
        d.  Loop through the required components (e.g., `Ex`, `Ey`, `Ez`).
        e.  Inside the loop:
            i.  Call the `generate_igf_kernel!` to get the corresponding Green's function on a padded grid.
            ii. Perform the forward FFT on the Green's function.
            iii.Multiply the FFT of `rho` and the FFT of the Green's function element-wise.
            iv. Perform the inverse FFT on the result.
        f.  Extract the real part of the result from the correct sub-region of the padded grid and store it in `mesh.efield`.
        g.  After computing the E-field, calculate the B-field from `Ex` and `Ey`.
7.  **Initial Testing (CPU):** Create a test script in `test/runtests.jl`. Generate a simple particle distribution (e.g., Gaussian bunch), run the full `deposit!` -> `solve!` -> `interpolate_field` pipeline on the CPU, and verify the results are physically plausible.

---

#### **Phase 4: GPU Acceleration and Cathode Solver**

8.  **Enable GPU Execution:**
    *   Write test cases where the initial `Mesh3D` and particle arrays are created as `CuArray`s using `CUDA.jl`.
    *   Thanks to the abstractions (`KernelAbstractions`, `AbstractFFTs`), the existing `deposit!`, `solve!`, and `interpolate_field` functions should work on the GPU with minimal to no changes.
    *   Profile and debug performance. Address any GPU-specific bottlenecks (e.g., scalar indexing, memory transfers).
9.  **Implement Cathode Image Solver:**
    *   Extend the `solve!` function in `solvers/free_space.jl` to handle the `at_cathode=true` case.
    *   Follow the clean logic from the Fortran `space_charge_3d` routine:
        a.  Calculate the free-space field of the real charge distribution.
        b.  Create a temporary, z-flipped, negated copy of `rho`.
        c.  Calculate the offset vector for the image charge distribution.
        d.  Call the free-space solver *again* for the image charge distribution with the calculated offset.
        e.  Add the real and image fields together to get the final result. Be careful with the signs for the B-field calculation, as the image charges move in the opposite direction.

---

#### **Phase 5: Rectangular Pipe Solver**

10. **Implement the Rectangular Pipe Solver:** This is the most complex part and will be a separate solver.
    *   In `solvers/rect_pipe.jl`, create a `solve!` method dispatched on a `RectangularPipe` boundary type.
    *   Translate the pipe Green's function (`rfun`) from Fortran, which involves a series summation. This can be implemented as a kernel.
    *   Implement the four-term convolution-correlation logic from `fftconvcorr3d`. This involves performing FFTs and IFFTs with different signs in the transform direction to achieve both convolution (`FFT(A) * FFT(B)`) and correlation (`FFT(A) * conj(FFT(B))`). The `AbstractFFTs` interface handles this (`fft` vs. `bfft`).
    *   This solver will likely be more complex to get running on the GPU and will require careful implementation.

---

#### **Phase 6: Finalization and Documentation**

11. **Comprehensive Testing:** Add tests for all boundary conditions. If possible, compare numerical outputs against the Fortran code for a given input to ensure correctness.
12. **Documentation and Examples:** Write clear docstrings for all public functions and structs using Julia's standard documentation system. Create a `README.md` and example scripts in an `examples/` directory showing how to use the package on both CPU and GPU.
13. **API Refinements:** Clean up the user-facing API. Ensure keyword arguments are used for all optional flags (`direct_field_calc`, `integrated_green_function`, etc.) to make the function calls clear and readable.
14. **Package Registration:** Prepare the package for registration in the official Julia General registry.