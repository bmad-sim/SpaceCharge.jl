# SpaceCharge.jl Implementation TODO

This TODO list tracks the implementation progress for SpaceCharge.jl based on the design document.

## Phase 1: Project Setup and Core Data Structures

### 1.1 Initialize Julia Project
- [x] Create `SpaceCharge.jl` package structure
- [x] Set up `Project.toml` with dependencies:
  - [x] `KernelAbstractions`
  - [x] `CUDA`
  - [x] `AbstractFFTs`
  - [x] `FFTW`
  - [x] `Test`
- [x] Create basic package directory structure:
  - [x] `src/SpaceCharge.jl` (main module)
  - [x] `src/mesh.jl`
  - [x] `src/deposition.jl`
  - [x] `src/interpolation.jl`
  - [x] `src/green_functions.jl`
  - [x] `src/solvers/` directory
  - [x] `src/utils.jl`
  - [x] `test/runtests.jl`

### 1.2 Implement Mesh3D Struct
- [x] Define `Mesh3D{T, A}` struct in `mesh.jl`
- [x] Implement constructor functions for `Mesh3D`
- [x] Add automatic `delta` calculation from grid dimensions and physical size
- [x] Add basic validation for mesh parameters

## Phase 2: Particle and Grid Operations (CPU & GPU)

### 2.1 Implement Particle Deposition
- [x] Create `deposition.jl` file
- [x] Write `deposit_kernel!` using `KernelAbstractions.jl`
- [x] Implement trilinear (Cloud-in-Cell) deposition logic
- [x] Create main `deposit!` function that handles both CPU and GPU arrays
- [x] Test deposition with simple particle distributions

### 2.2 Implement Field Interpolation
- [x] Create `interpolation.jl` file
- [x] Write `interpolate_kernel!` using `KernelAbstractions.jl`
- [x] Implement trilinear interpolation for electric and magnetic fields
- [x] Create main `interpolate_field` function for both CPU and GPU
- [x] Test interpolation with known field distributions

## Phase 3: The Free-Space Solver

### 3.1 Implement Integrated Green's Functions
- [x] Create `green_functions.jl` file
- [x] Translate `lafun2` elemental function from Fortran to Julia
- [x] Translate `xlafun2` elemental function from Fortran to Julia
- [x] Translate `ylafun` elemental function from Fortran to Julia
- [x] Translate `zlafun` elemental function from Fortran to Julia
- [ ] Ensure numerical stability (handle division by zero, log of non-positive numbers)
- [x] Write `generate_igf_kernel!` using `KernelAbstractions.jl`
- [x] Implement 8-point differencing for integrated values
- [x] Test Green's function generation against analytical cases

### 3.2 Implement the Main Solver
- [x] Create `solvers/free_space.jl` file
- [x] Define `FreeSpace` boundary condition type
- [x] Implement `solve!` method for `FreeSpace` boundary conditions
- [x] Implement grid padding for convolution
- [x] Add FFT planning using `AbstractFFTs`
- [x] Implement convolution loop for field components:
  - [x] Ex component calculation
  - [x] Ey component calculation
  - [x] Ez component calculation
- [x] Implement B-field calculation from Ex and Ey
- [x] Add proper grid extraction from padded results

### 3.3 Initial Testing (CPU)
- [x] Create basic test in `test/runtests.jl`
- [x] Generate simple particle distribution (Gaussian bunch)
- [x] Test full pipeline: `deposit!` → `solve!` → `interpolate_field`
- [x] Verify results are physically plausible
- [x] Add unit tests for individual components

## Phase 4: GPU Acceleration and Cathode Solver

### 4.1 Enable GPU Execution
- [x] Write GPU test cases using `CuArray`s
- [x] Verify `deposit!` works on GPU without changes
- [x] Verify `solve!` works on GPU without changes
- [ ] Verify `interpolate_field` works on GPU without changes
- [ ] Profile GPU performance
- [ ] Debug and fix any GPU-specific bottlenecks:
  - [ ] Eliminate scalar indexing
  - [ ] Optimize memory transfers
- [x] Benchmark CPU vs GPU performance

### 4.2 Implement Cathode Image Solver
- [x] Extend `solve!` in `free_space.jl` for `at_cathode=true` case
- [x] Implement real charge distribution field calculation
- [x] Create z-flipped, negated copy of charge density
- [x] Calculate offset vector for image charge distribution
- [x] Implement image charge field calculation
- [x] Add real and image fields with proper signs
- [x] Handle B-field sign correction for image charges
- [ ] Test cathode solver with known solutions

## Phase 5: Rectangular Pipe Solver

### 5.1 Implement Rectangular Pipe Solver
- [x] Create `solvers/rect_pipe.jl` file
- [x] Define `RectangularPipe` boundary condition type
- [x] Translate `rfun` (pipe Green's function) from Fortran
- [x] Implement series summation for pipe Green's function
- [x] Write pipe Green's function as a kernel
- [x] Implement four-term convolution-correlation logic:
  - [x] FFT convolution terms
  - [x] FFT correlation terms
  - [x] Proper handling of `fft` vs `bfft`
- [ ] Test rectangular pipe solver on CPU
- [ ] Optimize and test on GPU (pending full `rfun` implementation)
- [ ] Validate against analytical solutions where possible (pending full `rfun` implementation)

## Phase 6: Finalization and Documentation

### 6.1 Comprehensive Testing
- [ ] Add tests for all boundary conditions:
  - [x] Free space solver tests
  - [ ] Cathode image solver tests
  - [ ] Rectangular pipe solver tests
- [ ] Compare numerical outputs against Fortran code
- [ ] Add edge case and error handling tests
- [ ] Add performance regression tests
- [ ] Ensure 100% test coverage for core functionality

### 6.2 Documentation and Examples
- [x] Write docstrings for all public functions and structs
- [x] Create comprehensive `README.md`
- [x] Create `examples/` directory with usage examples:
  - [x] Basic CPU usage example
  - [x] GPU acceleration example
  - [x] Free space solver example
  - [x] Cathode solver example
  - [x] Rectangular pipe solver example
- [x] Add inline code documentation
- [x] Create performance comparison documentation

### 6.3 API Refinements
- [ ] Review and clean up user-facing API
- [ ] Ensure keyword arguments for all optional flags:
  - [ ] `direct_field_calc`
  - [ ] `integrated_green_function`
  - [ ] `at_cathode`
- [ ] Add parameter validation and helpful error messages
- [ ] Implement sensible defaults for all parameters
- [ ] Add convenience constructors and helper functions

### 6.4 Package Registration
- [x] Finalize package metadata in `Project.toml`
- [x] Add license file
- [ ] Create GitHub repository with CI/CD
- [ ] Set up automatic testing on multiple Julia versions
- [ ] Prepare package for Julia General registry registration
- [ ] Create release notes and versioning strategy

## Additional Tasks (As Needed)

- [ ] Performance optimization passes
- [ ] Memory usage optimization
- [ ] Support for additional boundary conditions
- [ ] Integration with visualization packages
- [ ] Support for different particle deposition schemes
- [ ] Multi-GPU support
- [ ] Distributed computing support 