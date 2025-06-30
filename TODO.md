# OpenSpaceCharge.jl Implementation TODO

This TODO list tracks the implementation progress for OpenSpaceCharge.jl based on the design document.

## Phase 1: Project Setup and Core Data Structures

### 1.1 Initialize Julia Project
- [ ] Create `OpenSpaceCharge.jl` package structure
- [ ] Set up `Project.toml` with dependencies:
  - [ ] `KernelAbstractions`
  - [ ] `CUDA`
  - [ ] `AbstractFFTs`
  - [ ] `FFTW`
  - [ ] `Test`
- [ ] Create basic package directory structure:
  - [ ] `src/OpenSpaceCharge.jl` (main module)
  - [ ] `src/mesh.jl`
  - [ ] `src/deposition.jl`
  - [ ] `src/interpolation.jl`
  - [ ] `src/green_functions.jl`
  - [ ] `src/solvers/` directory
  - [ ] `src/utils.jl`
  - [ ] `test/runtests.jl`

### 1.2 Implement Mesh3D Struct
- [ ] Define `Mesh3D{T, A}` struct in `mesh.jl`
- [ ] Implement constructor functions for `Mesh3D`
- [ ] Add automatic `delta` calculation from grid dimensions and physical size
- [ ] Add basic validation for mesh parameters

## Phase 2: Particle and Grid Operations (CPU & GPU)

### 2.1 Implement Particle Deposition
- [ ] Create `deposition.jl` file
- [ ] Write `deposit_kernel!` using `KernelAbstractions.jl`
- [ ] Implement trilinear (Cloud-in-Cell) deposition logic
- [ ] Create main `deposit!` function that handles both CPU and GPU arrays
- [ ] Test deposition with simple particle distributions

### 2.2 Implement Field Interpolation
- [ ] Create `interpolation.jl` file
- [ ] Write `interpolate_kernel!` using `KernelAbstractions.jl`
- [ ] Implement trilinear interpolation for electric and magnetic fields
- [ ] Create main `interpolate_field` function for both CPU and GPU
- [ ] Test interpolation with known field distributions

## Phase 3: The Free-Space Solver

### 3.1 Implement Integrated Green's Functions
- [ ] Create `green_functions.jl` file
- [ ] Translate `lafun2` elemental function from Fortran to Julia
- [ ] Translate `xlafun2` elemental function from Fortran to Julia
- [ ] Ensure numerical stability (handle division by zero, log of non-positive numbers)
- [ ] Write `generate_igf_kernel!` using `KernelAbstractions.jl`
- [ ] Implement 8-point differencing for integrated values
- [ ] Test Green's function generation against analytical cases

### 3.2 Implement the Main Solver
- [ ] Create `solvers/free_space.jl` file
- [ ] Define `FreeSpace` boundary condition type
- [ ] Implement `solve!` method for `FreeSpace` boundary conditions
- [ ] Implement grid padding for convolution
- [ ] Add FFT planning using `AbstractFFTs`
- [ ] Implement convolution loop for field components:
  - [ ] Ex component calculation
  - [ ] Ey component calculation
  - [ ] Ez component calculation
- [ ] Implement B-field calculation from Ex and Ey
- [ ] Add proper grid extraction from padded results

### 3.3 Initial Testing (CPU)
- [ ] Create basic test in `test/runtests.jl`
- [ ] Generate simple particle distribution (Gaussian bunch)
- [ ] Test full pipeline: `deposit!` → `solve!` → `interpolate_field`
- [ ] Verify results are physically plausible
- [ ] Add unit tests for individual components

## Phase 4: GPU Acceleration and Cathode Solver

### 4.1 Enable GPU Execution
- [ ] Write GPU test cases using `CuArray`s
- [ ] Verify `deposit!` works on GPU without changes
- [ ] Verify `solve!` works on GPU without changes
- [ ] Verify `interpolate_field` works on GPU without changes
- [ ] Profile GPU performance
- [ ] Debug and fix any GPU-specific bottlenecks:
  - [ ] Eliminate scalar indexing
  - [ ] Optimize memory transfers
- [ ] Benchmark CPU vs GPU performance

### 4.2 Implement Cathode Image Solver
- [ ] Extend `solve!` in `free_space.jl` for `at_cathode=true` case
- [ ] Implement real charge distribution field calculation
- [ ] Create z-flipped, negated copy of charge density
- [ ] Calculate offset vector for image charge distribution
- [ ] Implement image charge field calculation
- [ ] Add real and image fields with proper signs
- [ ] Handle B-field sign correction for image charges
- [ ] Test cathode solver with known solutions

## Phase 5: Rectangular Pipe Solver

### 5.1 Implement Rectangular Pipe Solver
- [ ] Create `solvers/rect_pipe.jl` file
- [ ] Define `RectangularPipe` boundary condition type
- [ ] Translate `rfun` (pipe Green's function) from Fortran
- [ ] Implement series summation for pipe Green's function
- [ ] Write pipe Green's function as a kernel
- [ ] Implement four-term convolution-correlation logic:
  - [ ] FFT convolution terms
  - [ ] FFT correlation terms
  - [ ] Proper handling of `fft` vs `bfft`
- [ ] Test rectangular pipe solver on CPU
- [ ] Optimize and test on GPU
- [ ] Validate against analytical solutions where possible

## Phase 6: Finalization and Documentation

### 6.1 Comprehensive Testing
- [ ] Add tests for all boundary conditions:
  - [ ] Free space solver tests
  - [ ] Cathode image solver tests
  - [ ] Rectangular pipe solver tests
- [ ] Compare numerical outputs against Fortran code
- [ ] Add edge case and error handling tests
- [ ] Add performance regression tests
- [ ] Ensure 100% test coverage for core functionality

### 6.2 Documentation and Examples
- [ ] Write docstrings for all public functions and structs
- [ ] Create comprehensive `README.md`
- [ ] Create `examples/` directory with usage examples:
  - [ ] Basic CPU usage example
  - [ ] GPU acceleration example
  - [ ] Free space solver example
  - [ ] Cathode solver example
  - [ ] Rectangular pipe solver example
- [ ] Add inline code documentation
- [ ] Create performance comparison documentation

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
- [ ] Finalize package metadata in `Project.toml`
- [ ] Add license file
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