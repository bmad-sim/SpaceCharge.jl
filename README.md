# SpaceCharge.jl

SpaceCharge.jl is a high-performance Julia package for 3D space-charge calculations, supporting both CPU and GPU (CUDA) acceleration. It is designed for beam physics and accelerator modeling applications.

## Algorithm and Citation

This package implements a parallel fast Fourier transform (FFT) based 3D space charge algorithm using integrated Green functions, as described in:

> Mayes, Christopher and Ryne, Robert and Sagan, David. "3D Space Charge in Bmad." 9th International Particle Accelerator Conference, 2018. [doi:10.18429/JACoW-IPAC2018-THPAK085](https://doi.org/10.18429/JACoW-IPAC2018-THPAK085)

The original Fortran implementation is available at: [https://github.com/RobertRyne/OpenSpaceCharge](https://github.com/RobertRyne/OpenSpaceCharge)

The package provides high-level routines to:
- Deposit weighted charged particles on a 3D rectangular grid.
- Calculate the space charge fields on this grid.
- Interpolate the field to an arbitrary point within its domain.

Convolutions of the Green functions and the charge density are performed efficiently with FFTs.

## Installation

To install SpaceCharge.jl, open the Julia REPL and run:

```julia
using Pkg
Pkg.add(url="https://github.com/ndwang/SpaceCharge.jl.git")
```

## Getting Started

- **Examples:** See `examples/` for scripts demonstrating basic usage, GPU acceleration, cathode boundary conditions, and analytical comparisons.
- **Benchmarks:** The `benchmark/` directory contains scripts to evaluate performance on different hardware and problem sizes.
- **Tests:** The `test/` directory provides a comprehensive test suite including an analytical comparison against an isotropic Gaussian bunch.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
