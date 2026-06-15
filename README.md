# QuantumAutoencoder.jl

This repository contains a Julia port of the QCompress Python framework (core quantum autoencoder functionality).

Ported from: https://github.com/hsim13372/QCompress

License: Apache-2.0 (see `LICENSE`)

Status
------
This is an initial port with a minimal, idiomatic Julia structure. Core modules added:

- `src/QAE/Config.jl` — configuration helpers
- `src/QAE/Utils.jl` — utility functions (HDF5 loading, simple helpers)
- `src/QAE/Engine.jl` — minimal QAE engine scaffold (QAEEngine struct with compress/decompress placeholders)

Usage
-----
Start Julia and run tests:

```bash
cd QuantumAutoencoder.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Notes
-----
This project is a port and retains attribution to the original QCompress repository linked above. The implementation is intended as a starting point — replace placeholder methods in `Engine.jl` with full algorithms as needed.

