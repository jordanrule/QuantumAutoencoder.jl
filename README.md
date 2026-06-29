*This project is an attempt to codify a latent space I have had in my head after experimenting with IBM quantum a few years back.  I am not qualified to lead the charge, but I'm here as an advocate.*

# QuantumAutoencoder.jl

This repository contains a Julia port of the QCompress Python framework (core quantum autoencoder functionality).

Ported from: https://github.com/hsim13372/QCompress

License: Apache-2.0 (see `LICENSE`)

Overview of this port
---------------------
What I implemented in this repository:

- A minimal, idiomatic Julia package skeleton at `QuantumAutoencoder.jl`.
- Core modules ported and restructured under the `QAE` submodule:
  - `src/QAE/Config.jl` — configuration container (`Config` struct with sensible defaults).
  - `src/QAE/Utils.jl` — utilities (HDF5 dataset loader; requires `HDF5.jl` to actually read files).
  - `src/QAE/Engine.jl` — the main engine. Two modes are supported:
	1. Circuit (quantum) mode using `Yao.jl`: a port of the circuit-level API in the original `qae_engine.py` (construct compression/recovery circuits, simulate, compute cost as the negative probability of measuring the all-zero bitstring, and run a classical optimizer over parameters).
	2. Classical (fallback) mode using `Flux.jl`: a small autoencoder implemented with Flux to preserve the original testing / usage patterns while a full circuit translation is developed.

Design decisions preserved from the original QCompress
----------------------------------------------------
- High-level API and workflow: the Julia `QAEEngine` exposes the same conceptual methods as the original (constructing compression and recovery circuits, splitting train/test, computing the loss for data points, training and predicting).
- Cost function intent: the original cost is defined from measurement outcomes (frequency/probability of measuring all zeros on certain qubits). The Yao-based implementation preserves this: it computes the probability of the all-zero outcome on the measured qubits and uses the negative of that probability as the loss (matching the author's intent).
- Classical outer-loop: the original uses classical optimizers (scipy.optimize); the port uses `Optim.jl`'s Nelder-Mead by default for similar behavior. A Flux-based training loop is also provided as a compatibility/classical alternative.

What this framework accomplishes (plain language)
-----------------------------------------------
Imagine you have many quantum states that encode data. A quantum autoencoder tries to learn a smaller quantum system (fewer qubits) that retains the important information — like compressing images so they take less space while keeping the important features. This port gives you:

- The plumbing to define how data states are prepared (state-prep circuits).
- A way to define a parameterized compression circuit (training circuit) and its inverse (recovery).
- Tools to evaluate how well the compressed data can be recovered (a loss computed from measurement results).
- A classical optimizer that adjusts circuit parameters to minimize that loss, effectively "training" the quantum autoencoder.

How the architecture fits together (Yao.jl + Flux.jl)
--------------------------------------------------
- Yao.jl (circuit / quantum mode):
  - Users provide state-preparation blocks and a parameterized training circuit (functions or Yao blocks). The engine composes the blocks (state prep -> training circuit -> optional recovery) and simulates them using Yao's statevector simulator.
  - The loss is computed by summing the probability amplitudes where the measured qubits are all 0 (so higher probability of all-zeros is better; port uses the negative probability as loss to minimize).
  - A classical optimizer (Optim.jl) searches parameter space to find circuit parameters that minimize the loss.

- Flux.jl (classical fallback):
  - For quick tests and compatibility, a small MLP autoencoder implemented with Flux is available via `QAEEngine(cfg::Config)` (classical mode). This lets tests and examples run without quantum backends while keeping the same `compress`/`decompress` API shape for simple experiments.

Type System & Multiple-Dispatch Design
----------------------------------------
The codebase uses Julia's **multiple-dispatch** paradigm to cleanly separate quantum and classical modes:

- **Type Hierarchy**: An abstract `AbstractQAEEngine` defines the contract, with `QuantumEngine` and `ClassicalEngine` as subtypes. Concrete implementations are `QAEEngine` (quantum circuits) and `ClassicalQAEEngine` (neural networks).
- **Dispatch-Based Methods**: Rather than runtime type checks (`if engine.is_classical`), the framework uses Julia's type-based dispatch. For example:
  - `compress(eng::QuantumEngine, params::Vector, index::Int)` — quantum circuit compression
  - `compress(eng::ClassicalQAEEngine, data::Array)` — classical network compression
  - The compiler selects the correct method based on the engine type passed in.
- **Backward Compatibility**: The constructor `QAEEngine(cfg::Config)` automatically dispatches to `ClassicalQAEEngine`, so existing code continues to work unchanged.
- **Extensibility**: Users can create custom engine types by subtyping `AbstractQAEEngine` and implementing the interface methods. The dispatch system automatically routes calls to the correct implementation.

This approach eliminates runtime overhead, improves type safety (errors caught at compile-time), and makes the code more maintainable. For more details, see `README_IMPROVEMENTS.md` in the repository root.

Quick usage notes
-----------------
- To run tests or use the package you will need Julia and the project dependencies. From the package root:

```bash
cd /Users/jrule/git/quantum/QuantumAutoencoder.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

- If you plan to use the circuit mode add Yao and Optim (and HDF5 if you need datasets):

```bash
julia --project=. -e 'using Pkg; Pkg.add(["Yao","Optim","Flux","HDF5"])'
```

- Example (classical fallback):

```julia
using QuantumAutoencoder
cfg = QAE.Config(n_qubits=2, latent_dim=1, epochs=10, batch_size=8)
engine = QAE.QAEEngine(cfg)   # classical Flux-based engine (via dispatch)
# Or be explicit: engine = QAE.ClassicalQAEEngine(cfg)
data = randn(100, cfg.n_qubits)
compressed = QAE.compress(engine, data)
reconstructed = QAE.decompress(engine, compressed)
```

- Example (quantum / Yao mode): supply `state_prep_circuits` (vector of Yao blocks or callables returning blocks) and a `training_circuit(params)` function that returns a parameterized Yao block. Use `QAE.QAEEngine(q_in, q_latent, state_prep_circuits, training_circuit, ...)` to construct the engine, then call `train` with an initial parameter vector.

- Built-in converted Yao examples (ported equivalents of common pyQuil patterns):

```julia
using QuantumAutoencoder

prep = QAE.bell_state_prep_circuit()                     # H(0); CNOT(0,1)
ghz = QAE.ghz_state_prep_circuit(3)                      # H(0); CNOT(0,1); CNOT(0,2)
ansatz = p -> QAE.simple_qae_training_circuit(p; n_qubits=2)
```

Limitations and next steps
--------------------------
- This is an initial port: the Yao-based engine assumes state-prep and training circuits are expressed as Yao blocks (no automatic translation from arbitrary pyQuil programs yet). A few common circuit patterns are now included in `src/QAE/Circuits.jl` as converted examples.
- `_execute_circuit` currently uses exact statevector simulation to compute probabilities. For larger systems or to mimic real-device sampling you may want shot-based sampling; this can be added using Yao's sampling utilities or a backend that supports measurements.
- Optimizer choices and gradient-based methods (parameter-shift rules, adjoint methods) are not yet implemented — only a Nelder-Mead style classical optimizer is provided by default.

Recent improvements
-------------------
- **Type System Refactoring (June 2026)**: The codebase now uses Julia's multiple-dispatch paradigm throughout. The type system has been restructured with an abstract type hierarchy (`AbstractQAEEngine`, `QuantumEngine`, `ClassicalEngine`) and concrete implementations (`QAEEngine`, `ClassicalQAEEngine`). This improves performance (compile-time dispatch vs. runtime checks), type safety, and extensibility. Backward compatibility is maintained. See `README_IMPROVEMENTS.md` for comprehensive documentation of the refactoring.

Attribution
-----------
This Julia package is a port and retains attribution to the original QCompress repository:
https://github.com/hsim13372/QCompress

License
-------
Apache-2.0 (see `LICENSE`)
