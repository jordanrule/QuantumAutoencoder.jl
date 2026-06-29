# QuantumAutoencoder.jl

This is a Julia port of the QCompress Python framework, providing tools to learn low-dimensional classical descriptions of quantum compression. Rather than treating quantum and classical subsystems separately, the framework is organized around a unified principle: **capturing the effect of quantum dynamics as typed, extensible operations in the classical state**.

Ported from: https://github.com/hsim13372/QCompress  
License: Apache-2.0 (see `LICENSE`)

## Design Philosophy

Quantum autoencoders represent a family of variational circuits that learn to compress high-dimensional quantum information into a lower-dimensional latent space. The key insight is that the compression itself—the learned transformation—can be captured and represented as classical parameters that can be systematically analyzed, optimized, and extended.

This implementation embodies this principle through:

1. **Quantum-to-Classical Reduction**: The framework defines parametrized quantum circuits that compress quantum states and recover them. The quality of this compression is measured through quantum mechanical loss functions (measurement probabilities). These measurements are classical observables—effects of the quantum dynamics captured in the classical substrate.

2. **Typed Extensibility**: Rather than ad-hoc conditionals or implicit mode switching, the codebase uses Julia's multiple-dispatch to make the distinction between quantum and classical subsystems explicit and extensible through the type system. An abstract `AbstractQAEEngine` hierarchy (`QuantumEngine` for circuit modes, `ClassicalEngine` for fallback modes) allows new compression strategies to be added as subtypes without modification of core logic.

3. **Unified Interface**: Both quantum circuit compression and classical neural network compression expose the same `compress`/`decompress` API, allowing the framework to be used at different abstraction levels while maintaining consistency.

## Architecture and Implementation

The framework comprises four core modules under `src/QAE/`:

- **Config.jl**: Configuration container with sensible defaults for experiment specification.
- **Engine.jl**: Core engine implementing both quantum and classical modes through multiple-dispatch. The quantum mode (`QAEEngine <: QuantumEngine`) constructs and simulates parametrized circuits using Yao.jl, computes loss from measurement probabilities, and optimizes parameters with `Optim.jl`. The classical mode (`ClassicalQAEEngine <: ClassicalEngine`) provides a Flux.jl-based neural network autoencoder for compatibility and rapid prototyping.
- **Circuits.jl**: Pre-built circuit templates (Bell states, GHZ states, parameterized ansätze) for common use cases.
- **Utils.jl**: Utility functions including HDF5 dataset loading.

### Quantum Mode (Yao.jl-based)

The quantum mode implements the full quantum autoencoder loop:

1. **State Preparation**: User-provided quantum state-preparation circuits encode classical data into quantum states.
2. **Compression**: A parametrized training circuit learns to compress these states into a latent subspace (fewer qubits).
3. **Recovery**: An inverse circuit (dagger) recovers the original state from the compressed representation.
4. **Loss**: Measurement of the recovered state projects onto computational basis states. The loss is computed as the negative probability of the all-zero measurement outcome on the unmeasured (trash) qubits—higher fidelity recovery corresponds to higher probability of measuring zeros, lower loss.
5. **Optimization**: Classical optimizer (Nelder-Mead by default, via `Optim.jl`) searches parameter space to minimize loss.

### Classical Mode (Flux.jl-based)

For rapid prototyping and when quantum hardware is unavailable, a classical autoencoder implemented in Flux.jl is available. It exposes the same `compress`/`decompress` interface, allowing experiments to progress without a quantum backend.

## Type System and Multiple-Dispatch Design

Rather than embedding mode information as boolean flags or runtime type checks, the framework uses Julia's type system to make operational distinctions explicit:

```julia
abstract type AbstractQAEEngine end
abstract type QuantumEngine <: AbstractQAEEngine end
abstract type ClassicalEngine <: AbstractQAEEngine end

mutable struct QAEEngine <: QuantumEngine
    # Quantum-specific fields
end

mutable struct ClassicalQAEEngine <: ClassicalEngine
    # Classical-specific fields
end
```

Operations dispatch based on type:

```julia
compress(eng::QuantumEngine, params::Vector, index::Int) → quantum circuit compression
compress(eng::ClassicalQAEEngine, data::Array) → classical network compression
```

**Benefits**: (1) Compiler can specialize each code path, eliminating runtime overhead. (2) Type errors are caught at compile-time, not runtime. (3) New engine implementations can be added by subtyping `AbstractQAEEngine` without modifying existing code. (4) The constructor `QAEEngine(cfg::Config)` automatically dispatches to `ClassicalQAEEngine` for backward compatibility. Users can be explicit (`ClassicalQAEEngine(cfg)`) or implicit (`QAEEngine(cfg)`)—both work.

For comprehensive documentation of the type system refactoring, see `README_IMPROVEMENTS.md`.

## Usage

### Installation and Setup

To run tests or use the package, you will need Julia and the project dependencies:

```bash
cd /Users/jrule/git/quantum/QuantumAutoencoder.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

For the full quantum mode with Yao and classical optimization:

```bash
julia --project=. -e 'using Pkg; Pkg.add(["Yao","Optim","Flux","HDF5"])'
```

### Classical Mode (Fallback)

The classical mode provides a complete autoencoder via Flux for rapid experimentation:

```julia
using QuantumAutoencoder

cfg = QAE.Config(n_qubits=2, latent_dim=1, epochs=10, batch_size=8)
engine = QAE.QAEEngine(cfg)   # Returns ClassicalQAEEngine via dispatch
# Or explicitly: engine = QAE.ClassicalQAEEngine(cfg)

data = randn(100, cfg.n_qubits)
compressed = QAE.compress(engine, data)
reconstructed = QAE.decompress(engine, compressed)
```

### Quantum Mode

To use the quantum circuit mode, supply state-preparation circuits and a parametrized training circuit:

```julia
using QuantumAutoencoder

prep = QAE.bell_state_prep_circuit()                     # H(0); CNOT(0,1)
ghz = QAE.ghz_state_prep_circuit(3)                      # H(0); CNOT(0,1); CNOT(0,2)
ansatz = p -> QAE.simple_qae_training_circuit(p; n_qubits=2)

engine = QAE.QAEEngine(q_in, q_latent, [prep], ansatz)
params = randn(4)
circuit = QAE.compress(engine, params, 1)
```

The framework provides built-in circuit templates (Bell states, GHZ states, parameterized ansätze) as starting points. Users supply these to the engine constructor along with qubit register specifications (`q_in` and `q_latent`), and the framework handles circuit composition, simulation, loss evaluation, and optimization.

## Scope and Future Work

**Current implementation**: The framework assumes state-preparation and training circuits are expressed as Yao blocks. Automatic translation from arbitrary gate sets (e.g., pyQuil programs) is not yet implemented; however, a collection of common circuit templates is provided in `src/QAE/Circuits.jl`.

**Measurement and sampling**: The current loss computation uses exact statevector simulation to obtain measurement probabilities. For larger systems or to study noise effects, shot-based sampling can be added using Yao's sampling utilities or a measurement-supporting backend.

**Optimization methods**: Currently, a Nelder-Mead classical optimizer is the default. Future work includes gradient-based methods (parameter-shift rules, adjoint differentiation) and adaptive learning rates.

**Type system**: The refactoring to multiple-dispatch (June 2026) enables extensibility; new engine implementations can be added as subtypes of `AbstractQAEEngine` without modifying core logic.

## Recent Developments

**Type System Refactoring (June 2026)**: The codebase underwent a significant refactoring to make the quantum-classical distinction explicit and extensible through Julia's type system. An abstract type hierarchy (`AbstractQAEEngine` → `{QuantumEngine, ClassicalEngine}`) replaces runtime type checks. This improves compile-time specialization, type safety, and allows users to implement custom engines by subtyping. Backward compatibility is maintained through constructor dispatch: `QAEEngine(cfg::Config)` automatically returns a `ClassicalQAEEngine`. See `README_IMPROVEMENTS.md` for comprehensive documentation.

Attribution
-----------
This Julia package is a port and retains attribution to the original QCompress repository:
https://github.com/hsim13372/QCompress

License
-------
Apache-2.0 (see `LICENSE`)
