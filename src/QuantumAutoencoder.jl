"""
QuantumAutoencoder.jl

This module is a lightweight Julia port of parts of the QCompress project:
https://github.com/hsim13372/QCompress

Licensed under Apache-2.0 (see project LICENSE).
"""

module QuantumAutoencoder

export QAEEngine, Config, load_dataset, compress, decompress,
	   bell_state_prep_circuit, ghz_state_prep_circuit, simple_qae_training_circuit

using LinearAlgebra

module QAE
include("QAE/Config.jl")
include("QAE/Utils.jl")
include("QAE/Circuits.jl")
include("QAE/Engine.jl")
end # module QAE

using .QAE: Config, QAEEngine, load_dataset, compress, decompress,
			bell_state_prep_circuit, ghz_state_prep_circuit, simple_qae_training_circuit

end # module

