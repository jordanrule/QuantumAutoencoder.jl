"""
QuantumAutoencoder.jl

This module is a lightweight Julia port of parts of the QCompress project:
https://github.com/hsim13372/QCompress

Licensed under Apache-2.0 (see project LICENSE).
"""

module QuantumAutoencoder

export QAEEngine, Config, load_dataset, compress, decompress

using LinearAlgebra

include("QAE/Config.jl")
include("QAE/Utils.jl")
include("QAE/Engine.jl")

using .QAE: Config, QAEEngine, load_dataset, compress, decompress

end # module

