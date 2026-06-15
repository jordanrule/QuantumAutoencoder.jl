module QAE

"""
Configuration container for the Quantum Autoencoder port.

This file is a Julia port inspired by `config.py` in QCompress:
https://github.com/hsim13372/QCompress
"""

export Config

mutable struct Config
    project_name::String
    n_qubits::Int
    latent_dim::Int
    epochs::Int
    batch_size::Int
    learning_rate::Float64

    function Config(; project_name::String="QuantumAutoencoder",
                      n_qubits::Int=2,
                      latent_dim::Int=1,
                      epochs::Int=100,
                      batch_size::Int=32,
                      learning_rate::Float64=1e-3)
        new(project_name, n_qubits, latent_dim, epochs, batch_size, learning_rate)
    end
end

end # module QAE

