
module QAE

"""
Flux-based Quantum Autoencoder engine port.

This file ports the high-level API of `qae_engine.py` to a Julia/Flux-based
autoencoder. Rather than executing quantum circuits, the port provides a
classical autoencoder implemented with `Flux.jl` to enable training loops and
behaviour similar to the original (train/test split, compute loss, train, predict).

Note: This is a pragmatic port to get a working training loop using Flux. If you
want a true circuit-based QAE, the model building and execution must be
replaced by quantum-circuit backends.
"""

export QAEEngine, compress, decompress, train, predict, train_test_split, compute_loss

using Flux
using Statistics
using Random
using ..QuantumAutoencoder: Config

mutable struct QAEEngine
    cfg::Config
    data::Union{Nothing, Array{Float64,2}}
    train_indices::Vector{Int}
    test_indices::Vector{Int}
    model::Chain
    encoder::Chain
    decoder::Chain
    opt::Any
    trained::Bool
    train_history::Vector{Float64}
    test_history::Vector{Float64}
    optimized_params::Any

    function QAEEngine(cfg::Config; data=nothing)
        # Build a simple MLP autoencoder by default
        n_in = cfg.n_qubits
        n_latent = cfg.latent_dim
        # encoder: single Dense layer
        enc = Chain(Dense(n_in, n_latent, relu))
        dec = Chain(Dense(n_latent, n_in))
        model = Chain(enc, dec)
        opt = ADAM(cfg.learning_rate)
        new(cfg, data, Int[], Int[], model, enc, dec, opt, false, Float64[], Float64[], nothing)
    end
end

"""train_test_split(engine; train_indices=nothing, train_ratio=0.25)

Divide data into training and test sets. Data must be supplied to the engine
either at construction time or set to `engine.data` before calling this.
"""
function train_test_split(engine::QAEEngine; train_indices=nothing, train_ratio=0.25)
    if engine.data === nothing
        throw(ErrorException("No data present on engine. Set `engine.data` to a samples x features matrix."))
    end
    data_size = size(engine.data, 1)
    if train_indices !== nothing
        engine.train_indices = collect(train_indices)
        if any(i->i < 1 || i > data_size, engine.train_indices)
            throw(ErrorException("Invalid training index/indices. They must be >= 1 and <= data_size"))
        end
    else
        ntrain = max(1, Int(round(train_ratio * data_size)))
        engine.train_indices = rand(1:data_size, ntrain)
    end
    engine.test_indices = collect(setdiff(1:data_size, engine.train_indices))
    return nothing
end

"""compute_loss(engine, indices)

Compute mean MSE loss for the samples specified by `indices`.
"""
function compute_loss(engine::QAEEngine, indices::Vector{Int})
    if engine.data === nothing
        throw(ErrorException("No data present on engine."))
    end
    if isempty(indices)
        return 0.0
    end
    # Prepare input as features x batch matrix for Flux
    batch = permutedims(engine.data[indices, :])  # features x samples
    preds = engine.model(batch)
    losses = mean.(sum((preds .- batch) .^ 2; dims=1))
    return Float64(mean(losses))
end

"""train(engine, initial_guess=nothing)

Train the autoencoder using Flux optimizers. `initial_guess` is ignored for
Flux-based training; kept for compatibility with the original API.
"""
function train(engine::QAEEngine, initial_guess=nothing)
    if engine.data === nothing
        throw(ErrorException("No data present on engine. Set `engine.data` before training."))
    end
    if isempty(engine.train_indices) || isempty(engine.test_indices)
        throw(ErrorException("Please split your data set into training and test sets before training."))
    end

    cfg = engine.cfg
    # Build dataset as list of column matrices
    train_x = permutedims(engine.data[engine.train_indices, :])  # features x samples
    n_samples = size(train_x, 2)

    batch_size = min(cfg.batch_size, n_samples)
    epochs = cfg.epochs

    ps = Flux.params(engine.model)
    loss_fn(x) = mean(sum((engine.model(x) .- x) .^ 2; dims=1))

    for epoch in 1:epochs
        # Shuffle samples
        perm = randperm(n_samples)
        epoch_losses = Float64[]
        i = 1
        while i <= n_samples
            batch_idx = perm[i:min(i+batch_size-1, n_samples)]
            xbatch = train_x[:, batch_idx]
            gs = gradient(ps) do
                l = loss_fn(xbatch)
                push!(epoch_losses, Float64(l))
                return l
            end
            Flux.Optimise.update!(engine.opt, ps, gs)
            i += batch_size
        end
        mean_epoch_loss = mean(epoch_losses)
        push!(engine.train_history, mean_epoch_loss)
        if cfg.verbose && (epoch % cfg.print_interval == 0 || epoch == 1)
            @info "Epoch $epoch Mean Loss: $mean_epoch_loss"
        end
    end

    # mark trained and store optimized params as a copy of weights
    engine.trained = true
    engine.optimized_params = [Array(p) for p in ps]

    # compute final avg loss on training set
    final_loss = compute_loss(engine, engine.train_indices)
    return final_loss
end

"""predict(engine)

Compute mean loss against the test set using trained parameters.
"""
function predict(engine::QAEEngine)
    if !engine.trained
        throw(ErrorException("Parameters have not yet been optimized. Please train first."))
    end
    avg_loss = compute_loss(engine, engine.test_indices)
    push!(engine.test_history, avg_loss)
    if engine.cfg.verbose
        @info "Mean loss for test data: $avg_loss"
    end
    return avg_loss
end

"""compress(engine, data_matrix)

Apply the encoder to `data_matrix` (samples x features) and return the latent
representation as samples x latent_dim.
"""
function compress(engine::QAEEngine, data_matrix::Array{Float64,2})
    batch = permutedims(data_matrix) # features x samples
    latent = engine.encoder(batch)
    return permutedims(Array(latent)) # samples x latent_dim
end

"""decompress(engine, latent_matrix)

Apply the decoder to `latent_matrix` (samples x latent_dim) and return reconstructed
data as samples x features.
"""
function decompress(engine::QAEEngine, latent_matrix::Array{Float64,2})
    batch = permutedims(latent_matrix) # latent x samples
    recon = engine.decoder(batch)
    return permutedims(Array(recon)) # samples x features
end

end # module QAE

