
"""
Yao-based Quantum Autoencoder engine port (initial implementation).

This module implements circuit-level APIs inspired by `qae_engine.py` from
QCompress. It attempts to use `Yao.jl` when available to construct and execute
parameterized circuits. If `Yao.jl` is not installed the module will raise a
clear error explaining how to install it.

This is a beginning port: the primary objective here is to provide the same
high-level methods as the original (construct_compression_circuit,
construct_recovery_circuit, compute_loss, train, predict) while keeping the
original intent (measurement-based cost functions, parameter optimization).
"""

using Statistics
using Random
using LinearAlgebra
using ..QuantumAutoencoder: Config


const HAVE_YAO = try
    @eval using Yao
    true
catch
    false
end

const HAVE_OPTIM = try
    @eval using Optim
    true
catch
    false
end

const HAVE_FLUX = try
    @eval using Flux
    true
catch
    false
end

export QAEEngine, ClassicalQAEEngine, AbstractQAEEngine, QuantumEngine, ClassicalEngine,
       construct_compression_circuit, construct_recovery_circuit,
       train_test_split, compute_loss_function, train, predict, compress, decompress

# Abstract types for multiple-dispatch
abstract type AbstractQAEEngine end
abstract type QuantumEngine <: AbstractQAEEngine end
abstract type ClassicalEngine <: AbstractQAEEngine end

mutable struct QAEEngine <: QuantumEngine
    q_in::Dict
    q_latent::Dict
    state_prep_circuits::Vector{Any}
    training_circuit::Function
    state_prep_circuits_dag::Vector{Any}
    training_circuit_dag::Function
    q_refresh::Dict
    trash_training::Bool
    reset::Bool
    compile_program::Bool

    data_size::Int
    train_indices::Vector{Int}
    test_indices::Vector{Int}

    n_shots::Int
    n_iter::Int
    optimized_params::Vector{Float64}
    train_history::Vector{Float64}
    test_history::Vector{Float64}
    verbose::Bool
    print_interval::Int

    memory_size::Int
    _physical_labels::Vector{Int}
    n_qubits::Int

    function QAEEngine(q_in::Dict, q_latent::Dict, state_prep_circuits::Vector, training_circuit::Function;
                       state_prep_circuits_dag::Vector=Any[], training_circuit_dag::Function=(x)->error("no training circuit dag"),
                       q_refresh=Dict(), trash_training=false, reset=true, compile_program=false,
                       n_shots=1000, verbose=true, print_interval=10)

        data_size = length(state_prep_circuits)

        # Infer number of qubits from the first state prep circuit (call if it's callable)
        first_prep = state_prep_circuits[1]
        prep_block = isa(first_prep, Function) ? first_prep() : first_prep
        n_qubits = 0
        if HAVE_YAO
            try
                n_qubits = Yao.nqubits(prep_block)
            catch
                # try field access as fallback
                if hasproperty(prep_block, :nqubits)
                    n_qubits = getproperty(prep_block, :nqubits)
                else
                    error("Unable to determine number of qubits from state_prep_circuits. Ensure Yao blocks are used.")
                end
            end
        else
            n_qubits = 0 # will error later when trying to run circuits
        end

        eng = new(q_in, q_latent, state_prep_circuits, training_circuit,
                  state_prep_circuits_dag, training_circuit_dag,
                  q_refresh, trash_training, reset, compile_program,
                  data_size, Int[], Int[], n_shots, 0, Float64[],
                  Float64[], Float64[], verbose, print_interval,
                  0, Int[], n_qubits)

        # Validate full training requirements
        if !trash_training
            if !reset && isempty(q_refresh)
                throw(ArgumentError("The full, no reset training requires q_refresh to be non-empty."))
            end
            if isempty(state_prep_circuits_dag) || isa(training_circuit_dag, Function) && training_circuit_dag == (x)->error("no training circuit dag")
                throw(ArgumentError("Full training requires daggered circuits for state preparation and training."))
            end
        else
            eng.reset = false
        end

        eng._determine_qubits_to_measure()
        return eng
    end
end

mutable struct ClassicalQAEEngine <: ClassicalEngine
    model::Any
    encoder::Any
    decoder::Any
    opt::Any
    config::Config
    data_size::Int
    train_indices::Vector{Int}
    test_indices::Vector{Int}
    verbose::Bool
    print_interval::Int
    n_iter::Int
    train_history::Vector{Float64}
    test_history::Vector{Float64}

    function ClassicalQAEEngine(cfg::Config)
        if !HAVE_FLUX
            throw(ErrorException("Flux.jl is required for the classical engine. Please add Flux via `] add Flux`."))
        end
        n_in = cfg.n_qubits
        n_latent = cfg.latent_dim
        enc = Flux.Chain(Flux.Dense(n_in, n_latent, relu))
        dec = Flux.Chain(Flux.Dense(n_latent, n_in))
        model = Flux.Chain(enc, dec)
        opt = Flux.ADAM(cfg.learning_rate)
        return new(model, enc, dec, opt, cfg, 0, Int[], Int[], cfg.epochs > 0, cfg.print_interval, 0, Float64[], Float64[])
    end
end

"""
Constructor dispatch for creating a classical engine from a Config.
This method is provided for backward compatibility.
"""
QAEEngine(cfg::Config; data=nothing) = ClassicalQAEEngine(cfg)

function Base.show(io::IO, eng::QuantumEngine)
    println(io, "Quantum QAE Engine: n_in=$(length(keys(eng.q_in))) n_latent=$(length(keys(eng.q_latent))) data_size=$(eng.data_size)")
end

function Base.show(io::IO, eng::ClassicalQAEEngine)
    println(io, "Classical QAE Engine: n_features=$(eng.config.n_qubits) n_latent=$(eng.config.latent_dim) data_size=$(eng.data_size)")
end

function _determine_qubits_to_measure!(eng::QuantumEngine)
    if eng.trash_training
        eng.memory_size = length(keys(eng.q_in)) - length(keys(eng.q_latent))
        eng._physical_labels = [v for (k,v) in eng.q_in if !(haskey(eng.q_latent, k))]
    else
        eng.memory_size = length(keys(eng.q_in))
        if eng.reset
            eng._physical_labels = collect(values(eng.q_in))
        else
            eng._physical_labels = vcat(collect(values(eng.q_latent)), collect(values(eng.q_refresh)))
        end
    end
    return nothing
end

_determine_qubits_to_measure!(eng::ClassicalQAEEngine) = nothing

function construct_compression_circuit(eng::QuantumEngine, parameters::Vector{Float64}, index::Int)
    if !HAVE_YAO
        throw(ErrorException("Yao.jl is required to construct quantum circuits. Please add Yao via `] add Yao`."))
    end
    prep = eng.state_prep_circuits[index]
    prep_block = isa(prep, Function) ? prep() : prep
    training_block = eng.training_circuit(parameters)
    # compose training after prep (apply prep then training)
    return training_block * prep_block
end

function construct_recovery_circuit(eng::QuantumEngine, parameters::Vector{Float64}, index::Int)
    if eng.trash_training
        throw(ErrorException("Invalid command for halfway training!"))
    end
    prep_dag = isa(eng.state_prep_circuits_dag[index], Function) ? eng.state_prep_circuits_dag[index]() : eng.state_prep_circuits_dag[index]
    training_dag = eng.training_circuit_dag(parameters)
    return prep_dag * training_dag
end

function _execute_circuit(eng::QuantumEngine, qae_block)
    if !HAVE_YAO
        throw(ErrorException("Yao.jl is required to execute quantum circuits. Please add Yao via `] add Yao`."))
    end
    n = eng.n_qubits
    if n <= 0
        throw(ErrorException("Number of qubits not known. Ensure state_prep_circuits are Yao blocks or supply n_qubits."))
    end
    # Create zero state and apply block
    state = Yao.zero_state(n)
    state = Yao.apply(qae_block, state)
    # Extract statevector as array
    vec = Array(state)
    probs = abs2.(vec)
    # measured qubits indices (1-based) to check for zero
    measured = eng._physical_labels[1:eng.memory_size]
    totalp = 0.0
    nstates = length(probs)
    for idx in 0:(nstates-1)
        ok = true
        for q in measured
            bit = (idx >> (q-1)) & 1
            if bit != 0
                ok = false; break
            end
        end
        if ok
            totalp += probs[idx+1]
        end
    end
    return totalp
end

function _compute_loss(eng::QuantumEngine, parameters::Vector{Float64}, history_list::Vector{Float64}, dataset_type::Int, indices::Vector{Int})
    losses = Float64[]
    for (i, index) in enumerate(indices)
        comp = construct_compression_circuit(eng, parameters, index)
        if !eng.trash_training
            rec = construct_recovery_circuit(eng, parameters, index)
            qae_block = rec * comp
        else
            qae_block = comp
        end
        single_prob = _execute_circuit(eng, qae_block)
        push!(losses, single_prob)
    end
    mean_loss = -mean(losses)
    push!(history_list, mean_loss)
    if eng.verbose
        if (length(history_list)-1) % eng.print_interval == 0
            @info "Iter $(eng.n_iter) Mean Loss: $(mean_loss)"
        end
    end
    eng.n_iter += 1
    return mean_loss
end

function compute_loss_function(eng::QuantumEngine, parameters::Vector{Float64})
    return _compute_loss(eng, parameters, eng.train_history, 0, eng.train_indices)
end

function train(eng::QuantumEngine, initial_guess::Vector{Float64})
    if isempty(eng.train_indices) || isempty(eng.test_indices)
        throw(ErrorException("Please split your data set into training and test sets before training."))
    end
    if !HAVE_OPTIM
        throw(ErrorException("Optim.jl is required for classical optimization. Please add it via `] add Optim`."))
    end
    obj = (x)->compute_loss_function(eng, x)
    # Use Nelder-Mead by default to mimic scipy.minimize behavior
    res = Optim.optimize(obj, initial_guess, NelderMead())
    try
        eng.optimized_params = Optim.minimizer(res)
    catch
        eng.optimized_params = Float64[]
    end
    if !isempty(eng.optimized_params)
        avg_loss = compute_loss_function(eng, eng.optimized_params)
    else
        avg_loss = compute_loss_function(eng, initial_guess)
    end
    if eng.verbose
        @info "Mean loss for training data: $(avg_loss)"
    end
    return avg_loss
end

function predict(eng::QuantumEngine)
    if isempty(eng.optimized_params)
        throw(ErrorException("Parameters have not yet been optimized. Please train first."))
    end
    avg_loss = _compute_loss(eng, eng.optimized_params, eng.test_history, 1, eng.test_indices)
    if eng.verbose
        @info "Mean loss for test data: $(avg_loss)"
    end
    return avg_loss
end

"""
    compress(eng::QuantumEngine, parameters::Vector{Float64}, index::Int)

Compress data using a quantum circuit (quantum mode).
"""
function compress(eng::QuantumEngine, parameters::Vector{Float64}, index::Int)
    return construct_compression_circuit(eng, parameters, index)
end

"""
    compress(eng::ClassicalQAEEngine, data::Array{Float64,2})

Compress data using the classical Flux-based autoencoder (classical mode).
"""
function compress(eng::ClassicalQAEEngine, data::Array{Float64,2})
    batch = permutedims(data) # features x samples
    latent = eng.encoder(batch)
    return permutedims(Array(latent)) # samples x latent_dim
end

"""
    decompress(eng::QuantumEngine, parameters::Vector{Float64}, index::Int)

Decompress data using a quantum circuit (quantum mode).
"""
function decompress(eng::QuantumEngine, parameters::Vector{Float64}, index::Int)
    return construct_recovery_circuit(eng, parameters, index)
end

"""
    decompress(eng::ClassicalQAEEngine, latent::Array{Float64,2})

Decompress data using the classical Flux-based autoencoder (classical mode).
"""
function decompress(eng::ClassicalQAEEngine, latent::Array{Float64,2})
    batch = permutedims(latent)
    recon = eng.decoder(batch)
    return permutedims(Array(recon))
end
