
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

export QAEEngine, construct_compression_circuit, construct_recovery_circuit,
       train_test_split, compute_loss_function, train, predict, compress, decompress

mutable struct QAEEngine
    q_in::Dict
    q_latent::Dict
    state_prep_circuits::Vector{Any}
    training_circuit::Function
    state_prep_circuits_dag::Union{Nothing, Vector{Any}}
    training_circuit_dag::Union{Nothing, Function}
    q_refresh::Dict
    trash_training::Bool
    reset::Bool
    compile_program::Bool

    data_size::Int
    train_indices::Vector{Int}
    test_indices::Vector{Int}

    n_shots::Int
    n_iter::Int
    optimized_params::Union{Nothing, Vector{Float64}}
    train_history::Vector{Float64}
    test_history::Vector{Float64}
    verbose::Bool
    print_interval::Int

    memory_size::Int
    _physical_labels::Vector{Int}

    n_qubits::Int
    # optional classical model (fallback) using Flux
    model::Union{Nothing,Any}
    encoder::Union{Nothing,Any}
    decoder::Union{Nothing,Any}
    opt::Union{Nothing,Any}
    is_classical::Bool

    function QAEEngine(q_in::Dict, q_latent::Dict, state_prep_circuits::Vector, training_circuit::Function;
                       state_prep_circuits_dag=nothing, training_circuit_dag=nothing,
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
                  data_size, Int[], Int[], n_shots, 0, nothing,
                  Float64[], Float64[], verbose, print_interval,
                  0, Int[], n_qubits, nothing, nothing, nothing, nothing, false)

        # Validate full training requirements
        if !trash_training
            if !reset && isempty(q_refresh)
                throw(ArgumentError("The full, no reset training requires q_refresh to be non-empty."))
            end
            if state_prep_circuits_dag === nothing || training_circuit_dag === nothing
                throw(ArgumentError("Full training requires daggered circuits for state preparation and training."))
            end
        else
            eng.reset = false
        end

                eng._determine_qubits_to_measure()
                return eng
            end
        end

        """
Classical constructor for compatibility with existing tests and the earlier
Flux-based scaffold. Calling `QAEEngine(cfg::Config)` returns an engine that
operates in classical mode: `compress(engine,data)` will apply a simple
Flux autoencoder.
"""
function QAEEngine(cfg::Config; data=nothing)
    if !HAVE_FLUX
        throw(ErrorException("Flux.jl is required for the classical engine. Please add Flux via `] add Flux`."))
    end
    n_in = cfg.n_qubits
    n_latent = cfg.latent_dim
    enc = Flux.Chain(Flux.Dense(n_in, n_latent, relu))
    dec = Flux.Chain(Flux.Dense(n_latent, n_in))
    model = Flux.Chain(enc, dec)
    opt = Flux.ADAM(cfg.learning_rate)
    eng = QAEEngine(Dict(), Dict(), Any[], x->error("no training circuit");
                    state_prep_circuits_dag=nothing, training_circuit_dag=nothing,
                    q_refresh=Dict(), trash_training=false, reset=true, compile_program=false,
                    n_shots=1000, verbose=cfg.epochs>0, print_interval=cfg.print_interval)
    eng.model = model
    eng.encoder = enc
    eng.decoder = dec
    eng.opt = opt
    eng.is_classical = true
    # attach data if provided
    if data !== nothing
        # expect samples x features
        eng.data_size = size(data, 1)
    end
    return eng
end

function Base.show(io::IO, eng::QAEEngine)
    println(io, "QAE Engine: n_in=$(length(keys(eng.q_in))) n_latent=$(length(keys(eng.q_latent))) data_size=$(eng.data_size)")
end

function _determine_qubits_to_measure!(eng::QAEEngine)
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

function (eng::QAEEngine)._determine_qubits_to_measure()
    return _determine_qubits_to_measure!(eng)
end

function construct_compression_circuit(eng::QAEEngine, parameters::Vector{Float64}, index::Int)
    if !HAVE_YAO
        throw(ErrorException("Yao.jl is required to construct quantum circuits. Please add Yao via `] add Yao`."))
    end
    prep = eng.state_prep_circuits[index]
    prep_block = isa(prep, Function) ? prep() : prep
    training_block = eng.training_circuit(parameters)
    # compose training after prep (apply prep then training)
    return training_block * prep_block
end

function construct_recovery_circuit(eng::QAEEngine, parameters::Vector{Float64}, index::Int)
    if eng.trash_training
        throw(ErrorException("Invalid command for halfway training!"))
    end
    prep_dag = isa(eng.state_prep_circuits_dag[index], Function) ? eng.state_prep_circuits_dag[index]() : eng.state_prep_circuits_dag[index]
    training_dag = eng.training_circuit_dag(parameters)
    return prep_dag * training_dag
end

function _execute_circuit(eng::QAEEngine, qae_block)
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

function _compute_loss(eng::QAEEngine, parameters::Vector{Float64}, history_list::Vector{Float64}, dataset_type::Int, indices::Vector{Int})
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

function compute_loss_function(eng::QAEEngine, parameters::Vector{Float64})
    return _compute_loss(eng, parameters, eng.train_history, 0, eng.train_indices)
end

function train(eng::QAEEngine, initial_guess::Vector{Float64})
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
        eng.optimized_params = nothing
    end
    if eng.optimized_params !== nothing
        avg_loss = compute_loss_function(eng, eng.optimized_params)
    else
        avg_loss = compute_loss_function(eng, initial_guess)
    end
    if eng.verbose
        @info "Mean loss for training data: $(avg_loss)"
    end
    return avg_loss
end

function predict(eng::QAEEngine)
    if eng.optimized_params === nothing
        throw(ErrorException("Parameters have not yet been optimized. Please train first."))
    end
    avg_loss = _compute_loss(eng, eng.optimized_params, eng.test_history, 1, eng.test_indices)
    if eng.verbose
        @info "Mean loss for test data: $(avg_loss)"
    end
    return avg_loss
end

function compress(eng::QAEEngine, parameters::Vector{Float64}, index::Int)
    return construct_compression_circuit(eng, parameters, index)
end

function decompress(eng::QAEEngine, parameters::Vector{Float64}, index::Int)
    return construct_recovery_circuit(eng, parameters, index)
end

# Classical-mode convenience methods for backward compatibility with tests
function compress(eng::QAEEngine, data::Array{Float64,2})
    if eng.is_classical && eng.encoder !== nothing
        batch = permutedims(data) # features x samples
        latent = eng.encoder(batch)
        return permutedims(Array(latent)) # samples x latent_dim
    else
        throw(ErrorException("compress(engine, data) is only available for classical engines. Use compress(engine, params, index) for quantum engines."))
    end
end

function decompress(eng::QAEEngine, latent::Array{Float64,2})
    if eng.is_classical && eng.decoder !== nothing
        batch = permutedims(latent)
        recon = eng.decoder(batch)
        return permutedims(Array(recon))
    else
        throw(ErrorException("decompress(engine, latent) is only available for classical engines. Use decompress(engine, params, index) for quantum engines."))
    end
end
