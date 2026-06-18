"""
Example Yao circuit conversions inspired by common pyQuil patterns in QCompress workflows.

These helpers provide ready-to-use state preparation and training circuits so users
can start in circuit mode without translating everything themselves.
"""

export bell_state_prep_circuit, ghz_state_prep_circuit, simple_qae_training_circuit

const HAVE_YAO_CIRCUITS = try
    @eval using Yao
    true
catch
    false
end

function _require_yao_for_circuits()
    if !HAVE_YAO_CIRCUITS
        throw(ErrorException("Yao.jl is required for circuit examples. Please add Yao via `] add Yao`."))
    end
end

"""
    bell_state_prep_circuit()

A direct Yao equivalent of a basic pyQuil Bell-state preparation:
`H(0); CNOT(0, 1)`.
"""
function bell_state_prep_circuit()
    _require_yao_for_circuits()
    return Yao.chain(2,
        Yao.put(2, 1 => Yao.H),
        Yao.control(2, 1, 2 => Yao.X),
    )
end

"""
    ghz_state_prep_circuit(n_qubits::Int=3)

A direct Yao equivalent of a GHZ preparation circuit:
`H(0); CNOT(0,1); CNOT(0,2); ...`.
"""
function ghz_state_prep_circuit(n_qubits::Int=3)
    _require_yao_for_circuits()
    n_qubits >= 2 || throw(ArgumentError("n_qubits must be >= 2"))

    layers = Any[Yao.put(n_qubits, 1 => Yao.H)]
    for target in 2:n_qubits
        push!(layers, Yao.control(n_qubits, 1, target => Yao.X))
    end
    return Yao.chain(n_qubits, layers...)
end

"""
    simple_qae_training_circuit(params::AbstractVector{<:Real}; n_qubits::Int=2)

A small parameterized ansatz that can be passed directly as `training_circuit`
when constructing `QAEEngine`.
"""
function simple_qae_training_circuit(params::AbstractVector{<:Real}; n_qubits::Int=2)
    _require_yao_for_circuits()
    n_qubits >= 2 || throw(ArgumentError("n_qubits must be >= 2"))
    expected = 2 * n_qubits
    length(params) == expected || throw(ArgumentError("Expected $expected parameters for n_qubits=$n_qubits, got $(length(params))."))

    layers = Any[]
    for i in 1:n_qubits
        push!(layers, Yao.put(n_qubits, i => Yao.rot(Yao.X, Float64(params[i]))))
    end
    for i in 1:(n_qubits - 1)
        push!(layers, Yao.control(n_qubits, i, (i + 1) => Yao.X))
    end
    for i in 1:n_qubits
        push!(layers, Yao.put(n_qubits, i => Yao.rot(Yao.Z, Float64(params[n_qubits + i]))))
    end

    return Yao.chain(n_qubits, layers...)
end

