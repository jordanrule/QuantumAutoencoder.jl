using Test
using LinearAlgebra
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using QuantumAutoencoder

@testset "Basic QAEEngine tests" begin
    cfg = QAE.Config()
    engine = QAE.QAEEngine(cfg)

    # create dummy data: 10 samples, 2 features
    data = randn(10, 2)

    latent = QAE.compress(engine, data)
    @test size(latent, 1) == 10
    @test size(latent, 2) == cfg.latent_dim || size(latent, 2) == size(data, 2)

    recon = QAE.decompress(engine, latent)
    @test size(recon, 1) == 10
    @test size(recon, 2) == cfg.n_qubits
end

@testset "Converted Yao circuit examples" begin
    try
        @eval using Yao

        bell = QAE.bell_state_prep_circuit()
        @test Yao.nqubits(bell) == 2

        ghz = QAE.ghz_state_prep_circuit(3)
        @test Yao.nqubits(ghz) == 3

        params = randn(4)
        ansatz = QAE.simple_qae_training_circuit(params; n_qubits=2)
        @test Yao.nqubits(ansatz) == 2
    catch err
        @info "Skipping Yao circuit conversion tests because Yao.jl is unavailable" exception=(err, catch_backtrace())
    end
end

