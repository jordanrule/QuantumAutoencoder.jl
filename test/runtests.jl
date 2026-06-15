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

