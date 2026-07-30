using LoopVectorization
using LinearAlgebra
using Octavian

function RMP2_canonical_energy(ints::IntegralHelper{T,RIFIT,O}, Alg::RMP2b) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}
    Bvo = permutedims(ints["BOV"], (1,3,2))
    ϵo = ints["Fii"]
    ϵv = ints["Faa"]

    output(" Computing DF-MP2 Energy")
    output(" - Contraction engine: Octavian")
    v_size = length(ϵv)
    o_size = length(ϵo)

    TWO = T(2.0)
    ONE = one(T)

    # Streaming worker-pool: each task owns its own local Bab buffer and energy
    # accumulator, pulling `i` values off a shared queue, instead of indexing
    # per-thread arrays by Threads.threadid() (unsafe under task migration).
    ntasks = Threads.nthreads()
    requests = Channel{Int}(o_size)
    for i in 1:o_size
        put!(requests, i)
    end
    close(requests)

    results = Channel{T}(ntasks)

    t = @elapsed begin
        @sync for _ in 1:ntasks
            Threads.@spawn begin
            Bab = zeros(T, v_size, v_size)
            ΔMP2_local = zero(T)

            for i in requests
                @views Bi = Bvo[:,:,i]

                for j in i:o_size

                    @views Bj = Bvo[:,:,j]
                    matmul_serial!(Bab, transpose(Bi), Bj)

                    eij = ϵo[i] + ϵo[j]
                    E = zero(T)
                    @turbo for a = eachindex(ϵv)
                        eija = eij - ϵv[a]
                        for b = eachindex(ϵv)
                            D = eija - ϵv[b]
                            E += Bab[a,b] * (TWO * Bab[a,b] - Bab[b,a]) / D
                        end
                    end
                    fac = i !== j ? TWO : ONE
                    ΔMP2_local += fac * E
                end
            end
            put!(results, ΔMP2_local)
            end # spawn
        end # sync

        Emp2 = zero(T)
        for _ in 1:ntasks
            Emp2 += take!(results)
        end
    end # time
    output("Done in {:5.5f} seconds.", t)

    return Emp2
end