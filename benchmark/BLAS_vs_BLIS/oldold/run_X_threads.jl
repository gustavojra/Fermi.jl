using Fermi
using LinearAlgebra

Nt = Threads.nthreads()
Fermi.tblis_set_num_threads(Nt)
BLAS.set_num_threads(Nt)
output("NUMBER OF THREADS: {}", Threads.nthreads())

function get_hc(N)

    molstring = ""
    f = false

    for l = eachline(joinpath(@__DIR__, "alkanes.xyz"))
        if occursin("C$N", l)
            f = true
            continue   
        end

        if f && occursin("--", l)
            break
        end

        if f
            molstring *= l*"\n"
        end
    end

    Fermi.Options.set("molstring", molstring)
end

MAX_CHAIN_LENGTH = 10

# Cold run
@set printstyle none
@energy CCSD

@reset
@set {
    basis cc-pvdz
    printstyle file
}

for N = 1:MAX_CHAIN_LENGTH

    get_hc(N)
    fc = 2*N
    Fermi.Options.set("drop_occ", fc)

    run(`echo TBLIS N = $N Threads = $Nt`)
    @set tblis true
    Fermi.Options.set("output", "TBLIS_C"*"$N"*"_T"*"$Nt"*".out")
    output("NUMBER OF THREADS: {}", Threads.nthreads())
    @energy ccsd

    run(`echo BLAS N = $N Threads = $Nt`)
    @set tblis false
    output("NUMBER OF THREADS: {}", Threads.nthreads())
    Fermi.Options.set("output", "BLAS_C"*"$N"*"_T"*"$Nt"*".out")
    @energy ccsd
end
