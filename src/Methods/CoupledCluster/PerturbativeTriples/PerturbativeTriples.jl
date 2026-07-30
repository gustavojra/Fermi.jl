abstract type RpTAlgorithm end

function get_rpt_alg()
    # NOTE: `abc` is intentionally excluded here (see below) — it is known to
    # produce an incorrect CCSD(T) energy and is disabled pending investigation.
    implemented = [ijk(), ijk2()]
    N = Options.get("pt_alg")
    try
        return implemented[N]
    catch BoundsError
        throw(FermiException("implementation number $N not available for RCCSD(T)."))
    end
end
"""
    Fermi.CoupledCluster.RCCSDpT

Fermi struct that holds information about RCCSD(T) wavefunctions

_struct tree:_

**RCCSD(T)** <: AbstractCCWavefunction <: AbstractCorrelatedWavefunction <: AbstractWavefunction
"""
struct RCCSDpT{T} <: AbstractCCWavefunction
    CCSD::RCCSD{T}
    energy::T
    correction::T
end

"""
    Fermi.CoupledCluster.get_pt_alg

Returns a singleton type corresponding to a RCCSD(T) implementation based on the options.
"""
function get_pt_alg(N::Int = Options.get("pt_alg"))
    try 
        return get_pt_alg(Val(N))
    catch MethodError
        throw(FermiException("implementation number $N not available for RCCSD(T)."))
    end
end

# Implementations
struct ijk <: RpTAlgorithm end
include("ijk.jl")
get_pt_alg(x::Val{1}) = ijk()
struct ijk2 <: RpTAlgorithm end
include("ijk2.jl")
get_pt_alg(x::Val{2}) = ijk2()

# `abc` is a legacy (T) implementation kept for reference. It is known to give
# an incorrect energy (see conversation/commit history) and is deliberately
# left out of both `get_rpt_alg` and `get_pt_alg`'s dispatch table, so it is
# unreachable through the `pt_alg` option. The type/file are still loaded so
# the code compiles and remains available for direct, manual use
# (`RCCSDpT(ccsd, moints, Fermi.CoupledCluster.abc())`) if someone wants to
# pick the investigation back up.
struct abc <: RpTAlgorithm end
include("abc.jl")

function RCCSDpT(x...)
    if !any(i-> i isa RpTAlgorithm, x)
        RCCSDpT(x..., get_rpt_alg())
    else
        # Print the type of arguments given for a better feedback
        args = "("
        for a in x[1:end-1]
            args *= "$(typeof(a)), "
        end
        args = args[1:end-2]*")"
        throw(FermiException("invalid arguments for RCCSD(T) method: $args"))
    end
end

# Gradient methods
include("Gradients/RCCSDpTgrad.jl")