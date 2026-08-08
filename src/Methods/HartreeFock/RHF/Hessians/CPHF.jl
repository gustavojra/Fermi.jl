# CPHF (Coupled-Perturbed Hartree-Fock) orbital-response solver.
#
# The occ-virt rotation U^(y) (y = a nuclear Cartesian perturbation) that
# describes how the MOs respond to a geometry displacement solves
#
#   (eps_a - eps_i) U_ai + sum_bj [4(ai|bj) - (ab|ij) - (aj|ib)] U_bj = -B_ai
#
# The coupling-matrix action A*U (the sum over b,j) is the only genuinely new
# two-electron contraction this needs, and it turns out to be exactly a
# standard Fock-build contraction in disguise -- no new ERI-handling code,
# just a different (symmetrized, trial) density plugged into the existing
# build_fock! dispatches.
#
# Derivation: let ΔD_μν = sum_ai [C_μa U_ai C_νi + C_μi U_ai C_νa] (the
# symmetric AO-basis density built from the trial rotation U, shape
# (nvir,ndocc)), and ΔF_μν = 2*ΔD_rs*(μν|rs) - ΔD_rs*(μr|νs) -- exactly
# build_fock!'s two-electron contraction, with ΔD standing in for the density
# and no one-electron piece added. Expanding ΔD in terms of U and transforming
# ΔF to the MO occ-virt block:
#
#   sum_μν C_μa ΔF_μν C_νi
#     = 4*sum_bj U_bj*(ai|bj) - sum_bj U_bj*[(ab|ij) + (aj|ib)]
#     = sum_bj [4(ai|bj) - (ab|ij) - (aj|ib)] U_bj
#
# using (μν|jb)=(μν|bj) (AO ERI symmetry under the ket-pair swap) to combine
# the two Coulomb-shaped terms, and the same relabeling for the exchange
# piece -- exactly the CPHF coupling matrix's action on U. So:
# Cv'*ΔF*Co = A*U, with ΔF obtained by literally reusing build_fock!.

"""
    build_response_fock!(ΔF, U, Co, Cv, ints)

Two-electron "response Fock" contraction used as the CPHF orbital-Hessian
matrix-vector product A*U. `U` is a trial occupied-virtual rotation, shape
`(nvir,ndocc)`; `Co`, `Cv` are the occupied/virtual blocks of the MO
coefficient matrix. Builds the symmetric trial AO density
`ΔD = Cv*U*Co' + Co*U'*Cv'`, then reuses `build_fock!`'s existing J/K
dispatch (Chonky/DF/SparseERI, whichever `ints` carries) with a zero
one-electron piece to get `ΔF` -- see this file's header for why that's
exactly the CPHF coupling matrix in disguise.
"""
function build_response_fock!(ΔF, U, Co, Cv, ints)
    ΔD = Cv * U * Co'
    ΔD .+= ΔD'
    H0 = zeros(size(ΔF))
    build_fock!(ΔF, H0, ΔD, ints)
    return ΔF
end

"""
    cphf_Amatvec(U, Co, Cv, ints)

The CPHF coupling matrix's action on a trial occ-virt rotation `U`,
`sum_bj [4(ai|bj) - (ab|ij) - (aj|ib)] U_bj`, returned as a `(nvir,ndocc)`
matrix in the same shape/index convention as `U`. Thin wrapper around
`build_response_fock!` for use as a KrylovKit linear map.
"""
function cphf_Amatvec(U, Co, Cv, ints)
    nbas = size(Co, 1)
    ΔF = zeros(nbas, nbas)
    build_response_fock!(ΔF, U, Co, Cv, ints)
    return Cv' * ΔF * Co
end

# --- CPHF right-hand side ---
#
# The CPHF equation is (εa-εi)U_ai + (A·U)_ai = -B_ai. B has two pieces:
#
#   1. The "skeleton" Fock derivative: differentiate the AO integrals
#      directly (H, ERI) with the density D=Co*Co' held FIXED at its
#      converged value -- exactly the quantity RHFgrad.jl already builds one
#      contraction short of (it goes on to dot this against P,Q to get a
#      gradient; here it's kept as a full AO matrix and transformed to the
#      MO occ-virt block instead), minus ε_i*S_ai^(y) from the generalized
#      eigenvalue problem's metric.
#
#   2. A correction from how the occupied-occupied block of the MO
#      coefficient response is fixed (not solved for) by the orthonormality
#      constraint C(y)'S(y)C(y)=I: writing C_i^(y) = sum_p C_p U_pi, the
#      symmetric part of U is forced to U_pq+U_qp=-S_pq^(y) (MO-transformed
#      overlap derivative); the standard gauge choice U_ij=-0.5*S_ij^(y) for
#      occ-occ i,j fixes that block outright (a pure occ-occ rotation
#      wouldn't change the density; this symmetric piece isn't a rotation,
#      so it DOES contribute to dD/dy through a known, computable term,
#      dD_S/dy = -Co*S_oo^(y)*Co', S_oo^(y):=Co'S^(y)Co). Since F depends on
#      the density D exactly the same way build_response_fock! already
#      exploits, this piece's contribution to B is just
#      build_fock!(0, dD_S/dy, ints) transformed to the ai block -- same
#      reuse trick, no new two-electron code.

"""
    eri_grad_JK(bset::BasisSet, D::AbstractMatrix, iA::Int)

Coulomb- and exchange-shaped contractions of the *first*-derivative ERI
against a density-like matrix `D`, for all three Cartesian directions of
atom `iA`: `Jq[m,n] = sum_rs D[r,s]*d(mn|rs)/dA_q`, `Kq[m,n] = sum_rs
D[r,s]*d(mr|ns)/dA_q`. Returned as two `(nbas,nbas,3)` arrays.

Built directly from `GaussianBasis.∇sparseERI_2e4c`'s compressed
(permutation-unique) derivative list -- never materializes the dense
`(nbas,nbas,nbas,nbas,3)` derivative ERI tensor `∇ERI_2e4c!` would. For
each stored unique quartet, accumulates into `Jq`/`Kq` over the
(deduplicated) 8-member index-permutation orbit that shares its value;
validated against the dense `∇ERI_2e4c!`-based reference to machine
precision (both sto-3g and cc-pVDZ, several atoms) before use here.
"""
function eri_grad_JK(bset::BasisSet, D::AbstractMatrix, iA::Int)
    nbas = bset.nbas
    Jq = zeros(nbas, nbas, 3)
    Kq = zeros(nbas, nbas, 3)
    idx, xyz... = GaussianBasis.∇sparseERI_2e4c(bset, iA)
    for i in eachindex(idx)
        μ, ν, λ, σ = Int.(idx[i])
        orbit = unique([(μ, ν, λ, σ), (ν, μ, λ, σ), (μ, ν, σ, λ), (ν, μ, σ, λ),
                        (λ, σ, μ, ν), (σ, λ, μ, ν), (λ, σ, ν, μ), (σ, λ, ν, μ)])
        for q in 1:3
            ∇k = xyz[q][i]
            abs(∇k) < 1e-12 && continue
            for (a, b, c, d) in orbit
                Jq[a, b, q] += D[c, d] * ∇k
                Kq[a, c, q] += D[b, d] * ∇k
            end
        end
    end
    return Jq, Kq
end

"""
    cphf_rhs(wfn, ints, iA)

CPHF right-hand side `B_ai^(y)` for all three Cartesian directions of atom
`iA`, returned as a `(nvir,ndocc,3)` array. Built entirely from existing
first-derivative integrals (`∇kinetic!`/`∇nuclear!`/`∇overlap!`/
`eri_grad_JK`) contracted against the converged density -- no
second-derivative integrals involved (those are the direct/"integral
response" Hessian piece, `ERI_hess_JK`/the one-electron Hessian code,
assembled separately).
"""
function cphf_rhs(wfn::RHF, ints, iA)
    bset = BasisSet(wfn.orbitals.basis, wfn.molecule.atoms)
    nbas = bset.nbas
    ndocc = wfn.ndocc
    nvir = nbas - ndocc
    C = wfn.orbitals.C
    Co = @view C[:, 1:ndocc]
    Cv = @view C[:, ndocc+1:end]
    eps_o = wfn.orbitals.eps[1:ndocc]
    D = Co * Co'

    ∂H = zeros(nbas, nbas, 3)
    ∂Vtmp = zeros(nbas, nbas, 3)
    GaussianBasis.∇kinetic!(∂H, bset, iA)
    GaussianBasis.∇nuclear!(∂Vtmp, bset, iA)
    ∂H .+= ∂Vtmp

    ∂S = zeros(nbas, nbas, 3)
    GaussianBasis.∇overlap!(∂S, bset, iA)

    Jq_all, Kq_all = eri_grad_JK(bset, D, iA)

    B = zeros(nvir, ndocc, 3)
    G = zeros(nbas, nbas)
    H0 = zeros(nbas, nbas)
    for q in 1:3
        Fskel = @view(∂H[:, :, q]) .+ 2 .* (@view Jq_all[:, :, q]) .- (@view Kq_all[:, :, q])

        Sq = @view ∂S[:, :, q]
        S_oo = Co' * Sq * Co
        dD_S = -Co * S_oo * Co'
        build_fock!(G, H0, dD_S, ints)

        B[:, :, q] .= Cv' * Fskel * Co .- (Cv' * Sq * Co) * Diagonal(eps_o) .+ Cv' * G * Co
    end

    return B
end

"""
    cphf_solve(wfn, ints, iA)

Solve the CPHF equations for the occupied-virtual orbital response
`U_ai^(y)` to displacing atom `iA`, for all three Cartesian directions,
returned as a `(nvir,ndocc,3)` array. Solves
`(εa-εi)U + cphf_Amatvec(U) = -cphf_rhs(...)` via `KrylovKit.linsolve` with
CG (the orbital Hessian is symmetric positive definite at a true RHF
minimum), one right-hand side at a time.
"""
function cphf_solve(wfn::RHF, ints, iA)
    ndocc = wfn.ndocc
    nbas = size(wfn.orbitals.C, 1)
    nvir = nbas - ndocc
    C = wfn.orbitals.C
    Co = @view C[:, 1:ndocc]
    Cv = @view C[:, ndocc+1:end]
    eps = wfn.orbitals.eps
    Δeps = eps[ndocc+1:end] .- eps[1:ndocc]'  # (nvir,ndocc), εa - εi

    maxiter = Options.get("cphf_max_iter")
    tol = Options.get("cphf_conv")

    B = cphf_rhs(wfn, ints, iA)
    U = zeros(nvir, ndocc, 3)
    for q in 1:3
        matvec(x) = Δeps .* x .+ cphf_Amatvec(x, Co, Cv, ints)
        sol, info = KrylovKit.linsolve(matvec, -B[:, :, q], zeros(nvir, ndocc), KrylovKit.CG(; maxiter=maxiter, tol=tol))
        U[:, :, q] .= sol
    end
    return U
end
