using LinearAlgebra
using TensorOperations

# DF-RHF analytic Hessian. Extends CPHF.jl/RHFhess.jl the same way DFgrad.jl
# extends RHFgrad.jl -- see DFgrad.jl's header for the underlying RI-HF
# energy expression and the already-validated (finite difference, ~1e-8,
# first attempt) Coulomb/exchange gradient formulas this builds on:
#
#   c_P := D_mn*(mn|P), d_P := J^-1_PQ*c_Q            (Coulomb)
#   W[m,i,P] := sum_n C_ni*(mn|P), V := W*J^-1         (exchange, i=occupied)
#
# Phase 3 (this section): Jq_DF/Kq_DF, the DF analog of CPHF.jl's
# eri_grad_JK -- matrix-valued (not summed against a second D/P) first
# derivatives, needed for cphf_rhs's "skeleton" term. Derivation (worked out
# and independently re-derived during planning, see the DF-RHF Analytic
# Hessian plan's Theory section):
#
#   Jq_DF[m,n] = sum_P [d(mn|P)/dA]*d_P + sum_P (mn|P)*(dd_P/dA - f_P)
#   Kq_DF[m,n] = sum_i,P dW[m,i,P]*V[n,i,P] + sum_i,P W[m,i,P]*dV_expl[n,i,P]
#                - sum_i V_i[m,:]'*(dJ/dA)*V_i[n,:]
#
# where dc_P/dA := sum_mn D_mn*d(mn|P)/dA, dd_P/dA := J^-1*(dc/dA),
# f_P := J^-1*(dJ/dA)*d, dW[m,i,P] := sum_n C_ni*d(mn|P)/dA (same C, fixed --
# no CPHF/orbital response needed for a FIRST derivative, same reasoning as
# the gradient), dV_expl := dW*J^-1 (the "explicit" part of dV/dA, not
# including the J^-1-derivative piece, which the last Kq_DF term handles
# separately -- exactly mirroring how DFgrad.jl splits the Coulomb term into
# dc.d and -0.5*d'(dJ/dA)d).
#
# Free, FD-independent correctness check: Kq_DF must come out symmetric
# (Kq_DF ≈ Kq_DF'), same as the exact-ERI Kq is provably symmetric --
# swapping m<->n in the formula above swaps the first two terms into each
# other and leaves the third manifestly symmetric (dJ/dA is a symmetric
# matrix). Checked as part of this phase's validation.
#
# eri_grad_JK (CPHF.jl) itself is untouched -- cphf_rhs/_rhf_hess_response
# (CPHF.jl/RHFhess.jl) instead get DF-specific counterparts here
# (cphf_rhs_df/cphf_solve_df) that call eri_grad_JK_df, reusing
# cphf_Amatvec/build_response_fock! unchanged (they only touch build_fock!,
# already eri_type-generic).

"""
    build_df_hess_cache(wfn::RHF, ints::IntegralHelper{Float64,<:AbstractDFERI})

Atom-independent quantities for the DF-RHF Hessian, built once and reused
across every atom: the 3-center/2-center integral tensors (`Pmn`, `Jmat`),
the metric inverse (`Jinv`), and the occupied-orbital half-transform (`W`,
`V`) -- exactly the objects `DFgrad.jl` builds before its own atom loop,
reused here to avoid recomputing an `naux x naux` inversion or the `W`
half-transform once per atom.
"""
function build_df_hess_cache(wfn::RHF, ints::Fermi.Integrals.IntegralHelper{Float64,<:Fermi.Integrals.AbstractDFERI})
    bset = BasisSet(ints.basis, ints.molecule.atoms)
    auxbset = ints.eri_type.basisset
    ndocc = wfn.ndocc
    Co = wfn.orbitals.C[:, 1:ndocc]
    D = Co * Co'

    Pmn = GaussianBasis.ERI_2e3c(bset, auxbset)
    Jmat = GaussianBasis.ERI_2e2c(auxbset)
    Jinv = inv(Jmat)

    @tensor c[A] := D[m, n] * Pmn[m, n, A]
    d = Jinv * c

    @tensor W[m, i, A] := Co[n, i] * Pmn[m, n, A]
    @tensor V[m, i, A] := W[m, i, B] * Jinv[B, A]

    return (bset=bset, auxbset=auxbset, Co=Co, D=D, Pmn=Pmn, Jmat=Jmat, Jinv=Jinv,
            c=c, d=d, W=W, V=V)
end

"""
    df_hess_atom_derivs(cache, iA)

Per-atom first-derivative quantities for atom `iA`, all three Cartesian
directions: `Jq`/`Kq` (see this file's header), plus the intermediates
(`dW`, `dV_expl`, `∇Jmat`) Phase 4's direct second-derivative term reuses
rather than recomputing. `eri_grad_JK_df` is a thin wrapper around this
returning just `(Jq, Kq)`.
"""
function df_hess_atom_derivs(cache, iA::Int)
    bset, auxbset = cache.bset, cache.auxbset
    D, Co = cache.D, cache.Co
    Pmn, Jinv, d, W, V = cache.Pmn, cache.Jinv, cache.d, cache.W, cache.V
    nbas = bset.nbas
    naux = auxbset.nbas

    ∇Pmn = zeros(nbas, nbas, naux, 3)
    GaussianBasis.∇ERI_2e3c!(∇Pmn, bset, auxbset, iA)
    ∇Jmat = zeros(naux, naux, 3)
    GaussianBasis.∇ERI_2e2c!(∇Jmat, auxbset, iA)

    Jq = zeros(nbas, nbas, 3)
    Kq = zeros(nbas, nbas, 3)
    dW_all = zeros(nbas, size(Co, 2), naux, 3)
    dV_expl_all = zeros(nbas, size(Co, 2), naux, 3)

    for q in 1:3
        ∇Pq = @view ∇Pmn[:, :, :, q]
        ∇Jq = @view ∇Jmat[:, :, q]

        @tensor dc[A] := D[m, n] * ∇Pq[m, n, A]
        dd = Jinv * dc
        f = Jinv * (∇Jq * d)

        @tensor Jq_q[m, n] := ∇Pq[m, n, A] * d[A] + Pmn[m, n, A] * (dd[A] - f[A])
        Jq[:, :, q] .= Jq_q

        @tensor dW[m, i, A] := Co[n, i] * ∇Pq[m, n, A]
        @tensor dV_expl[m, i, A] := dW[m, i, B] * Jinv[B, A]
        dW_all[:, :, :, q] .= dW
        dV_expl_all[:, :, :, q] .= dV_expl

        @tensor Kq_q[m, n] := dW[m, i, A] * V[n, i, A] + W[m, i, A] * dV_expl[n, i, A]
        @tensor Kcorr[m, n] := V[m, i, A] * ∇Jq[A, B] * V[n, i, B]
        Kq[:, :, q] .= Kq_q .- Kcorr
    end

    return (Jq=Jq, Kq=Kq, ∇Pmn=∇Pmn, ∇Jmat=∇Jmat, dW=dW_all, dV_expl=dV_expl_all)
end

eri_grad_JK_df(cache, iA::Int) = (r = df_hess_atom_derivs(cache, iA); (r.Jq, r.Kq))

"""
    cphf_rhs_df(wfn, ints, iA, cache)

DF analog of `cphf_rhs` (CPHF.jl) -- identical structure, using
`eri_grad_JK_df`/`cache` instead of `eri_grad_JK`/`ij_vals`/`σvals`.
`build_fock!` (via `ints`) is unchanged and already dispatches on DF.
"""
function cphf_rhs_df(wfn::RHF, ints, iA::Int, cache)
    bset = cache.bset
    nbas = bset.nbas
    ndocc = wfn.ndocc
    nvir = nbas - ndocc
    C = wfn.orbitals.C
    Co = @view C[:, 1:ndocc]
    Cv = @view C[:, ndocc+1:end]
    eps_o = wfn.orbitals.eps[1:ndocc]

    ∂H = zeros(nbas, nbas, 3)
    ∂Vtmp = zeros(nbas, nbas, 3)
    GaussianBasis.∇kinetic!(∂H, bset, iA)
    GaussianBasis.∇nuclear!(∂Vtmp, bset, iA)
    ∂H .+= ∂Vtmp

    ∂S = zeros(nbas, nbas, 3)
    GaussianBasis.∇overlap!(∂S, bset, iA)

    Jq_all, Kq_all = eri_grad_JK_df(cache, iA)

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
    cphf_solve_df(wfn, ints, iA, cache)

DF analog of `cphf_solve` (CPHF.jl). `cphf_Amatvec` is reused unchanged --
it only calls `build_fock!`, already eri_type-generic.
"""
function cphf_solve_df(wfn::RHF, ints, iA::Int, cache)
    ndocc = wfn.ndocc
    nbas = size(wfn.orbitals.C, 1)
    nvir = nbas - ndocc
    C = wfn.orbitals.C
    Co = @view C[:, 1:ndocc]
    Cv = @view C[:, ndocc+1:end]
    eps = wfn.orbitals.eps
    Δeps = eps[ndocc+1:end] .- eps[1:ndocc]'

    maxiter = Options.get("cphf_max_iter")
    tol = Options.get("cphf_conv")

    B = cphf_rhs_df(wfn, ints, iA, cache)
    U = zeros(nvir, ndocc, 3)
    for q in 1:3
        matvec(x) = Δeps .* x .+ cphf_Amatvec(x, Co, Cv, ints)
        sol, info = KrylovKit.linsolve(matvec, -B[:, :, q], zeros(nvir, ndocc), KrylovKit.CG(; maxiter=maxiter, tol=tol))
        U[:, :, q] .= sol
    end
    return U
end
