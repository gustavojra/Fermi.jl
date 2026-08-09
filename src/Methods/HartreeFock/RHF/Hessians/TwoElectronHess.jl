# Two-electron (Coulomb + exchange) contribution to the RHF Hessian, contracted
# against a FIXED density matrix P -- i.e. the "integral response"/direct piece
# only (mirrors RHFgrad.jl's existing gradient formula one derivative order up;
# CPHF's density-*response* contribution, needing dP/dR, is a separate term
# assembled elsewhere). Never materializes the raw (nbas,nbas,nbas,nbas,3,3) ERI
# Hessian tensor -- contracts against P immediately, shell-quartet by
# shell-quartet, same spirit as a Fock build.
#
# GaussianBasis.jl's ∇2ERI_2e4c(bset,iA,iB,i,j,k,l) is the light-bridge
# shell-quartet Hessian primitive this contracts against -- the raw libcint
# kernel plumbing this file used to do directly (cint2e_ipip1/ipvip1/ip1ip2,
# with their non-obvious kernel-to-shell-pair mapping and reversed axis
# orientation for cross-shell kernels) now lives there instead, ported
# verbatim rather than touched, since getting that right the first time
# already needed careful cint_funcs.h reading and finite-difference checks.
# ∇2ERI_2e4c does its own posA/posB summation internally (every valid
# derivative-landing-position pair for the requested (iA,iB)), so this file
# only ever needs ONE call per shell quartet per term (C1 or C2), not the
# nested posA/posB loop the raw-kernel version needed.
#
# Schwarz screening, added here (previously absent entirely -- this loop was
# unrestricted nshells^4 with only atom-membership filtering): C1 needs
# (pq|rs), screened by σ_pq·σ_rs; C2 needs the DIFFERENT physical integral
# (ps|rq) (not a reweighting of (pq|rs) -- a genuinely different AO-index
# pairing), so it needs its OWN, separately-computed bound σ_ps·σ_rq. Reusing
# C1's bound for C2 (or vice versa) would be a real bug, not just a missed
# optimization -- the two bounds can differ arbitrarily depending on the
# shell quartet, and a quartet screened out for one term can still matter for
# the other. This is exactly analogous to how the *gradient*'s single-integral
# formula avoids the issue entirely (its `4PP-PP-PP` terms all reweight the
# SAME integral value, since AO permutation symmetry keeps them tied to one
# shell quartet) -- the Hessian's C1/C2 split doesn't have that luxury because
# `∇2ERI_2e4c(bset,iA,iB,p,s,r,q)` is a genuinely different integral, not a
# relabeling of `∇2ERI_2e4c(bset,iA,iB,p,q,r,s)`.
#
# Still an unrestricted nshells^4 shell-quartet loop (no permutation-symmetry
# reduction, unlike RHFgrad.jl's canonical-quartet-only scheme) -- Schwarz
# screening alone is the fix being applied here; revisit the reduction
# separately if profiling calls for it.

"""
    ERI_hess_JK(bset::BasisSet, P::AbstractMatrix, iA::Int, iB::Int)

Two-electron Coulomb+exchange contribution to the RHF Hessian block (iA,iB),
`0.5*C1 - 0.25*C2` where `C1[x,y] = sum_mnrs P[m,n]*P[r,s]*d2(mn|rs)/dA_x dB_y`
(Coulomb-shape) and `C2[x,y] = sum_mnrs P[m,n]*P[r,s]*d2(ms|rn)/dA_x dB_y`
(exchange-shape) -- matching RHFgrad.jl's existing `0.5*P*P*ERI - 0.25*P*P*ERI`
gradient contraction one derivative order up. `P` is treated as fixed (the
converged SCF density); the CPHF density-response contribution is a separate
term. See this file's header comment for the Schwarz-screening/integral-direct
strategy.
"""
function ERI_hess_JK(bset::BasisSet, P::AbstractMatrix, iA::Int, iB::Int)
    Nvals = GaussianBasis.num_basis.(bset.basis)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:bset.nshells]
    nshells = bset.nshells

    _, σvals = GaussianBasis.schwarz_bounds(bset)
    cutoff = 1e-12

    Aat = bset.atoms[iA]
    Bat = bset.atoms[iB]

    C1 = zeros(3, 3)
    C2 = zeros(3, 3)

    # Atom-membership pre-filter, same as ∇2ERI_2e4c's own internal check but
    # done here too so quartets touching neither atom (the vast majority, for
    # any molecule with more than a couple atoms) never even reach a function
    # call into the primitive -- mirrors RHFgrad.jl's own on_atom pre-filter.
    @inbounds for p in 1:nshells, q in 1:nshells, r in 1:nshells, s in 1:nshells
        shellval = (p, q, r, s)
        any(ii -> bset.basis[shellval[ii]].atom == Aat, 1:4) || continue
        any(ii -> bset.basis[shellval[ii]].atom == Bat, 1:4) || continue

        σpq = σvals[GaussianBasis.index2(p-1,q-1)+1]
        σrs = σvals[GaussianBasis.index2(r-1,s-1)+1]
        σps = σvals[GaussianBasis.index2(p-1,s-1)+1]
        σrq = σvals[GaussianBasis.index2(r-1,q-1)+1]

        skip_C1 = σpq*σrs <= cutoff
        skip_C2 = σps*σrq <= cutoff
        (skip_C1 && skip_C2) && continue

        Np, Nq, Nr, Ns = Nvals[p], Nvals[q], Nvals[r], Nvals[s]
        poff, qoff, roff, soff = ao_offset[p], ao_offset[q], ao_offset[r], ao_offset[s]
        Pp = (poff+1):(poff+Np)
        Qq = (qoff+1):(qoff+Nq)
        Rr = (roff+1):(roff+Nr)
        Ss = (soff+1):(soff+Ns)
        Pmn = @view P[Pp, Qq]
        Prs = @view P[Rr, Ss]

        # --- C1: Coulomb-shape, integral (pq|rs) itself ---
        if !skip_C1
            d1 = GaussianBasis.∇2ERI_2e4c(bset, iA, iB, p, q, r, s)
            for y in 1:3, x in 1:3
                acc = 0.0
                for s_ in 1:Ns, r_ in 1:Nr, n in 1:Nq, m in 1:Np
                    acc += Pmn[m, n] * Prs[r_, s_] * d1[m, n, r_, s_, x, y]
                end
                C1[x, y] += acc
            end
        end

        # --- C2: exchange-shape, integral (ps|rq) [same AO indices m,n,r_,s_
        # weighted the same way, but a genuinely different physical integral --
        # shells differentiated in (p,s,r,q) role order] ---
        if !skip_C2
            d2 = GaussianBasis.∇2ERI_2e4c(bset, iA, iB, p, s, r, q)
            # d2 is in (p,s,r,q) AO-axis order -> reorder to (p,q,r,s) i.e. (m,n,r_,s_)
            for y in 1:3, x in 1:3
                acc = 0.0
                for s_ in 1:Ns, r_ in 1:Nr, n in 1:Nq, m in 1:Np
                    acc += Pmn[m, n] * Prs[r_, s_] * d2[m, s_, r_, n, x, y]
                end
                C2[x, y] += acc
            end
        end
    end

    return 0.5 .* C1 .- 0.25 .* C2
end
