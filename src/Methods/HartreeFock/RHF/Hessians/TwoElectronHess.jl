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
#
# Visits only CANONICAL shell quadruples (i≤j, k≤l, (i,j)-pair ≥ (k,l)-pair --
# the same restriction RHFgrad.jl's outer loop uses), one `∇2ERI_2e4c` call
# each, rather than the unrestricted nshells^4 loop (up to 8x more calls) an
# earlier version of this file used. Getting BOTH C1 and C2's full
# contribution (summed over all 8 permutation-symmetric arrangements of that
# quadruple) from that SINGLE call needed its own derivation, verified
# against a brute-force reference (each of the 8 arrangements computed
# independently via its own permuted view of the canonical block, exactly as
# `RHFgrad.jl`'s own `quartet_contribution` derivation was verified) over
# 1000 random trials before trusting it:
#
#   C1 (Coulomb-shape, weight P[I,J]⊗P[K,L]) is IDENTICAL across all 8
#   arrangements -- same result RHFgrad.jl's Coulomb term found. So a
#   canonical quadruple's total C1 contribution is just `mult * Vcan_C1`,
#   `mult` being the count of distinct arrangements (1, 2, 4, or 8 depending
#   on i==j?/k==l?/(i,j)-pair==(k,l)-pair? degeneracies -- same combinatorics
#   as RHFgrad.jl's `quartet_contribution`).
#
#   C2 (exchange-shape, weight P[I,L]⊗P[K,J]) splits into exactly the same
#   "canonical-class"/"cross-class" (4-and-4, or fewer when degenerate)
#   pattern RHFgrad.jl's own exchange term did -- `half*Vcan_C2 +
#   half*Vcross_C2` in general, `Vcan_C2` alone (with the right multiplier)
#   when i==j and k==l simultaneously (bra-swap and ket-swap both become
#   no-ops, so there's no "cross" arrangement to distinguish).
#
# A pleasant side effect: since C1 and C2 now both derive from the SAME
# canonical block (i,j,k,l) -- not two independent calls
# (∇2ERI_2e4c(...,p,q,r,s) for C1, ∇2ERI_2e4c(...,p,s,r,q) for C2, a genuinely
# different integral needing its own Schwarz bound, as an earlier version of
# this file had to handle) -- one Schwarz screening decision (σ_ij·σ_kl)
# now correctly covers both terms. There's no longer a "different bound for
# each term" subtlety to get right.

function contract_C1(P, I, J, K, L, blk)
    Pij = @view P[I, J]; Pkl = @view P[K, L]
    c = 0.0
    @inbounds for d in axes(blk,4), cc in axes(blk,3), b in axes(blk,2), a in axes(blk,1)
        c += Pij[a,b]*Pkl[cc,d]*blk[a,b,cc,d]
    end
    return c
end

function contract_C2_canonical(P, I, J, K, L, blk)
    Pil = @view P[I, L]; Pkj = @view P[K, J]
    e = 0.0
    @inbounds for d in axes(blk,4), cc in axes(blk,3), b in axes(blk,2), a in axes(blk,1)
        e += Pil[a,d]*Pkj[cc,b]*blk[a,b,cc,d]
    end
    return e
end

function contract_C2_cross(P, I, J, K, L, blk)
    Pjl = @view P[J, L]; Pki = @view P[K, I]
    e = 0.0
    @inbounds for d in axes(blk,4), cc in axes(blk,3), b in axes(blk,2), a in axes(blk,1)
        e += Pjl[b,d]*Pki[cc,a]*blk[a,b,cc,d]
    end
    return e
end

"""
    ERI_hess_JK(bset::BasisSet, P::AbstractMatrix, iA::Int, iB::Int)

Two-electron Coulomb+exchange contribution to the RHF Hessian block (iA,iB),
`0.5*C1 - 0.25*C2` where `C1[x,y] = sum_mnrs P[m,n]*P[r,s]*d2(mn|rs)/dA_x dB_y`
(Coulomb-shape) and `C2[x,y] = sum_mnrs P[m,n]*P[r,s]*d2(ms|rn)/dA_x dB_y`
(exchange-shape) -- matching RHFgrad.jl's existing `0.5*P*P*ERI - 0.25*P*P*ERI`
gradient contraction one derivative order up. `P` is treated as fixed (the
converged SCF density); the CPHF density-response contribution is a separate
term. See this file's header comment for the canonical-quadruple/Schwarz
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

    @inbounds for i in 1:nshells, j in i:nshells
        σij = σvals[GaussianBasis.index2(i-1,j-1)+1]
        ij_idx = GaussianBasis.index2(i-1,j-1)
        for k in 1:nshells, l in k:nshells
            ij_idx < GaussianBasis.index2(k-1,l-1) && continue
            σkl = σvals[GaussianBasis.index2(k-1,l-1)+1]
            σij*σkl <= cutoff && continue

            shellval = (i, j, k, l)
            any(p -> bset.basis[shellval[p]].atom == Aat, 1:4) || continue
            any(p -> bset.basis[shellval[p]].atom == Bat, 1:4) || continue

            Ni, Nj, Nk, Nl = Nvals[i], Nvals[j], Nvals[k], Nvals[l]
            ioff, joff, koff, loff = ao_offset[i], ao_offset[j], ao_offset[k], ao_offset[l]
            I = (ioff+1):(ioff+Ni)
            J = (joff+1):(joff+Nj)
            K = (koff+1):(koff+Nk)
            L = (loff+1):(loff+Nl)

            blk_full = GaussianBasis.∇2ERI_2e4c(bset, iA, iB, i, j, k, l)

            ij_ne_kl = ij_idx != GaussianBasis.index2(k-1,l-1)
            mult = (i != j ? 2 : 1) * (k != l ? 2 : 1) * (ij_ne_kl ? 2 : 1)

            for y in 1:3, x in 1:3
                blk = @view blk_full[:,:,:,:,x,y]

                C1[x, y] += mult * contract_C1(P, I, J, K, L, blk)

                if i == j && k == l
                    Vcan_C2 = contract_C2_canonical(P, I, J, K, L, blk)
                    C2[x, y] += ij_ne_kl ? 2*Vcan_C2 : Vcan_C2
                else
                    Vcan_C2 = contract_C2_canonical(P, I, J, K, L, blk)
                    Vcross_C2 = contract_C2_cross(P, I, J, K, L, blk)
                    half = mult ÷ 2
                    C2[x, y] += half*Vcan_C2 + half*Vcross_C2
                end
            end
        end
    end

    return 0.5 .* C1 .- 0.25 .* C2
end
