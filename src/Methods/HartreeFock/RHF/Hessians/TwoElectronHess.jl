using GaussianBasis.Libcint

# Two-electron (Coulomb + exchange) contribution to the RHF Hessian, contracted
# against a FIXED density matrix P -- i.e. the "integral response"/direct piece
# only (mirrors RHFgrad.jl's existing gradient formula one derivative order up;
# CPHF's density-*response* contribution, needing dP/dR, is a separate term
# assembled elsewhere). Never materializes the raw (nbas,nbas,nbas,nbas,3,3) ERI
# Hessian tensor -- contracts against P immediately, shell-quartet by
# shell-quartet, same spirit as a Fock build.
#
# Which kernel gives which shell-pair's second derivative is NOT what the "ip1",
# "ip2" naming suggests at first glance -- confirmed against libcint's own
# cint_funcs.h doc comments and validated against finite differences:
#
#   cint2e_ipip1  "(NABLA NABLA i j|R12|k l)"   both derivatives on shell i (same
#                                                shell, position 1)
#   cint2e_ipvip1 "(NABLA i NABLA j|R12|k l)"    one derivative each on shells i,j
#                                                (SAME electron pair -- i,j are
#                                                both "bra", i.e. positions 1,2)
#   cint2e_ip1ip2 "(NABLA i j|R12|NABLA k l)"    one derivative each on shells i,k
#                                                (OPPOSITE electron pairs -- i is
#                                                "bra" position 1, k is "ket"
#                                                position 3)
#
# ip1ip2 giving the opposite-pair (i.e. i vs k, not i vs j) cross derivative
# directly is the crucial piece -- with it, all 10 distinct shell-pair second
# derivatives of a quartet (same-shell x4, same-pair-cross x2, opposite-pair-cross
# x4) are directly available via these 3 kernels plus shell-argument reordering
# (exploiting the standard 8-fold ERI permutation symmetry
# (pq|rs)=(qp|rs)=(pq|sr)=(rs|pq)=...). No translational-invariance reconstruction
# is needed anywhere in this file.
#
# Every raw kernel whose two derivatives land on DIFFERENT shells (ipvip1,
# ip1ip2) has its two derivative-component axes in (position-2-or-4, position-1-
# or-3) order in the raw buffer -- reversed from the naive expectation, same
# pattern found for the one-electron mixed kernels (cint1e_iprinvip_sph!) and
# confirmed here by direct finite difference. ipip1 (same shell, mixed partials
# of the same point commute) needs no such correction.

function eri_hess_kernel(kern, bset::BasisSet, a, b, c, d)
    Na, Nb, Nc, Nd = GaussianBasis.num_basis.(bset.basis[[a, b, c, d]])
    buf = zeros(Cdouble, 9 * Na * Nb * Nc * Nd)
    kern(buf, [a, b, c, d], bset.lib)
    return permutedims(reshape(buf, Na, Nb, Nc, Nd, 3, 3), (1, 2, 3, 4, 6, 5))
end

# Second derivative of the (p,q,r,s) integral w.r.t. the shells at argument
# positions posA, posB (1..4, corresponding to p,q,r,s respectively), returned
# as a (Np,Nq,Nr,Ns,3,3) tensor in (p,q,r,s) AO-axis order regardless of which
# positions were differentiated. posA == posB (same-shell case) is handled by
# eri_hess_same below; this function only handles posA != posB.
function eri_hess_cross(bset::BasisSet, p, q, r, s, posA::Int, posB::Int)
    swp = posA > posB
    a, b = swp ? (posB, posA) : (posA, posB)

    d = if (a, b) == (1, 2)
        eri_hess_kernel(cint2e_ipvip1_sph!, bset, p, q, r, s)
    elseif (a, b) == (3, 4)
        raw = eri_hess_kernel(cint2e_ipvip1_sph!, bset, r, s, p, q)
        permutedims(raw, (3, 4, 1, 2, 5, 6))
    elseif (a, b) == (1, 3)
        eri_hess_kernel(cint2e_ip1ip2_sph!, bset, p, q, r, s)
    elseif (a, b) == (1, 4)
        raw = eri_hess_kernel(cint2e_ip1ip2_sph!, bset, p, q, s, r)
        permutedims(raw, (1, 2, 4, 3, 5, 6))
    elseif (a, b) == (2, 3)
        raw = eri_hess_kernel(cint2e_ip1ip2_sph!, bset, q, p, r, s)
        permutedims(raw, (2, 1, 3, 4, 5, 6))
    elseif (a, b) == (2, 4)
        raw = eri_hess_kernel(cint2e_ip1ip2_sph!, bset, q, p, s, r)
        permutedims(raw, (2, 1, 4, 3, 5, 6))
    else
        throw(ArgumentError("unexpected position pair ($a,$b)"))
    end

    # d(posB,posA)[...,x,y] = d(posA,posB)[...,y,x] (mixed partials commute)
    return swp ? permutedims(d, (1, 2, 3, 4, 6, 5)) : d
end

# Same-shell second derivative: both derivatives land on the shell at position
# posK (1..4). Returned in (p,q,r,s) AO-axis order.
function eri_hess_same(bset::BasisSet, p, q, r, s, posK::Int)
    if posK == 1
        eri_hess_kernel(cint2e_ipip1_sph!, bset, p, q, r, s)
    elseif posK == 2
        raw = eri_hess_kernel(cint2e_ipip1_sph!, bset, q, p, r, s)
        permutedims(raw, (2, 1, 3, 4, 5, 6))
    elseif posK == 3
        raw = eri_hess_kernel(cint2e_ipip1_sph!, bset, r, s, p, q)
        permutedims(raw, (3, 4, 1, 2, 5, 6))
    elseif posK == 4
        raw = eri_hess_kernel(cint2e_ipip1_sph!, bset, s, r, p, q)
        permutedims(raw, (3, 4, 2, 1, 5, 6))
    else
        throw(ArgumentError("posK must be 1..4"))
    end
end

"""
    ERI_hess_JK(bset::BasisSet, P::AbstractMatrix, iA::Int, iB::Int)

Two-electron Coulomb+exchange contribution to the RHF Hessian block (iA,iB),
`0.5*C1 - 0.25*C2` where `C1[x,y] = sum_mnrs P[m,n]*P[r,s]*d2(mn|rs)/dA_x dB_y`
(Coulomb-shape) and `C2[x,y] = sum_mnrs P[m,n]*P[r,s]*d2(ms|rn)/dA_x dB_y`
(exchange-shape) -- matching RHFgrad.jl's existing `0.5*P*P*ERI - 0.25*P*P*ERI`
gradient contraction one derivative order up. `P` is treated as fixed (the
converged SCF density); the CPHF density-response contribution is a separate
term. Brute-force over shell quartets (no permutational-symmetry reduction
yet) -- fine for the small molecules this is validated against so far;
revisit for performance once the full Hessian pipeline is in place.
"""
function ERI_hess_JK(bset::BasisSet, P::AbstractMatrix, iA::Int, iB::Int)
    Aat = bset.atoms[iA]
    Bat = bset.atoms[iB]
    Nvals = GaussianBasis.num_basis.(bset.basis)
    ao_offset = [sum(Nvals[1:(i-1)]) for i = 1:bset.nshells]

    C1 = zeros(3, 3)
    C2 = zeros(3, 3)

    @inbounds for p in 1:bset.nshells, q in 1:bset.nshells, r in 1:bset.nshells, s in 1:bset.nshells
        shellval = (p, q, r, s)
        Xflag = ntuple(i -> bset.basis[shellval[i]].atom == Aat, 4)
        Yflag = ntuple(i -> bset.basis[shellval[i]].atom == Bat, 4)
        any(Xflag) || continue
        any(Yflag) || continue

        Np, Nq, Nr, Ns = Nvals[p], Nvals[q], Nvals[r], Nvals[s]
        poff, qoff, roff, soff = ao_offset[p], ao_offset[q], ao_offset[r], ao_offset[s]
        Pp = (poff+1):(poff+Np)
        Qq = (qoff+1):(qoff+Nq)
        Rr = (roff+1):(roff+Nr)
        Ss = (soff+1):(soff+Ns)
        Pmn = @view P[Pp, Qq]
        Prs = @view P[Rr, Ss]

        # --- C1: Coulomb-shape, integral (pq|rs) itself ---
        for posA in 1:4
            Xflag[posA] || continue
            for posB in 1:4
                Yflag[posB] || continue
                d = posA == posB ? eri_hess_same(bset, p, q, r, s, posA) :
                                    eri_hess_cross(bset, p, q, r, s, posA, posB)
                for y in 1:3, x in 1:3
                    acc = 0.0
                    for s_ in 1:Ns, r_ in 1:Nr, n in 1:Nq, m in 1:Np
                        acc += Pmn[m, n] * Prs[r_, s_] * d[m, n, r_, s_, x, y]
                    end
                    C1[x, y] += acc
                end
            end
        end

        # --- C2: exchange-shape, integral (ps|rq) [same AO indices m,n,r_,s_
        # weighted the same way, but the shells DIFFERENTIATED are p,s,r,q in
        # that role order -- relabel positions accordingly (position2 <-> 4)]
        XflagC2 = (Xflag[1], Xflag[4], Xflag[3], Xflag[2])
        YflagC2 = (Yflag[1], Yflag[4], Yflag[3], Yflag[2])
        for posA in 1:4
            XflagC2[posA] || continue
            for posB in 1:4
                YflagC2[posB] || continue
                d = posA == posB ? eri_hess_same(bset, p, s, r, q, posA) :
                                    eri_hess_cross(bset, p, s, r, q, posA, posB)
                # d is in (p,s,r,q) AO-axis order -> reorder to (p,q,r,s) i.e. (m,n,r_,s_)
                for y in 1:3, x in 1:3
                    acc = 0.0
                    for s_ in 1:Ns, r_ in 1:Nr, n in 1:Nq, m in 1:Np
                        acc += Pmn[m, n] * Prs[r_, s_] * d[m, s_, r_, n, x, y]
                    end
                    C2[x, y] += acc
                end
            end
        end
    end

    return 0.5 .* C1 .- 0.25 .* C2
end
