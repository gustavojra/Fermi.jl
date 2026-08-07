using GaussianBasis
using Molecules

@reset
@set printstyle none

# Validates the two-electron (Coulomb+exchange) contribution to the RHF Hessian,
# ERI_hess_JK, against finite difference of the *same* 0.5*P*P*(ERI - 0.5*ERI_exch)
# contraction RHFgrad.jl already uses at first order -- i.e. this checks the
# "integral response" piece only (P held fixed), not the full CPHF-coupled
# Hessian, which doesn't exist yet.
@testset "Hessian" begin
    @testset "Two-electron JK contraction (finite difference)" begin
        path = joinpath(@__DIR__, "xyz/water.xyz")
        mol = open(f->read(f,String), path)

        Fermi.Options.set("molstring", mol)
        Fermi.Options.set("basis", "sto-3g")

        wf = @energy rhf
        bset = GaussianBasis.BasisSet(wf.orbitals.basis, wf.molecule.atoms)
        Co = wf.orbitals.C[:, 1:wf.ndocc]
        P = 2.0 .* Co * Co'

        function E2_coulomb(bs, P)
            eri = GaussianBasis.ERI_2e4c(bs)
            return sum(P[m,n]*P[r,s]*eri[m,n,r,s] for m in 1:bs.nbas, n in 1:bs.nbas, r in 1:bs.nbas, s in 1:bs.nbas)
        end

        function E2_exchange(bs, P)
            eri = GaussianBasis.ERI_2e4c(bs)
            return sum(P[m,n]*P[r,s]*eri[m,s,r,n] for m in 1:bs.nbas, n in 1:bs.nbas, r in 1:bs.nbas, s in 1:bs.nbas)
        end

        function FD2(f, bs, iA, iB, h=1e-4)
            out = zeros(3,3)
            B2A = Molecules.bohr_to_angstrom
            for kA in 1:3, kB in 1:3
                bAp, bAm = GaussianBasis.create_displacement(bs, iA, kA, h)
                bApBp, _ = GaussianBasis.create_displacement(bAp, iB, kB, h)
                _, bApBm = GaussianBasis.create_displacement(bAp, iB, kB, h)
                bAmBp, _ = GaussianBasis.create_displacement(bAm, iB, kB, h)
                _, bAmBm = GaussianBasis.create_displacement(bAm, iB, kB, h)
                out[kA,kB] = (f(bApBp) - f(bApBm) - f(bAmBp) + f(bAmBm)) / (4*(h/B2A)^2)
            end
            return out
        end

        natoms = length(wf.molecule.atoms)
        for iA in 1:natoms, iB in iA:natoms
            H2e_analytic = Fermi.HartreeFock.ERI_hess_JK(bset, P, iA, iB)
            H2e_FD = 0.5 .* FD2(bs_->E2_coulomb(bs_,P), bset, iA, iB) .-
                     0.25 .* FD2(bs_->E2_exchange(bs_,P), bset, iA, iB)
            @test H2e_analytic ≈ H2e_FD atol=1e-4
        end
    end
end

@reset
