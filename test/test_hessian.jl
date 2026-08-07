using GaussianBasis
using Molecules
using TensorOperations

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

    @testset "CPHF response-Fock matrix-vector product" begin
        # Validates cphf_Amatvec (the CPHF coupling matrix's action A*U) against
        # a brute-force dense MO-ERI contraction of sum_bj[4(ai|bj)-(ab|ij)-(aj|ib)]U_bj,
        # a structurally independent computation (explicit AO->MO transform of the
        # full ERI tensor, not the build_fock!-reuse trick cphf_Amatvec relies on).
        path = joinpath(@__DIR__, "xyz/water.xyz")
        mol = open(f->read(f,String), path)

        Fermi.Options.set("molstring", mol)
        Fermi.Options.set("basis", "sto-3g")

        wf = @energy rhf
        ndocc = wf.ndocc
        nbas = size(wf.orbitals.C, 1)
        nvir = nbas - ndocc
        C = wf.orbitals.C
        Co = C[:, 1:ndocc]
        Cv = C[:, ndocc+1:end]

        ints = Fermi.Integrals.IntegralHelper(molecule=wf.molecule, eri_type=Fermi.Integrals.Chonky())
        eri_ao = ints["ERI"]

        function mo_eri(eri_ao, Ca, Cb, Cc, Cd)
            @tensoropt eri_mo[a,b,c,d] := Ca[m,a]*Cb[n,b]*Cc[r,c]*Cd[s,d]*eri_ao[m,n,r,s]
            return eri_mo
        end

        eri_aibj = mo_eri(eri_ao, Cv, Co, Cv, Co)
        eri_abij = mo_eri(eri_ao, Cv, Cv, Co, Co)
        eri_ajib = mo_eri(eri_ao, Cv, Co, Co, Cv)

        # deterministic pseudo-random trial matrix (avoids a Random stdlib
        # test-dependency for what only needs to be "generic, not special")
        U = [sin(1.3*a + 2.7*i) for a in 1:nvir, i in 1:ndocc]

        AU_brute = zeros(nvir, ndocc)
        for a in 1:nvir, i in 1:ndocc
            s = 0.0
            for b in 1:nvir, j in 1:ndocc
                s += (4*eri_aibj[a,i,b,j] - eri_abij[a,b,i,j] - eri_ajib[a,j,i,b]) * U[b,j]
            end
            AU_brute[a,i] = s
        end

        AU_fast = Fermi.HartreeFock.cphf_Amatvec(U, Co, Cv, ints)
        @test AU_fast ≈ AU_brute atol=1e-10
    end

    @testset "CPHF orbital response (finite difference of the SCF density)" begin
        # The definitive CPHF check: does the analytic dD/dy built from the
        # CPHF-solved U actually match how the *real*, re-converged SCF
        # density changes under a geometry displacement? This is independent
        # of every other test here -- it reruns full SCF at displaced
        # geometries rather than reusing any of cphf_rhs/cphf_Amatvec's own
        # machinery.
        path = joinpath(@__DIR__, "xyz/water.xyz")
        mol = open(f->read(f,String), path)
        basis = "sto-3g"

        Fermi.Options.set("molstring", mol)
        Fermi.Options.set("basis", basis)
        wf = @energy rhf

        ndocc = wf.ndocc
        C = wf.orbitals.C
        Co = C[:, 1:ndocc]
        Cv = C[:, ndocc+1:end]
        bset = GaussianBasis.BasisSet(wf.orbitals.basis, wf.molecule.atoms)
        ints = Fermi.Integrals.IntegralHelper(molecule=wf.molecule, eri_type=Fermi.Integrals.Chonky())
        atoms = wf.molecule.atoms

        function displaced_molstring(atoms, iA, k, delta)
            newatoms = deepcopy(atoms)
            xyz = collect(newatoms[iA].xyz)
            xyz[k] += delta
            newatoms[iA] = Molecules.Atom(newatoms[iA].Z, newatoms[iA].mass, xyz)
            return join(["$(Int(a.Z))   $(a.xyz[1])   $(a.xyz[2])   $(a.xyz[3])" for a in newatoms], "\n")
        end

        function scf_D(molstring)
            Fermi.Options.set("molstring", molstring)
            Fermi.Options.set("basis", basis)
            wf_ = @energy rhf
            Co_ = wf_.orbitals.C[:, 1:wf_.ndocc]
            return Co_ * Co_'
        end

        B2A = Molecules.bohr_to_angstrom
        h = 5e-4

        for iA in 1:length(atoms)
            U = Fermi.HartreeFock.cphf_solve(wf, ints, iA)

            ∂S = zeros(size(Co, 1), size(Co, 1), 3)
            GaussianBasis.∇overlap!(∂S, bset, iA)

            for q in 1:3
                Dp = scf_D(displaced_molstring(atoms, iA, q, h))
                Dm = scf_D(displaced_molstring(atoms, iA, q, -h))
                dD_FD = (Dp .- Dm) ./ (2h / B2A)

                ΔD = Cv * U[:, :, q] * Co'
                ΔD .+= ΔD'
                S_oo = Co' * (@view ∂S[:, :, q]) * Co
                dD_analytic = ΔD .- Co * S_oo * Co'

                @test dD_analytic ≈ dD_FD atol=1e-5
            end
        end

        Fermi.Options.set("molstring", mol)
        Fermi.Options.set("basis", basis)
    end

    @testset "Full RHF Hessian (finite difference of the gradient, and Psi4)" begin
        Fermi.Options.set("molstring", """
        O   0.000000000000   0.000000000000   0.000000000000
        H   0.758602190000   0.000000000000   0.504284980000
        H   0.758602190000   0.000000000000  -0.504284980000
        """)
        basis = "sto-3g"
        Fermi.Options.set("basis", basis)
        wf = @energy rhf
        H = Fermi.HartreeFock.RHFhess(wf)

        @test H ≈ H' atol=1e-10

        # Reference from Psi4 1.10 (`hessian('scf')`, basis sto-3g, this exact
        # geometry, noreorient/nocom so the atom/axis ordering matches).
        H_psi4 = [
            1.13060065634761   0.0                0.0               -0.56530032817424   0.0               -0.44471126604731  -0.56530032817424   0.0                0.44471126604731
            0.0               -0.07707295928527   0.0                0.0                0.03853647964198   0.0                0.0                0.03853647964198   0.0
            0.0                0.0                0.61840092098178  -0.52310501956430   0.0               -0.30920046049119   0.52310501956430   0.0               -0.30920046049119
           -0.56530032817424   0.0               -0.52310501956430   0.56393084645098   0.0                0.48390814280581   0.00136948172326   0.0                0.03919687675849
            0.0                0.03853647964198   0.0                0.0               -0.13144426267318   0.0                0.0                0.09290778303120   0.0
           -0.44471126604731   0.0               -0.30920046049119   0.48390814280581   0.0                0.53179651540804  -0.03919687675849   0.0               -0.22259605491685
           -0.56530032817424   0.0                0.52310501956430   0.00136948172326   0.0               -0.03919687675849   0.56393084645098   0.0               -0.48390814280581
            0.0                0.03853647964198   0.0                0.0                0.09290778303120   0.0                0.0               -0.13144426267318   0.0
            0.44471126604731   0.0               -0.30920046049119   0.03919687675849   0.0               -0.22259605491685  -0.48390814280581   0.0                0.53179651540804
        ]
        @test H ≈ H_psi4 atol=1e-6

        # Independent check: finite difference of Fermi's own analytic gradient.
        function displaced_molstring(atoms, iA, k, delta)
            newatoms = deepcopy(atoms)
            xyz = collect(newatoms[iA].xyz)
            xyz[k] += delta
            newatoms[iA] = Molecules.Atom(newatoms[iA].Z, newatoms[iA].mass, xyz)
            return join(["$(Int(a.Z))   $(a.xyz[1])   $(a.xyz[2])   $(a.xyz[3])" for a in newatoms], "\n")
        end

        function scf_grad(molstring)
            Fermi.Options.set("molstring", molstring)
            Fermi.Options.set("basis", basis)
            wf_ = @energy rhf
            return Fermi.HartreeFock.RHFgrad(wf_)
        end

        atoms = wf.molecule.atoms
        natm = length(atoms)
        B2A = Molecules.bohr_to_angstrom
        h = 5e-4
        H_FD = zeros(3*natm, 3*natm)
        for iB in 1:natm, qB in 1:3
            gp = scf_grad(displaced_molstring(atoms, iB, qB, h))
            gm = scf_grad(displaced_molstring(atoms, iB, qB, -h))
            H_FD[:, 3*(iB-1)+qB] .= vec(((gp .- gm) ./ (2h/B2A))')
        end
        @test H ≈ H_FD atol=1e-5
    end

    @testset "@hessian macro and vibrational_analysis" begin
        Fermi.Options.set("molstring", """
        O   0.000000000000   0.000000000000   0.000000000000
        H   0.758602190000   0.000000000000   0.504284980000
        H   0.758602190000   0.000000000000  -0.504284980000
        """)
        Fermi.Options.set("basis", "sto-3g")
        wf = @energy rhf
        H = @hessian wf => rhf

        freqs, modes = vibrational_analysis(H, wf.molecule)

        # 6 translation/rotation modes (nonlinear triatomic) should be
        # essentially zero; the remaining 3 are the genuine vibrations.
        @test count(f -> abs(f) < 1.0, freqs) == 6

        vib_freqs = sort(filter(f -> abs(f) >= 1.0, freqs))
        # Cross-validated independently: matches an from-scratch numpy
        # reimplementation of mass-weighting+projection+diagonalization
        # applied to Psi4's own (already Hessian-matrix-validated) result,
        # to ~0.5 cm^-1 (consistent with the ~1e-8 Fermi/Psi4 Hessian gap).
        @test vib_freqs ≈ [2400.9613, 5156.3764, 5540.6078] atol=1.0

        # Linear molecule sanity check: only 5 (not 6) trans/rot zero modes,
        # since rotation about the molecular axis isn't a physical DOF.
        Fermi.Options.set("molstring", """
        N   0.000000000000   0.000000000000   0.000000000000
        N   0.000000000000   0.000000000000   1.100000000000
        """)
        wf_n2 = @energy rhf
        H_n2 = @hessian wf_n2 => rhf
        freqs_n2, _ = vibrational_analysis(H_n2, wf_n2.molecule)
        @test count(f -> abs(f) < 1.0, freqs_n2) == 5
    end
end

@reset
