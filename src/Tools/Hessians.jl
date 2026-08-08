using LinearAlgebra
using Molecules

export @hessian, vibrational_analysis

h_dict = Dict{String, String}(
    "rhf" => "Fermi.HartreeFock.RHFhess()"
)

"""
    Fermi.@hessian

Macro to call functions to compute analytic Hessians given current options. Arguments may be
passed using "=>" or "<=", same convention as `@gradient`.

# Examples

Generating a RHF Hessian
```
@hessian rhf
```

Returns the raw Cartesian Hessian (`3*Natoms x 3*Natoms` matrix, atomic units) -- pass it to
`vibrational_analysis` for mass-weighting, projection, and a frequency table.

`@hessian` requires a wave function (`@hessian wfn => rhf`) -- there is no bare
`@hessian rhf` form. Without an `aoints` (`IntegralHelper`), it builds one
internally and prints a warning, since CPHF's Fock builds genuinely need
atomic integrals and this throws away any that were already computed for
`wfn`. To avoid the rebuild -- especially useful right after `@optimize`,
which already has one live at the converged geometry -- pass `aoints` in
directly:
```
aoints = Fermi.Integrals.IntegralHelper()
wfn = @energy aoints => rhf
hess = @hessian aoints, wfn => rhf
```

# Implemented methods:
    Method   Description
    ------   -----------
    RHF      Restricted Hartree-Fock, fully analytic (integrals + CPHF).
"""
macro hessian(comm)
    clean_up(s) = String(filter(c->!occursin(c," :"),s))
    A = clean_up(repr(comm))

    while A[1] == '(' && A[end] == ')'
        A = A[2:end-1]
    end

    if occursin("=>", A)
        arg, method = split(A, "=>")
    elseif occursin("<=", A)
        method, arg = split(A, "<=")
    else
        method = A
        arg = ""
    end

    method = lowercase(String(method))
    arg = String(arg)

    out = ""
    try
        out = replace(h_dict[method], "()" => "("*arg*")")
    catch KeyError
        throw(FermiException("Invalid or unsupported method for Hessian computation: \"$A\""))
    end

    expr_out = Meta.parse(out)

    return esc(expr_out)
end

"""
    Fermi.vibrational_analysis(H::AbstractMatrix, molecule::Molecules.Molecule; project=true)

Mass-weights a Cartesian Hessian (`3*Natoms x 3*Natoms`, atomic units, as returned by
`@hessian`), projects out the translational and rotational zero modes (5 for a linear
molecule, 6 otherwise), diagonalizes, and returns/prints the harmonic vibrational
frequencies (cm⁻¹).

Returns `(freqs, modes)`: `freqs` is a `3*Natoms`-length vector of wavenumbers, sorted
ascending (the leading translation/rotation entries are near-zero; imaginary/unstable
modes are reported as negative wavenumbers, the usual convention), and `modes` is the
corresponding matrix of mass-weighted Cartesian eigenvectors (columns).

Set `project=false` to skip the translation/rotation projection (e.g. for testing).
"""
function vibrational_analysis(H::AbstractMatrix, molecule::Molecule; project=true)
    atoms = molecule.atoms
    natm = length(atoms)
    ndof = 3*natm

    masses_au = [a.mass * Fermi.PhysicalConstants.amu_to_me for a in atoms]
    Mvec = repeat(masses_au, inner=3)
    sqrtM = sqrt.(Mvec)

    Hmw = H ./ (sqrtM * sqrtM')

    if project
        com = sum(a.xyz .* a.mass for a in atoms) ./ sum(a.mass for a in atoms)
        r = [collect(a.xyz .- com) ./ Molecules.bohr_to_angstrom for a in atoms]

        Itensor = zeros(3,3)
        for a in 1:natm
            ra = r[a]
            m = masses_au[a]
            Itensor .+= m .* ((ra⋅ra) .* Matrix{Float64}(I, 3, 3) .- ra*ra')
        end
        Ivals, Ivecs = eigen(Symmetric(Itensor))

        linear = Ivals[1] < 1e-4 * Ivals[3]
        rot_axes = linear ? (2:3) : (1:3)
        nmodes = 3 + length(rot_axes)

        Dtr = zeros(ndof, nmodes)
        for k in 1:3, a in 1:natm
            Dtr[3*(a-1)+k, k] = sqrt(masses_au[a])
        end
        for (col, k) in enumerate(rot_axes)
            n = Ivecs[:, k]
            for a in 1:natm
                v = sqrt(masses_au[a]) .* (n × r[a])
                Dtr[(3*(a-1)+1):(3*a), 3+col] .= v
            end
        end

        Q = Matrix(qr(Dtr).Q)[:, 1:nmodes]  # orthonormal basis for the trans/rot subspace
        Proj = Matrix{Float64}(I, ndof, ndof) .- Q*Q'
        Hmw = Proj * Hmw * Proj
    end

    λ, modes = eigen(Symmetric(Hmw))
    freqs = sign.(λ) .* sqrt.(abs.(λ)) .* Fermi.PhysicalConstants.hartree_to_cm

    output("\n   • Harmonic Vibrational Frequencies")
    output("\n {:>6}   {:>15}", "Mode", "ν̃ (cm⁻¹)")
    for i in eachindex(freqs)
        output(" {:>6}   {:>15.4f}", i, freqs[i])
    end
    output("")

    return freqs, modes
end
