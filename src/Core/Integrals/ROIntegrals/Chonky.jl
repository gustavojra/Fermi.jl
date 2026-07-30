function compute_ERI!(I::IntegralHelper{T, Chonky, O}, aoints::IntegralHelper{T, Chonky, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}
    AOERI = aoints["ERI"]
    C = I.orbitals.C
    if eltype(I.orbitals.C) !== T
        C = T.(C)
    end
    @tensoropt MOERI[i,j,k,l] :=  AOERI[μ, ν, ρ, σ]*C[μ, i]*C[ν, j]*C[ρ, k]*C[σ, l]
    I["ERI"] = MOERI
end

"""
    compute_block!(I, aoints, entry)

Shared implementation for all six occ/virt-blocked MO integral types
("OOOO", "OOOV", "OOVV", "OVOV", "OVVV", "VVVV"). Each character of `entry`
selects whether the corresponding tensor leg is transformed with occupied
(`Co`) or virtual (`Cv`) orbitals — e.g. "OOVV" contracts legs 1-2 against
`Co` and legs 3-4 against `Cv`.
"""
function compute_block!(I::IntegralHelper{T,Chonky,O}, aoints::IntegralHelper{T,Chonky, AtomicOrbitals}, entry::String) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals}

    AOERI = aoints["ERI"]

    inac = Options.get("drop_vir")
    core = Options.get("drop_occ")
    ndocc = I.molecule.Nα
    nbf = size(I.orbitals.C,1)
    o = (1+core):ndocc
    v = (ndocc+1):(nbf - inac)

    Co = I.orbitals.C[:,o]
    Cv = I.orbitals.C[:,v]
    if eltype(I.orbitals.C) !== T
        Co = T.(Co)
        Cv = T.(Cv)
    end

    C1, C2, C3, C4 = (entry[k] == 'O' ? Co : Cv for k = 1:4)

    @tensoropt (μ=>100x, ν=>100x, ρ=>100x, σ=>100x, i=>10x, j=>10x, k=>10x, l=>10x, a=>80x, b=>80x, c=>80x, d=>80) begin
        OUT[i,j,k,l] :=  AOERI[μ, ν, ρ, σ]*C1[μ, i]*C2[ν, j]*C3[ρ, k]*C4[σ, l]
    end
    I[entry] = OUT
end

compute_OOOO!(I::IntegralHelper{T,Chonky,O}, aoints::IntegralHelper{T,Chonky, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals} = compute_block!(I, aoints, "OOOO")
compute_OOOV!(I::IntegralHelper{T,Chonky,O}, aoints::IntegralHelper{T,Chonky, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals} = compute_block!(I, aoints, "OOOV")
compute_OOVV!(I::IntegralHelper{T,Chonky,O}, aoints::IntegralHelper{T,Chonky, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals} = compute_block!(I, aoints, "OOVV")
compute_OVOV!(I::IntegralHelper{T,Chonky,O}, aoints::IntegralHelper{T,Chonky, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals} = compute_block!(I, aoints, "OVOV")
compute_OVVV!(I::IntegralHelper{T,Chonky,O}, aoints::IntegralHelper{T,Chonky, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals} = compute_block!(I, aoints, "OVVV")
compute_VVVV!(I::IntegralHelper{T,Chonky,O}, aoints::IntegralHelper{T,Chonky, AtomicOrbitals}) where {T<:AbstractFloat, O<:AbstractRestrictedOrbitals} = compute_block!(I, aoints, "VVVV")