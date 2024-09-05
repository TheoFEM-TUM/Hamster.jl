struct TBBasis
    orbitals :: Vector{Orbital}
    overlaps
    parameters
end

function TBBasis(strc::Structure, conf=get_empty_config())

end