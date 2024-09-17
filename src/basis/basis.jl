struct TBBasis{Orb, Ov, P}
    orbitals :: Orb
    overlaps :: Ov
    parameters :: P
end

function TBBasis(strc::Structure, conf=get_empty_config())
    orbitals = get_orbitals(strc, conf)
    overlaps = get_overlaps(strc.ions, orbitals, conf)
end