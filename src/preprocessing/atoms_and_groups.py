import typing as tp


def enumerate_hydrides() -> tp.Set[str]:
    """
    Enumerates possible fragments of non-metal hydrides in a molecule.
    Enumerated fragments are likely a bit redundant, but it shouldn't be a problem.
    :return: Set of strings - molecular fragments
    """
    hydrides = set()
    for i in ("", "2", "3", "4"):
        for a in ("Si", "Sn", "As", "S", "Sb", "Te", "Se", "Ge", "P", "Al", "I", "Bi", "Ta"):
            hydrides.add(a + "H" + i)
    return hydrides


_isotopes = {"131I"}
_special_elements = {"Li", "Be", "B",
                     "Na", "Mg", "Al", "Si", "Cl",
                     "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
                     "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
                     "Cs", "Ba", "La", "Ce", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
                     "Fr", "Ra", "Th", "U"} | _isotopes

SPECIAL_ELEMENTS_AND_GROUPS = _special_elements | enumerate_hydrides()

NONMETALS = ["Si", "P", "S", "Cl", "Ge", "As", "Se", "Br", "Sn", "Sb", "Te", "I", "C", "N", "O", "F"]
