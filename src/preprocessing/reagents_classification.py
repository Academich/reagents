from string import digits
from typing import Tuple, List
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Fragments

SOLVENTS = [
    "CC#N"
    "C[N+](=O)[O-]",
    "Cc1ccccc1",
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc(cc1)Cl",
    "c1ccc(c(c1)Cl)Cl",
    "ClCCl",
    "ClC(Cl)C",
    "ClCCCl",
    "ClC(Cl)Cl",
    "ClC(Cl)(Cl)Cl",
    "C1CCOC1",
    "O1C(C)CCC1",
    "O1CCOCC1",
    "CCCCC",
    "CCCCCC",
    "CCCCCCC",
    "C1CCCCC1",
    "C1CCCC1",
    "CO",
    "CCO",
    "CCCO",
    "CCCCO",
    "CC(O)C",
    "CC(C)(C)O",
    "CC(C)CO",
    "CC(=O)C",
    "O=C(C)CC",
    "CN(C)C=O",
    "CN1CCCC1=O",
    "CC(=O)NC",
    "CC(=O)N(C)C",
    "CS(C)=O",
    "O=P(N(C)C)(N(C)C)N(C)C",
    "OCCO",
    "CC(O)=O",
    "O=C(O)C(F)(F)F",
    "CCOCC",
    "COCCOC",
    "O(C(C)(C)C)C",
    "O=C(OCC)C",
    "S=C=S"
]

CATALYTIC_METALS = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
                    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "La", "Ce", "Hf", "Ta",
                    "W",
                    "Re", "Os", "Ir", "Pt", "Au", "Hg",
                    "Tl", "Pb", "Bi"}


class HeuristicRoleClassifier:
    """
    Splits a set of reagents into reagent roles using heuristics.
    Roles are the following (ordered by decreasing priority)
    1. Catalysts
    2. Oxidizing agents
    3. Reducing agents
    4. Acids
    5. Bases
    6. Unspecified
    7. Solvents
    """
    solvents = SOLVENTS
    cat_metals = CATALYTIC_METALS
    types = ["Catalyst", "Ox", "Red", "Acid", "Base", "Unspecified", "Solvent"]

    @classmethod
    def classify(cls, reagents_smi: str) -> Tuple[List[str], ...]:
        mols = reagents_smi.split('.')

        catal = []
        ox = []
        red = []
        acid = []
        base = []
        solvent = []
        unspec = []

        for m in mols:
            if cls.is_solvent(m):
                solvent.append(m)
                continue

            graph = Chem.MolFromSmiles(m)
            if cls.is_catalyst(m):
                catal.append(m)
                continue
            if cls.is_oxidizing_agent(m):
                ox.append(m)
                continue
            if graph is not None and cls.is_reducing_agent(m, graph):
                red.append(m)
                continue
            if graph is not None and cls.is_acid(m, graph):
                acid.append(m)
                continue
            if graph is not None and cls.is_base(m, graph):
                base.append(m)
                continue
            unspec.append(m)

        return catal, ox, red, acid, base, unspec, solvent

    @classmethod
    def classify_to_str(cls, reagents_smi: str) -> str:
        catal, ox, red, acid, base, solvent, unspec = cls.classify(reagents_smi)
        catal_smi = ".".join(sorted(catal, key=lambda x: len(x), reverse=True))
        ox_smi = ".".join(sorted(ox, key=lambda x: len(x), reverse=True))
        red_smi = ".".join(sorted(red, key=lambda x: len(x), reverse=True))
        acid_smi = ".".join(sorted(acid, key=lambda x: len(x), reverse=True))
        base_smi = ".".join(sorted(base, key=lambda x: len(x), reverse=True))
        unspec_smi = ".".join(sorted(unspec, key=lambda x: len(x), reverse=True))
        return "&".join((catal_smi, ox_smi, red_smi, acid_smi, base_smi, unspec_smi))

    @classmethod
    def role_voting(cls, regents_smi: str):
        catal, ox, red, acid, base, solvent, unspec = cls.classify(regents_smi)
        best_catals = '.'.join(
            [i[0] for i in (Counter(catal) + Counter(acid) + Counter(base)).most_common(2)]
        )
        best_solvents = '.'.join(
            [i[0] for i in Counter(solvent).most_common(2)]
        )
        best_redox = '.'.join(
            [i[0] for i in (Counter(ox) + Counter(red)).most_common(2)]
        )
        best_unspec = '.'.join(
            [i[0] for i in Counter(unspec).most_common(2)]
        )
        return best_catals, best_redox, best_unspec, best_solvents

    @classmethod
    def is_solvent(cls, smi: str) -> bool:
        return smi in cls.solvents

    @classmethod
    def is_catalyst(cls, smi: str) -> bool:
        # Single metals are catalysts
        # Molecules that contain transition metals and cycles are catalysts
        # Charcoal is a catalyst
        # Triphenylphosphin also counts here

        contains_cycle = smi.count("1") > 1
        contains_phosphorus = "P" in smi or "p" in smi
        stripped_smi = smi.strip("[]" + digits + "+-")
        is_metal = any([f"[{i}]" == smi for i in cls.cat_metals]) or stripped_smi in cls.cat_metals
        is_charcoal = smi == "[C]"
        contains_metal = any([i in smi for i in cls.cat_metals])
        is_metal_halogenide = contains_metal and smi.replace("Cl", '').replace("Br", '').replace("I", '').strip(
            "[]") in cls.cat_metals
        if is_metal or is_charcoal:
            return True
        elif contains_cycle and (contains_metal or contains_phosphorus):
            return True
        elif is_metal_halogenide:
            return True
        elif smi == "CCOC(=O)N=NC(=O)OCC":  # DEAD for Mitsunobu reactions
            return True
        else:
            return False

    @staticmethod
    def is_acid(smi: str, graph: 'Chem.Mol') -> bool:
        # Derivatives of sulfuric and phosphoric acids are acids
        # Sulfamides are acids
        # HCl, HBr, HF, HI are acids
        # Some standard carboxylic acids are acids

        sulf_deriv = Chem.MolFromSmiles('S(=O)(=O)')
        sulf_am_deriv = Chem.MolFromSmarts("S(=O)(=O)N")
        phos_deriv = Chem.MolFromSmiles('P(=O)(O)O')

        _graph_with_h = Chem.MolToSmiles(Chem.AddHs(graph.__copy__()))
        is_acid_deriv = (graph.HasSubstructMatch(sulf_deriv) or graph.HasSubstructMatch(
            sulf_am_deriv) or graph.HasSubstructMatch(
            phos_deriv)) and "O-]" not in smi and any([i in _graph_with_h for i in ["O[H]", "[H]O"]])

        is_standard_acid = any([i == smi for i in ('Cl', 'Br', 'I', 'F', '[NH4+]', 'BrB(Br)Br')])
        is_carbon_acid = (Fragments.fr_COO(graph) > 0) and (
                len(set([i for i in smi if i.isalpha()]) - {'C', 'O'}) == 0) and ("-" not in smi)
        return is_acid_deriv or is_standard_acid or is_carbon_acid

    @staticmethod
    def is_base(smi: str, graph: 'Chem.Mol') -> bool:
        # Salts of alcohols and carbon acids are bases
        # Amines are bases
        # Diimides and free nitrogen also end up here, but it should not be a problem
        is_tertiary_or_secondary_amine = ((Fragments.fr_NH1(graph) > 0) or (Fragments.fr_NH0(graph) > 0)) and (
                len(set([i.upper() for i in smi if i.isalpha()]) - {'C', 'N'}) == 0)
        is_salt = "O-]" in smi and (len(set([i for i in smi if i.isalpha()]) - {'C', 'O', 'S', 'P'}) == 0)
        is_lithium_base = "[Li]" in smi and (len(set([i for i in smi.replace("[Li]", "") if i.isalpha()]) - {'C'}) == 0)
        is_hydroxide_ion = smi == "[OH-]"
        is_hydride = "[H-]" in smi
        return is_salt or is_lithium_base or is_tertiary_or_secondary_amine or is_hydroxide_ion or is_hydride

    @classmethod
    def is_oxidizing_agent(cls, smi: str) -> bool:
        # TODO Thinks that all nitrates are ox. agents even if they are not
        # TODO Also a problem with derivatives of Ti4
        # Molecules with transition metals and more than one oxygen are oxidizing agents
        # Molecules with peroxide fragments are oxidizing agents
        # Molecules with '+' and [O-] are oxidizing agents
        # Free halogenes, free oxygen, ozone are oxidizing agents
        # Halogenating agents are oxidizing agents

        is_peroxide = "OO" in smi
        is_nbs = smi == "O=C1CCC(=O)N1Cl"
        is_chlorinator = any([i == smi for i in ('O=S(Cl)Cl', 'O=C(Cl)C(=O)Cl')])
        is_standard = any([i == smi for i in ('ClCl', 'BrBr', 'II', 'FF', 'O=O', 'O=[O+][O-]')])
        is_metal_ox = smi.count("O") >= 2 and any([i in smi for i in ("Os", "Re", "Ru", "Pt", "Pd", "Ir")])
        is_iod_ox = smi.count("O") > 2 and "I" in smi
        return ("+" in smi and "O-]" in smi and smi != "O=[N+]([O-])[O-]"
                ) or is_peroxide or is_standard or is_metal_ox or is_chlorinator or is_iod_ox or is_nbs

    @staticmethod
    def is_reducing_agent(smi: str) -> bool:
        # Some standard molecules are reducing agents
        is_standard = any([j == smi for j in ("[H][H]", "B", "[BH4-]", "NN", "Cl[Sn]Cl", "[S-2]", "O=[PH2]O")])
        is_compound_boron = any([j == smi for j in ("[BH3-]C#N", "CC(=O)O[BH-](OC(C)=O)OC(C)=O")])
        return is_standard or is_compound_boron


if __name__ == '__main__':
    s = "CC(=O)C.[I-].[Na+]"
    print(HeuristicRoleClassifier.classify(s))
