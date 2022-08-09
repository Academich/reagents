import sys
import re
from typing import Callable, List, Set
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial

from pandas import Series, concat
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import RDConfig

from IPython.display import SVG

from src.preprocessing.atoms_and_groups import SPECIAL_ELEMENTS_AND_GROUPS

# Import reaction role assignment functionality from the rdkit Contrib directory.
# Don't mind the red underlining in the IDE, it works.
sys.path.append(RDConfig.RDContribDir)
from RxnRoleAssignment import identifyReactants

AAM_TEMPLATE = re.compile('\[[a-zA-Z0-9]+:[0-9]+]')
RESIDUAL_AAM = re.compile(':[0-9]+')

SPECIAL_SYMBOLS = {"@", "@@"}
SPECIAL_SYMBOLS |= {"+{}:".format(i) for i in [""] + list(range(9))}
SPECIAL_SYMBOLS |= {"-{}:".format(i) for i in [""] + list(range(9))}


# === Reaction role assignment


def assign_reaction_roles_schneider(smi: str) -> str:
    try:
        reassigned = identifyReactants.reassignReactionRoles(smi)
    except RuntimeError:
        return assign_reaction_roles_by_aam(smi)

    if reassigned.startswith(">"):
        reassigned = assign_reaction_roles_by_aam(smi)
    return reassigned


def assign_reaction_roles_by_aam(smi: str) -> str:
    """
    Molecules that appear in both sides of the reaction are reagents.
    Aside from that, all molecules that have atom map numbers that appear on the right side are reactants.
    :param smi:
    :return:
    """

    def match_found(mol: str, tgt_labels: List[str]) -> bool:
        aam_labels = pattern.findall(mol)
        for a in aam_labels:
            if a in tgt_labels:
                return True
        return False

    pattern = re.compile(":(\d+)\]")  # atom map numbers
    reactants, reagents = [], []
    left, center, right = smi.split(">")

    all_rs = left.split(".") + center.split(".")
    right_mols_set = set(right.split("."))
    for m in all_rs:
        if m in right_mols_set:
            reagents.append(m)
    all_rs = [m for m in all_rs if m not in reagents]

    tgt_aam_labels = pattern.findall(right)
    for m in all_rs:
        if match_found(m, tgt_aam_labels):
            reactants.append(m)
        else:
            reagents.append(m)

    return ">".join(
        (".".join(reactants),
         ".".join(reagents),
         right)
    )


# === Atom mapping ===

def _extract_atom_from_aam(term: re.Match) -> str:
    """
    Extracts group without AAM from an AAM term (substring)
    :param term: Matched substring which is an atom-mapped group
    :return: A group without atom-mapping
    """
    labeled_atom_group = term.group(0)
    atom_group = labeled_atom_group[1:labeled_atom_group.index(':')]
    if atom_group in SPECIAL_ELEMENTS_AND_GROUPS:
        atom_group = "[" + atom_group + "]"
        return atom_group
    if 'H' in atom_group and 'Hg' not in atom_group:
        atom_group = atom_group[:atom_group.index('H')]
    return atom_group


def _clear_atom_mapping_in_molecule_by_rdkit(smi: str) -> str:
    """
    Drops atom-mapping in a molecule by resorting to RDKit.
    Pretty slow and not applicable for the transformation of large data sets.
    :param smi: Molecule SMILES with atom-mapping.
    :return: Molecule SMILES without atom-mapping.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol)


def _clear_atom_mapping_in_molecule_by_re(smi: str) -> str:
    """
    Drops atom-mapping in a molecule.
    Does this by pattern-matching and subgroup extraction.
    Is intended to be pretty fast by avoiding RDKit methods.
    :param smi: Molecule SMILES with atom-mapping.
    :return: Molecule SMILES without atom-mapping.
    """
    res = re.sub(AAM_TEMPLATE, _extract_atom_from_aam, smi)
    if any((sym in res for sym in SPECIAL_SYMBOLS)):
        res = re.sub(RESIDUAL_AAM, '', res)
    return res


def drop_atom_mapping(smi: str) -> str:
    """
    Drops atom-mapping in an atom-mapped reaction.
    :return: Reaction SMILES without atom-mapping.
    """
    left, center, right = smi.split(">")

    left = ".".join([_clear_atom_mapping_in_molecule_by_re(s) for s in left.split(".")])
    center = ".".join([_clear_atom_mapping_in_molecule_by_re(s) for s in center.split(".")])
    right = ".".join([_clear_atom_mapping_in_molecule_by_re(s) for s in right.split(".")])
    return left + ">" + center + ">" + right


# === SMILES Canonicalization

def canonicalize_smiles(m: str) -> str:
    mol = Chem.MolFromSmiles(m)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    else:
        return ''


def canonicalize_reaction(smi: str) -> str:
    left, center, right = smi.split(">")
    left = ".".join([Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in left.split(".")])
    center = ".".join([Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in center.split(".")])
    right = ".".join([Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in right.split(".")])
    return left + ">" + center + ">" + right


# === SMILES trimming and reordering
def _extract_isotope(term: re.Match) -> str:
    """
    Extracts group without AAM from an AAM term (substring)
    :param term: Matched substring which is an atom-mapped group
    :return: A group without atom-mapping
    """
    labeled_atom_group = term.group(0)
    isotope_group = labeled_atom_group.strip("[]")
    for i, c in enumerate(isotope_group):
        if c.isalpha():
            return Chem.MolToSmiles(Chem.MolFromSmiles("[{}]".format(isotope_group[i:])))


def drop_isotopes(smi: str) -> str:
    isotope_template = re.compile('\[[0-9]+[a-zA-Z0-9]+]')
    res = re.sub(isotope_template, _extract_isotope, smi)
    return res


def mix_reagents(smi: str) -> str:
    """
    Mixes reagents together with reactants in a reaction.
    :return: Reaction SMILES with mixed reagents and reactants.
    """
    left, center, right = smi.split(">")
    return left + "." * bool(center) + center + ">>" + right


def separate_molecule_tokens(lst: List[int], sep: int) -> List[int]:
    """
    Separates numbers in a list of numbers with a specified number
    :param lst:
    :param sep:
    :return:
    """
    if len(lst) < 2:
        return lst
    out = []
    _lst = iter(lst)
    for i in range(len(lst) * 2 - 1):
        if i % 2 == 0:
            out.append(next(_lst))
        else:
            out.append(sep)
    return out


def drop_cxsmiles_info(smi: str) -> str:
    return smi.split("|")[0]


def keep_only_unique_molecules(smi: str) -> str:
    """
    Removes duplicates of molecules in every part of a reaction
    :param smi:
    :return:
    """

    def process_sequence(s: str) -> str:
        unique_mols = set()
        res = []
        for m in s.split("."):
            if m not in unique_mols:
                unique_mols.add(m)
                res.append(m)
        return ".".join(res)

    left, center, right = smi.split(">")
    left = process_sequence(left)
    center = process_sequence(center)
    right = process_sequence(right)

    return ">".join((left, center, right))


def order_molecules(smi: str) -> str:
    """
    Orders a set of molecules written as SMILES separated by dots.
    The longest molecules come first. In the same length they are ordered in alphabetical order.
    """
    left, center, right = smi.split(">")
    left = ".".join(sorted(left.split("."), key=lambda x: (len(x), x), reverse=True))
    center = ".".join(sorted(center.split("."), key=lambda x: (len(x), x), reverse=True))
    right = ".".join(sorted(right.split("."), key=lambda x: (len(x), x), reverse=True))
    return left + ">" + center + ">" + right


# === Chemical curation

def fix_charcoal(smi: str) -> str:
    left, center, right = smi.split(">")
    not_left = center + ">" + right
    if ".C>" in not_left and ("[Pd]" in center or "[H][H]" in center):
        return left + ">" + not_left.replace(".C>", ".[C]>")
    return smi


class IonAssembler:
    """
    Assembles broken ions of Al and B.
    Like [Al+3].[H-].[H-].[H-].[H-] becomes [AlH4-]
    Assembles hydrides, chlorides and bromides of Al and B.
    """

    @classmethod
    def _get_charge(cls, mol: str, at: str) -> int:
        charge = mol.split(f"{at}+")[1][0]
        if charge.isdigit():
            return int(charge)
        elif charge == "]":
            return 1
        else:
            raise ValueError("Cannot get charge")

    @classmethod
    def _signed_number(cls, n: int) -> str:
        if n == 1:
            return "+"
        elif n == 0:
            return ""
        elif n == -1:
            return "-"
        elif n > 0:
            return f"+{n}"
        else:
            return f"-{n}"

    @classmethod
    def _balance_ion(cls, smi: str, atom: str) -> str:
        if f"[{atom}+" in smi:
            mols = smi.split(".")
            result = []
            particles = Counter(mols)
            met_prt = []
            for p, n in particles.items():
                if f"[{atom}+" in p:
                    met_prt += [p] * n
            met_prt.sort(key=lambda x: cls._get_charge(x, atom), reverse=True)
            particles = {p: n for p, n in particles.items() if p not in met_prt}

            total_charge_to_balance = sum((cls._get_charge(i, atom) for i in met_prt))
            total_charge_to_balance += sum((cls._get_charge(i, "Zn") for i in mols if "[Zn+" in i))
            total_charge_to_balance += sum((cls._get_charge(i, "Ti") for i in mols if "[Ti+" in i))
            total_charge_to_balance += sum((cls._get_charge(i, "Ca") for i in mols if "[Ca+" in i))
            total_balancers = particles.get("[Cl-]", 0)
            if total_balancers == 0:
                total_balancers += particles.get("[Br-]", 0)
            if total_balancers == 0:
                total_balancers += particles.get("[H-]", 0)
            split_even = total_balancers % total_charge_to_balance == 0 and (
                    len(met_prt) + int("[Zn+" in smi) + int("[Ti+" in smi) + int("[Ca+" in smi)) > 1

            for p in met_prt:
                charge = cls._get_charge(p, atom)

                if bool(particles.get("[Cl-]")):
                    n_balance = min(charge + 1, particles["[Cl-]"])

                    if charge < 3 or n_balance < 3:
                        result.append(p)
                        continue
                    if split_even:
                        n_balance -= 1
                        split_even = False

                    balanced = Chem.CanonSmiles(
                        p.replace(f"[{atom}+" + str(charge) * (charge > 1) + "]",
                                  f"[{atom}" + cls._signed_number(charge - n_balance) + "]" + "(Cl)" * (
                                          n_balance - 1) + "Cl")
                    )
                    result.append(balanced)
                    particles["[Cl-]"] -= n_balance
                    continue

                if bool(particles.get("[Br-]")):
                    n_balance = min(charge + 1, particles["[Br-]"])

                    if charge < 3 or n_balance < 3:
                        result.append(p)
                        continue

                    balanced = Chem.CanonSmiles(
                        p.replace(f"[{atom}+" + str(charge) * (charge > 1) + "]",
                                  f"[{atom}" + cls._signed_number(charge - n_balance) + "]" + "(Br)" * (
                                          n_balance - 1) + "Br")
                    )
                    result.append(balanced)
                    particles["[Br-]"] -= n_balance
                    continue

                if bool(particles.get("[H-]")):
                    n_balance = min(charge + 1, particles["[H-]"])
                    if split_even:
                        n_balance -= 1
                        split_even = False
                    if charge == 3 and n_balance == 1:
                        result.append(p)
                        continue
                    balanced = p.replace(f"[{atom}+" + str(charge) * (charge > 1) + "]",
                                         f"[{atom}H{n_balance}".replace("1", "") + cls._signed_number(
                                             charge - n_balance) + "]")
                    result.append(balanced)
                    particles["[H-]"] -= n_balance
                    continue

                result.append(p)

            leftovers = [[p] * n for p, n in particles.items()]
            leftovers = sum(leftovers, [])
            return ".".join(result + leftovers)
        else:
            return smi

    @classmethod
    def run(cls, smi: str) -> str:
        # Assemble aluminium and boron halogenides and hydrides
        res = cls._balance_ion(smi, "Al")
        res = cls._balance_ion(res, "B")
        return res


def assemble_ions(smi: str) -> str:
    left, center, right = smi.split(">")
    try:
        _center = IonAssembler.run(center)
    except:
        print(center)
        raise
    return left + ">" + _center + ">" + right


def separate_solvents(solvents_set: Set[str], smi: str) -> str:
    mols = smi.split('.')
    agents, slvs = [], []
    for m in mols:
        if m in solvents_set:
            slvs.append(m)
        else:
            agents.append(m)
    return ".".join(agents) + "&" + ".".join(slvs)


# === Reagent statistics

def get_reagent_statistics(reactions_left_side: Series, chunk_size: int = 1000):
    """
    Obtain frequency of the molecular occurrence among the reactants/reagents of all reactions
    in the form of a Counter. Uses process pool for faster processing.
    :param: chunk_size: size of a subset of the data to process at once
    :returns: One counter with reagent occurrences for all reactions in the specified pandas Series.
    """
    n_entries = reactions_left_side.shape[0]
    global sum_counters

    _individual_counters = reactions_left_side.apply(lambda smi: Counter(smi.split(".")))

    def sum_counters(i):
        return np.sum(_individual_counters[i: i + chunk_size])

    with Pool(cpu_count()) as p:
        bigger_counters = p.map(sum_counters, range(0, n_entries, chunk_size))

    return np.sum(bigger_counters)


# === Visualization ===

def draw_reaction_smarts(rxn_smiles: str, use_smiles: bool = True, highlight: bool = True) -> 'SVG':
    """
    Draws a reaction from a reaction smiles string
    """
    _rxn = rxn_smiles.split("|")[0]
    rxn = AllChem.ReactionFromSmarts(_rxn, useSmiles=use_smiles)
    d = Draw.MolDraw2DSVG(900, 300)
    if highlight:
        colors = [(0.3, 0.7, 0.9), (0.9, 0.7, 0.9), (0.6, 0.9, 0.3), (0.9, 0.9, 0.1)]
        d.DrawReaction(rxn, highlightByReactant=True, highlightColorsReactants=colors)
    else:
        d.DrawReaction(rxn, highlightByReactant=False)
    d.FinishDrawing()

    svg = d.GetDrawingText()
    return SVG(svg.replace('svg:', '').replace(':svg', ''))


# === Tools for faster data processing on CPU using pool of processes ===

def __parallelize(d: Series, func: Callable, num_of_processes: int) -> Series:
    data_split = np.array_split(d, num_of_processes)
    pool = Pool(num_of_processes)
    d = concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return d


def run_on_subset(func: Callable, use_tqdm, data_subset):
    if use_tqdm:
        return data_subset.progress_apply(func)
    return data_subset.apply(func)


def parallelize_on_rows(d: Series, func, num_of_processes: int, use_tqdm=False) -> Series:
    return __parallelize(d, partial(run_on_subset, func, use_tqdm), num_of_processes)


if __name__ == '__main__':
    problems = ["C1=CC=CC=C1.CC[Al+2].CC[Al+2].[Cl-].[Cl-].[Cl-].[Cl-].O",
                "CC(C[Al+]CC(C)C)C.CC(C)C[Al+]CC(C)C.O=S(=O)(O)O.C(CCC)CC.C1CCOC1.CCCCCC.CCOCC.[H-].[H-]",
                "C1=CC([Ti+2]C2C=CC=C2)C=C1.C1CN2CCN1CC2.CC[Al+2].[Cl-].[Cl-].[Cl-].[Cl-]",
                "CC(C)C[Al+]CC(C)C.CC(C)C[Al+]CC(C)C.CC1=CC=CC=C1.CC1=CC=CC=C1.[H-].[H-].CO.O",
                "ClC(Cl)(Cl)Cl.ClC(Cl)(Cl)Cl.[Al+3].[Al+3].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-].Cl",
                "CC(C)C[Al+]CC(C)C.CCCC[Al+]CCCC.CC1=CC=CC=C1.CCOC(C)=O.[H-].[H-].O",
                "[Zn+2].[Al+3].ClCCCl.[Cl-].[Cl-].[Cl-].[Cl-].[Cl-]"]
    s1 = "[Al+].[H-].[Cl-]"
    res = IonAssembler.run("ClC(Cl)(Cl)Cl.ClC(Cl)(Cl)Cl.[Al+3].[Al+3].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-].[Cl-].Cl")
    print(res)
