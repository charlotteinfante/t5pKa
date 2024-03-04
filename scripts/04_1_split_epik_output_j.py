'''
All credit for this script goes to:
Mayr F, Wieder M, Wieder O, Langer T. 
Improving Small Molecule pK a Prediction Using Transfer Learning With Graph Neural Networks. 
Front Chem. 2022 May 26;10:866585. doi: 10.3389/fchem.2022.866585. PMID: 35721000; PMCID: PMC9204323.

Find github to script here: 
https://github.com/wiederm/pkasolver-data/blob/main/scripts/04_1_split_epik_output.py
'''

from rdkit import Chem
from rdkit.Chem import PandasTools, PropertyMol
from typing import Tuple
#from pkasolver.data import iterate_over_acids, iterate_over_bases

import argparse
import gzip
import logging
#from molvs import Standardizer
from copy import deepcopy
import pickle

logger = logging.getLogger(__name__)
#s = Standardizer()

PH = 7.4

def create_conjugate(
    mol_initial: Chem.rdchem.Mol,
    idx: int,
    pka: float,
    pH: float = 7.4,
    ignore_danger: bool = False,
    known_pka_values: bool = True,
) -> Chem.rdchem.Mol:

    """Create the conjugated base/acid of the input molecule depending on if the input molecule is the protonated or
    deprotonated molecule in the acid-base reaction. This is inferred from the pka and pH input.
    If the resulting molecule is illegal, e.g. has negative number of protons on a heavy atom, or highly unlikely, 
    e.g. atom charge of +2 or -2, the opposite ionization state is returned instead
    
    Parameters
    ----------
    mol_initial
        molecule from which either a proton is removed or added
    atom_idx
        atom index of ionization center of the acid-base reaction
    pka
        pka value of the acid-base reaction
    pH
        pH of the ionization state of the input molecule
    ignore_danger
        If false, runtime error is raised if conjugate molecule is illegal or highly unlikely.
        If true, opposite conjugate is output, without raising runtime error
    Raises
    ------
    RuntimeError
        is raised if conjugate molecule is illegal or highly unlikely and ignore_danger is set to False
    Returns
    -------
    Chem.rdchem.Mol
        conjugated molecule
    """

    mol = deepcopy(mol_initial)
    mol_changed = Chem.RWMol(mol)
    Chem.SanitizeMol(mol_changed)
    atom = mol_changed.GetAtomWithIdx(idx)
    charge = atom.GetFormalCharge()
    Ex_Hs = atom.GetNumExplicitHs()
    Tot_Hs = atom.GetTotalNumHs()
    danger = False
    # make deprotonated conjugate as pKa > pH with at least one proton or
    # mol charge is positive (otherwise conjugate reaction center would have charge +2 --> highly unlikely)
    if (pka > pH and Tot_Hs > 0) or (charge > 0 and known_pka_values):
        atom.SetFormalCharge(charge - 1)
        if Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs - 1)

    # make protonated conjugate as pKa < pH and charge is neutral or negative
    elif pka <= pH and charge <= 0:
        atom.SetFormalCharge(charge + 1)
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)

    # make protonated conjugate as pKa > pH and there are no proton at the reaction center
    elif pka > pH and Tot_Hs == 0:
        atom.SetFormalCharge(charge + 1)
        danger = True
        if Tot_Hs == 0 or Ex_Hs > 0:
            atom.SetNumExplicitHs(Ex_Hs + 1)

    else:
        raise RuntimeError(
            f"pka: {pka},charge:{charge},Explicit Hs:{Ex_Hs}, Total Hs:{Tot_Hs}, reaction center atomic number: {atom.GetAtomicNum()}"
        )
    atom.UpdatePropertyCache()

    if (
        atom.GetSymbol() == "O"
        and atom.GetFormalCharge() == 1
        and known_pka_values == False
    ):
        raise RuntimeError("Protonating already protonated oxygen. Aborting.")

    Tot_Hs_after = atom.GetTotalNumHs()
    assert Tot_Hs != Tot_Hs_after
    # mol = next(ResonanceMolSupplier(mol))
    if danger and not ignore_danger:
        logger.debug(f"Original mol: {Chem.MolToSmiles(mol)}")
        logger.debug(f"Changed mol: {Chem.MolToSmiles(mol_changed)}")
        logger.debug(
            f"This should only happen for the test set. pka: {pka},charge:{charge},Explicit Hs:{Ex_Hs}, Total Hs:{Tot_Hs}, reaction center atomic number: {atom.GetAtomicNum()}"
        )
        raise RuntimeError("danger")
    return mol_changed

def iterate_over_acids(
    acidic_mols_properties: list,
    nr_of_mols: int,
    partner_mol: Chem.Mol,
    nr_of_skipped_mols: int,
    pka_list: list,
    GLOBAL_COUNTER: int,
    pH: float,
    counter_list: list,
    smiles_list: list,
) -> Tuple[list, int, int, int]:
    """Processes the acidic pKa values of an Schrödinger EPIK pka query
    and returns a pair of protonated and deprotonated molecules for every
    pKa. Takes and updates global counters and skip trackers.
    Parameters
    ----------
    acidic_mols_properties
        list of of dictionaries, each containing pka, ionization index
        and CHEMBL id of an acidic input molecule
    nr_of_mols
        index number of molecule
    partner_mol
        molecule in protonation state at pH=pH
    nr_of_skipped_mols
        global number of skipped molecules
    pka_list
        list of all pKas already found for this molecule
    GLOBAL_COUNTER
        counts total number of pKas processed
    pH
        pH of protonations state of partner_mol
    counter_list: list,
        list of values of GLOBAL_COUNTER
    smiles_list: list,
        list of smiles tuples of protonated and deprotonated molecules of all pka reactions
    Returns
    -------
    acidic_mols (list)
        list of tuples of protonated and deprotonated molecules for
        every pka in acidic_mols_properties that did not yield an error
    nr_of_skipped_mols (int)
        global number of skipped molecules
    GLOBAL_COUNTER (int)
        counts total number of pKas processed
    skipping_acids
        number of acids skipped (max 1)
    """
    acidic_mols = []
    skipping_acids = 0

    for idx, acid_prop in enumerate(
        reversed(acidic_mols_properties)
    ):  # list must be iterated in reverse, in order to protonated the strongest conjugate base first

        if skipping_acids == 0:  # if a acid was skipped, all further acids are skipped
            try:
                new_mol = create_conjugate(
                    partner_mol,
                    acid_prop["atom_idx"],
                    acid_prop["pka_value"],
                    pH=pH,
                )
                Chem.SanitizeMol(new_mol)

            except Exception as e:
                print(f"Error at molecule number {nr_of_mols} - acid enumeration")
                print(e)
                print(acid_prop)
                print(acidic_mols_properties)
                if partner_mol:
                    print(Chem.MolToSmiles(partner_mol))
                skipping_acids += 1
                nr_of_skipped_mols += 1
                continue  # continue instead of break, will not enter this routine gain since skipping_acids != 0

            pka_list.append(acid_prop["pka_value"])
            smiles_list.append(
                (Chem.MolToSmiles(new_mol), Chem.MolToSmiles(partner_mol))
            )

            for mol in [new_mol, partner_mol]:
                GLOBAL_COUNTER += 1
                counter_list.append(GLOBAL_COUNTER)
                mol.SetProp(f"CHEMBL_ID", str(acid_prop["chembl_id"]))
                mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                mol.SetProp(f"pKa", str(acid_prop["pka_value"]))
                mol.SetProp(f"epik_atom", str(acid_prop["atom_idx"]))
                mol.SetProp(f"pKa_number", f"acid_{idx + 1}")
                mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(mol)}")

            # add current mol to list of acidic mol. for next
            # lower pKa value, this mol is starting structure
            acidic_mols.append(
                (
                    PropertyMol.PropertyMol(new_mol),
                    PropertyMol.PropertyMol(partner_mol),
                )
            )
            partner_mol = deepcopy(new_mol)

        else:
            skipping_acids += 1
    return acidic_mols, nr_of_skipped_mols, GLOBAL_COUNTER, skipping_acids


def iterate_over_bases(
    basic_mols_properties: Chem.Mol,
    nr_of_mols: int,
    partner_mol: Chem.Mol,
    nr_of_skipped_mols,
    pka_list: list,
    GLOBAL_COUNTER: int,
    pH: float,
    counter_list: list,
    smiles_list: list,
) -> Tuple[list, int, int, int]:
    """Processes the basic pKa values of an Schrödinger EPIK pka query
    and returns a pair of protonated and deprotonated molecules for every
    pKa. Takes and updates global counters and skip trackers.
    Parameters
    ----------
    basic_mols_properties
        list of of dictionaries, each containing pka, ionization index
        and CHEMBL id of an basic input molecule
    nr_of_mols
        index number of molecule
    partner_mol
        molecule in protonation state at pH=pH
    nr_of_skipped_mols
        global number of skipped molecules
    pka_list
        list of all pKas already found for this molecule
    GLOBAL_COUNTER
        counts total number of pKas processed
    pH
        pH of protonations state of partner_mol
    counter_list: list,
        list of values of GLOBAL_COUNTER
    smiles_list: list,
        list of smiles tuples of protonated and deprotonated molecules of all pka reactions
    Returns
    -------
    basic_mols (list)
        list of tuples of protonated and deprotonated molecules for
        every pka in basic_mols_properties that did not yield an error
    nr_of_skipped_mols (int)
        global number of skipped molecules
    GLOBAL_COUNTER (int)
        counts total number of pKas processed
    skipping_bases
        number of bases skipped (max 1)
    """
    basic_mols = []
    skipping_bases = 0
    for idx, basic_prop in enumerate(basic_mols_properties):
        if skipping_bases == 0:  # if a base was skipped, all further bases are skipped
            try:
                new_mol = create_conjugate(
                    partner_mol,
                    basic_prop["atom_idx"],
                    basic_prop["pka_value"],
                    pH=pH,
                )

                Chem.SanitizeMol(new_mol)

            except Exception as e:
                # in case error occurs new_mol is not in basic list
                print(f"Error at molecule number {nr_of_mols} - bases enumeration")
                print(e)
                print(basic_prop)
                print(basic_mols_properties)
                if partner_mol:
                    print(Chem.MolToSmiles(partner_mol))
                skipping_bases += 1
                nr_of_skipped_mols += 1
                continue

            pka_list.append(basic_prop["pka_value"])
            smiles_list.append(
                (Chem.MolToSmiles(partner_mol), Chem.MolToSmiles(new_mol))
            )

            for mol in [partner_mol, new_mol]:
                GLOBAL_COUNTER += 1
                counter_list.append(GLOBAL_COUNTER)
                mol.SetProp(f"CHEMBL_ID", str(basic_prop["chembl_id"]))
                mol.SetProp(f"INTERNAL_ID", str(GLOBAL_COUNTER))
                mol.SetProp(f"pKa", str(basic_prop["pka_value"]))
                mol.SetProp(f"epik_atom", str(basic_prop["atom_idx"]))
                mol.SetProp(f"pKa_number", f"acid_{idx + 1}")
                mol.SetProp(f"mol-smiles", f"{Chem.MolToSmiles(mol)}")

            # add current mol to list of acidic mol. for next
            # lower pKa value, this mol is starting structure
            basic_mols.append(
                (PropertyMol.PropertyMol(partner_mol), PropertyMol.PropertyMol(new_mol))
            )
            partner_mol = deepcopy(new_mol)

        else:
            skipping_bases += 1

    return basic_mols, nr_of_skipped_mols, GLOBAL_COUNTER, skipping_bases

def main():
    """
    takes sdf file with molcules containing epik pka predictions in their properties
    and outputs a pkl file in which pairs of molecules are deposited
    that describe the protonated and deprotonated species for each pka value.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename, type: .sdf.gz or .sdf")
    parser.add_argument("--output", help="output filename, type: .pkl")
    args = parser.parse_args()
    input_zipped = False
    print(f"pH splitting used: {PH}")
    print("inputfile:", args.input)
    print("outputfile:", args.output)

    #  test if it's gzipped
    with gzip.open(args.input, "r") as fh:
        try:
            fh.read(1)
            input_zipped = True
        except gzip.BadGzipFile:
            input_zipped = False

    if input_zipped:
        with gzip.open(args.input, "r") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            processing(suppl, args)
    else:
        with open(args.input, "rb") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, removeHs=True)
            processing(suppl, args)


def processing(suppl, args):
    GLOBAL_COUNTER = 0
    nr_of_skipped_mols = 0
    all_protonation_states_enumerated = dict()

    # iterating through mols
    for nr_of_mols, mol in enumerate(suppl):
        # skit if mol can not be read
        if not mol:
            continue

        skipping_bases = 0
        skipping_acids = 0

        # test if mol has pka values
        try:
            props = mol.GetPropsAsDict()
        except AttributeError as e:
            # this mol has no pka value
            nr_of_skipped_mols += 1
            print(e)
            continue

        # count  number of pka states that epik predicted
        nr_of_protonation_states = len([s for s in props.keys() if "r_epik_pKa" in s])

        # for each protonation state extract the pka value, the atom idx and the chembl id
        properties_for_each_protonation_state = []
        for i in range(nr_of_protonation_states):
            properties_for_each_protonation_state.append(
                {
                    "pka_value": float(props[f"r_epik_pKa_{i+1}"]),
                    "atom_idx": int(props[f"i_epik_pKa_atom_{i+1}"]) - 1,
                    "chembl_id": props[f"chembl_id"],
                }
            )

        # eventhough we restricted epik predictions within a pH range of 0 to 14 there were some
        # additional pka values predicted. We introduce here a cutoff for these extrem pKa values
        upper_pka_limit = 16
        lower_pka_limit = -2

        # calculate number of acidic and basic pka values
        nr_of_acids = sum(
            pka["pka_value"] <= PH and pka["pka_value"] > lower_pka_limit
            for pka in properties_for_each_protonation_state
        )
        nr_of_bases = sum(
            pka["pka_value"] > PH and pka["pka_value"] < upper_pka_limit
            for pka in properties_for_each_protonation_state
        )
        # make sure that the sum of acid and bases equals to the number of extracted pka values
        assert nr_of_acids + nr_of_bases <= len(properties_for_each_protonation_state)
        # split properties_for_each_protonation_state into acids and bases
        acidic_mols_properties = [
            mol_pka
            for mol_pka in properties_for_each_protonation_state
            if mol_pka["pka_value"] <= PH and mol_pka["pka_value"] > lower_pka_limit
        ]
        basic_mols_properties = [
            mol_pka
            for mol_pka in properties_for_each_protonation_state
            if mol_pka["pka_value"] > PH and mol_pka["pka_value"] < upper_pka_limit
        ]
        # double check
        if len(acidic_mols_properties) != nr_of_acids:
            raise RuntimeError(f"{acidic_mols_properties=}, {nr_of_acids=}")
        if len(basic_mols_properties) != nr_of_bases:
            raise RuntimeError(f"{basic_mols_properties=}, {nr_of_bases=}")

        # clear porps for the mol at pH 7.4
        for prop in props.keys():
            mol.ClearProp(prop)

        # prepare lists in which we save the pka values, smiles and atom_idxs
        pka_list = []
        smiles_list = []
        counter_list = []

        # add mol at pH=7.4
        mol_at_ph7 = mol

        # generate states for acids and save them in acidic_mols list
        acidic_mols = []
        partner_mol = deepcopy(mol_at_ph7)
        (
            acidic_mols,
            nr_of_skipped_mols,
            GLOBAL_COUNTER,
            skipping_acids,
        ) = iterate_over_acids(
            acidic_mols_properties,
            nr_of_mols,
            partner_mol,
            nr_of_skipped_mols,
            pka_list,
            GLOBAL_COUNTER,
            PH,
            counter_list,
            smiles_list,
        )

        # generate states for bases and save them in acidic_mols list
        basic_mols = []
        partner_mol = deepcopy(mol_at_ph7)
        (
            basic_mols,
            nr_of_skipped_mols,
            GLOBAL_COUNTER,
            skipping_bases,
        ) = iterate_over_bases(
            basic_mols_properties,
            nr_of_mols,
            partner_mol,
            nr_of_skipped_mols,
            pka_list,
            GLOBAL_COUNTER,
            PH,
            counter_list,
            smiles_list,
        )

        # combine basic and acidic mols, skip neutral mol for acids
        combined_mols = acidic_mols + basic_mols
        # make sure that the number of acids and bases make sense
        if (
            len(combined_mols)
            != len(acidic_mols_properties)
            - skipping_acids
            + len(basic_mols_properties)
            - skipping_bases
        ):
            raise RuntimeError(
                combined_mols,
                acidic_mols_properties,
                skipping_acids,
                basic_mols_properties,
                skipping_bases,
            )

        # if protonation states were present
        if len(combined_mols) != 0:
            # extract chembl id
            chembl_id = combined_mols[0][0].GetProp("CHEMBL_ID")
            print(f"CHEMBL_ID: {chembl_id}")
            # iterate over protonation states
            for mols in combined_mols:
                if mols[0].GetProp("pKa") != mols[1].GetProp("pKa"):
                    raise AssertionError(mol[0].GetProp("pKa"), mol[1].GetProp("pKa"))

                mol1, mol2 = mols
                pka = mol1.GetProp("pKa")
                counter = mol1.GetProp("INTERNAL_ID")
                print(
                    f"{counter=}, {pka=}, {mol1.GetProp('mol-smiles')}, prot, {mol1.GetProp('epik_atom')}"
                )
                pka = mol2.GetProp("pKa")
                counter = mol2.GetProp("INTERNAL_ID")
                print(
                    f"{counter=}, {pka=}, {mol2.GetProp('mol-smiles')}, deprot, {mol1.GetProp('epik_atom')}"
                )
            print(pka_list)
            if chembl_id in all_protonation_states_enumerated.keys():
                raise RuntimeError("Repeated chembl id!")

            all_protonation_states_enumerated[chembl_id] = {
                "mols": combined_mols,
                "pKa_list": pka_list,
                "smiles_list": smiles_list,
                "counter_list": counter_list,
            }

    print(f"finished splitting {nr_of_mols} molecules")
    print(f"skipped mols: {nr_of_skipped_mols}")
    # save everything
    pickle.dump(all_protonation_states_enumerated, open(args.output, "wb+"))


if __name__ == "__main__":
    main()
