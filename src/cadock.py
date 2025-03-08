# src/cadock.py

import os
import re
import subprocess
import pandas as pd

from Bio.PDB import PDBList
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from rdkit import Chem
from rdkit.Chem import AllChem

########################################
# 1) Helpers
########################################
def run_command_with_live_output(command, log_file):
    """
    Runs a command in a subprocess, capturing stdout/stderr to console & log file.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    with open(log_file, 'w') as lf:
        for line in process.stdout:
            print(line, end="")  # to console
            lf.write(line)       # to log file
    return process.wait()


def keep_only_chain_A_with_fallback(input_pdb, output_pdb):
    """
    Extract chain A lines. If none, fallback to entire input_pdb.
    We also skip TER lines not in chain A to avoid PDBFixer NoneType errors.
    """
    chain_a_count = 0
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            chain_id = line[21] if len(line) >= 22 else None

            if line.startswith(("ATOM", "HETATM")):
                if chain_id == 'A':
                    outfile.write(line)
                    chain_a_count += 1
            elif line.startswith("TER"):
                if chain_id == 'A':
                    outfile.write(line)
            elif line.startswith("END"):
                outfile.write(line)

    if chain_a_count == 0:
        print("[WARN] No chain A lines found; using full PDB instead.")
        with open(input_pdb, 'r') as inf, open(output_pdb, 'w') as outf:
            outf.write(inf.read())


def fix_with_pdbfixer(pdb_in, pdb_out):
    """
    Use PDBFixer to add missing residues, atoms, and hydrogens at pH=7.
    """
    fixer = PDBFixer(filename=pdb_in)
    # Do NOT remove heterogens => keep co-ligand if in chain A
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    with open(pdb_out, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)


def convert_pdb_to_pdbqt_receptor(input_pdb, output_pdbqt):
    """
    Receptor PDB -> PDBQT using obabel
    """
    cmd = [
        "obabel", "-i", "pdb", input_pdb,
        "-o", "pdbqt", "-O", output_pdbqt,
        "-xr", "-xn", "-xp"
    ]
    subprocess.run(cmd, check=True)


def convert_pdb_to_pdbqt_ligand(input_pdb, output_pdbqt):
    """
    Ligand PDB -> PDBQT
    """
    cmd = [
        "obabel", "-i", "pdb", input_pdb,
        "-o", "pdbqt", "-O", output_pdbqt,
        "-h"
    ]
    subprocess.run(cmd, check=True)


########################################
# 2) p2rank
########################################
def run_p2rank_and_get_center(receptor_pdb, pdb_id):
    """
    Run p2rank -> parse top pocket center from CSV, return (x,y,z).
    """
    p2rank_path = os.path.join(os.getcwd(), "p2rank_2.4.2", "prank")
    if not os.path.isfile(p2rank_path):
        raise FileNotFoundError(f"p2rank not found at {p2rank_path}")

    # Ensure it's executable
    if not os.access(p2rank_path, os.X_OK):
        os.chmod(p2rank_path, 0o755)

    cmd = [p2rank_path, "predict", "-f", receptor_pdb]
    code = subprocess.run(cmd, check=False)
    if code.returncode != 0:
        raise RuntimeError("p2rank failed. Check your environment.")

    # p2rank output location example:
    # p2rank_2.4.2/test_output/predict_{filename}/{filename}.pdb_predictions.csv
    base_name = os.path.splitext(os.path.basename(receptor_pdb))[0]
    predictions_csv = f"p2rank_2.4.2/test_output/predict_{base_name}/{base_name}.pdb_predictions.csv"
    df = pd.read_csv(predictions_csv, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    cx = float(df["center_x"].iloc[0])
    cy = float(df["center_y"].iloc[0])
    cz = float(df["center_z"].iloc[0])

    print(f"[p2rank] Pocket center: ({cx}, {cy}, {cz})")
    return (cx, cy, cz)


########################################
# 3) Ligand Prep
########################################
def generate_multiple_conformers(mol, num_confs=3):
    """
    RDKit: embed multiple conformers, minimize each (MMFF or UFF).
    """
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42)
    for cid in cids:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=200)
    return mol

def prepare_ligands(smiles_list, num_confs=3, out_dir="ligand_prep"):
    """
    Takes a list of SMILES, generates multiple conformers for each.
    Writes each conformer to a separate PDB file.
    Returns a list of (pdb_file, label).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    results = []
    for idx, smi in enumerate(smiles_list, start=1):
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            print(f"[WARN] Invalid SMILES: {smi}")
            continue
        mol3d = generate_multiple_conformers(mol, num_confs=num_confs)

        # Write each conformer
        for cid, conf in enumerate(mol3d.GetConformers(), start=1):
            tmp_mol = Chem.Mol(mol3d, False, conf.GetId())
            pdb_path = os.path.join(out_dir, f"lig_{idx}_conf{cid}.pdb")
            Chem.MolToPDBFile(tmp_mol, pdb_path)
            label = os.path.splitext(os.path.basename(pdb_path))[0]
            results.append((pdb_path, label))
    return results


########################################
# 4) The Main Docking Function
########################################
def perform_docking(
    smiles_list=None,   # e.g. ["CCOC(=O)C1=CC=CC=C1Cl", ...]
    pdb_id="5ZMA",      # The PDB ID
    num_confs=3,
    docking_folder="docking_results"
):
    """
    1) Download + keep chain A (fallback) -> fix with PDBFixer
    2) p2rank -> get pocket center
    3) Receptor -> PDBQT
    4) Prepare ligands (multi-conformers)
    5) Dock each conformer with AutoDock Vina
    6) Convert best pose -> final complex
    """
    if smiles_list is None or len(smiles_list) == 0:
        print("[ERROR] No SMILES provided.")
        return

    if not os.path.exists(docking_folder):
        os.makedirs(docking_folder, exist_ok=True)

    # A) Receptor Prep
    print(f"[Docking] PDB ID: {pdb_id}")
    pdbl = PDBList()
    raw_file = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir=docking_folder)
    raw_pdb = os.path.join(docking_folder, f"{pdb_id}_raw.pdb")
    os.rename(raw_file, raw_pdb)

    chainA_tmp = os.path.join(docking_folder, f"{pdb_id}_chainA_tmp.pdb")
    keep_only_chain_A_with_fallback(raw_pdb, chainA_tmp)

    receptor_pdb = os.path.join(docking_folder, f"{pdb_id}_prepared.pdb")
    fix_with_pdbfixer(chainA_tmp, receptor_pdb)

    # B) p2rank -> pocket center
    cx, cy, cz = run_p2rank_and_get_center(receptor_pdb, pdb_id)
    box_size = 20.0

    # Convert receptor -> PDBQT
    receptor_pdbqt = os.path.join(docking_folder, f"{pdb_id}_prepared.pdbqt")
    convert_pdb_to_pdbqt_receptor(receptor_pdb, receptor_pdbqt)

    # C) Prepare ligands
    lig_out_dir = os.path.join(docking_folder, "ligands")
    lig_pdb_list = prepare_ligands(smiles_list=smiles_list, num_confs=num_confs, out_dir=lig_out_dir)
    if len(lig_pdb_list) == 0:
        print("[Docking] No valid ligands to dock.")
        return

    # D) Dock each conformer
    results_csv = os.path.join(docking_folder, "docking_results.csv")
    with open(results_csv, 'w') as out_f:
        out_f.write("LigandLabel,Score\n")

        for i, (lig_pdb, label) in enumerate(lig_pdb_list, start=1):
            print(f"\n[Docking] Processing ligand conformer #{i}: {label}")

            lig_pdbqt = os.path.join(docking_folder, f"{label}.pdbqt")
            convert_pdb_to_pdbqt_ligand(lig_pdb, lig_pdbqt)

            out_pdbqt = os.path.join(docking_folder, f"{label}_out.pdbqt")
            log_file = os.path.join(docking_folder, f"{label}_vina.log")

            vina_cmd = [
                "vina",
                "--receptor", receptor_pdbqt,
                "--ligand", lig_pdbqt,
                "--out", out_pdbqt,
                "--center_x", str(cx),
                "--center_y", str(cy),
                "--center_z", str(cz),
                "--size_x", str(box_size),
                "--size_y", str(box_size),
                "--size_z", str(box_size),
                "--num_modes", "1"
            ]
            ret_code = run_command_with_live_output(vina_cmd, log_file)

            best_score = "N/A"
            if ret_code == 0:
                with open(log_file, 'r') as lg:
                    for line in lg:
                        # The line with rank=1 typically starts with something like '  1 '
                        if re.match(r'^\s*1\s+', line):
                            parts = line.split()
                            if len(parts) >= 2:
                                best_score = parts[1]
                            break
            else:
                best_score = "ERROR"

            print(f"[Docking] Best docking score for {label}: {best_score}")
            out_f.write(f"{label},{best_score}\n")

            # Convert best pose to PDB
            docked_pdb = os.path.join(docking_folder, f"{label}_docked.pdb")
            subprocess.run([
                "obabel",
                "-ipdbqt", out_pdbqt,
                "-opdb", "-O", docked_pdb,
                "-d"
            ], check=True)

            # Merge with receptor => final complex
            final_complex = os.path.join(docking_folder, f"{label}_complex.pdb")
            with open(final_complex, 'w') as fc:
                with open(receptor_pdb, 'r') as recf:
                    for line in recf:
                        if line.startswith("END"):
                            continue
                        fc.write(line)
                fc.write("TER\n")
                with open(docked_pdb, 'r') as dockf:
                    for line in dockf:
                        fc.write(line)
                fc.write("END\n")

            print(f"[Docking] Final complex saved as: {final_complex}")

    print(f"\n[DONE] Docking complete. Results in {results_csv}")
