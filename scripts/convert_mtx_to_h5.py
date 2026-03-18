import sys, gzip, re, h5py
import pandas as pd
import numpy as np
import scipy.io as sio

if len(sys.argv) != 4:
    sys.exit(
        "Usage: python convert_mtx_to_h5.py <mtx_dir> <gff.gz> <out.h5>\n"
        "\n"
        "<mtx_dir>: Directory containing matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz\n"
        "<gff.gz>: Stringtie GFF file containing transcript annotations\n"
        "<out.h5>: Output H5 file\n"
    )

mtx_dir, gff_file, out_file = sys.argv[1:4]

# 1. Parse GFF to map cmp_ref (transcript) -> (ref_gene_id, ref_gene_name)
t2info = {}
with gzip.open(gff_file, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.split("\t")

        if len(parts) > 8 and parts[2] == "transcript":
            attr = parts[8]

            # The ENSMUST ID is typically in cmp_ref for StringTie assemblies
            cmp_ref = re.search(r'cmp_ref[= ]"?([^";]+)', attr)
            if not cmp_ref:
                # Fallback to transcript_id if cmp_ref is missing
                cmp_ref = re.search(r'transcript_id[= ]"?([^";]+)', attr)

            if cmp_ref:
                t_id = cmp_ref.group(1)

                # Extract Gene ID (prefer ref_gene_id, fallback to gene_id)
                r_g_id = re.search(r'ref_gene_id[= ]"?([^";]+)', attr)
                g_id = re.search(r'gene_id[= ]"?([^";]+)', attr)
                final_g_id = (
                    r_g_id.group(1) if r_g_id else (g_id.group(1) if g_id else t_id)
                )

                # Extract Gene Name (prefer ref_gene_name, fallback to gene_name)
                r_g_name = re.search(r'ref_gene_name[= ]"?([^";]+)', attr)
                g_name = re.search(r'gene_name[= ]"?([^";]+)', attr)
                final_g_name = (
                    r_g_name.group(1)
                    if r_g_name
                    else (g_name.group(1) if g_name else final_g_id)
                )

            # Keep only reference transcripts that map to valid mouse genes (ENSMUSG)
            if final_g_id.startswith("ENSMUSG") and t_id.startswith("ENSMUST"):
                t2info[t_id] = (final_g_id, final_g_name)

# 2. Load MTX and features
mat = sio.mmread(f"{mtx_dir}/matrix.mtx.gz").tocsc()
barcodes = pd.read_csv(f"{mtx_dir}/barcodes.tsv.gz", header=None)[0].values.astype("S")
features = pd.read_csv(f"{mtx_dir}/features.tsv.gz", sep="\t", header=None)

# Column 1 contains the transcript IDs (ENSMUST)
t_ids = features[1].values
g_ids = [t2info.get(t, (t, t))[0] for t in t_ids]
g_names = [t2info.get(t, (t, t))[1] for t in t_ids]

# Make sure barcodes are in the correct format s_002um_xxxxx_xxxxx-1
if not re.match(r"^s_\d+um_\d+_\d+-\d+$", barcodes[0].decode()):
    new_barcodes = []
    for bc in barcodes:
        bc_str = bc.decode()
        match = re.match(r"^(s_\d+um)_(\d+)_(\d+)", bc_str)
        if match:
            prefix, x, y = match.groups()
            new_barcodes.append(f"{prefix}_{x}_{y}-1".encode())
        else:
            new_barcodes.append(
                bc
            )  # Keep original if it doesn't match expected pattern
    barcodes = np.array(new_barcodes).astype("S")


# 3. Write standard 10x H5 format
with h5py.File(out_file, "w") as f:
    grp = f.create_group("matrix")
    grp.create_dataset("barcodes", data=barcodes)
    grp.create_dataset("data", data=mat.data)
    grp.create_dataset("indices", data=mat.indices)
    grp.create_dataset("indptr", data=mat.indptr)
    grp.create_dataset("shape", data=mat.shape)

    f_grp = grp.create_group("features")
    f_grp.create_dataset("id", data=np.array(t_ids, dtype="S"))
    f_grp.create_dataset("name", data=np.array(t_ids, dtype="S"))
    f_grp.create_dataset("gene_id", data=np.array(g_ids, dtype="S"))
    f_grp.create_dataset("gene_name", data=np.array(g_names, dtype="S"))
    f_grp.create_dataset(
        "feature_type", data=np.array(["Gene Expression"] * len(t_ids), dtype="S")
    )
    f_grp.create_dataset("genome", data=np.array([""] * len(t_ids), dtype="S"))
