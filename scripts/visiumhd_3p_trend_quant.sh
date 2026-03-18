#!/bin/bash
set -eo pipefail

# ====================================================
# DEPENDENCIES:
# - Space Ranger (>=v4.0 for barcode_mappings.parquet)
# - regtools
# - samtools
# - Subread (featureCounts)
# - R (with Sierra package installed)
# - Python 3
#   - pandas
#   - scipy
#   - h5py
#   - umi_tools
#   - splisosm (>=v1.0.4 for load_visiumhd_probe)
# ====================================================

# ==========================================
# CONFIGURATION (Update these paths)
# ==========================================
THREADS=16
# REF_DIR="/gpfs/commons/home/jsu/reference/cellranger/refdata-gex-mm10-2020-A"
# GTF_FILE="/gpfs/commons/home/jsu/reference/annotations/gencode.vM10.annotation.gtf"
REF_DIR="/gpfs/commons/home/jsu/reference/cellranger/refdata-gex-GRCm39-2024-A"
GTF_FILE="/gpfs/commons/home/jsu/reference/annotations/gencode.vM33.annotation.gtf"

DATA_DIR="$HOME/data/visiumhd/mouse_brain_3prime"
FASTQ_DIR="${DATA_DIR}/inputs/Visium_HD_3prime_Mouse_Brain_fastqs/"
CYTA_IMAGE="${DATA_DIR}/inputs/Visium_HD_3prime_Mouse_Brain_image.tif"
TISSUE_IMAGE="${DATA_DIR}/inputs/Visium_HD_3prime_Mouse_Brain_tissue_image.btf"
SLIDE_ID="H1-JDQ83M6"
AREA_ID="A1"

# All results will be saved under OUT_DIR
OUT_DIR="${DATA_DIR}/test_run"
PEAK_DIR="${OUT_DIR}/sierra_peaks"

# Load dependencies
# module load spaceranger regtools samtools subread
# mamba activate r-4.4

# ==============================================
# STEP 0: Space Ranger (Optional if already run)
# ==============================================
echo "0. Running Space Ranger to generate BAM file..."
spaceranger count --id=hd_count \
  --transcriptome="$REF_DIR" \
  --fastqs="$FASTQ_DIR" \
  --slide=$SLIDE_ID --area=$AREA_ID \
  --cytaimage="$CYTA_IMAGE" \
  --image="$TISSUE_IMAGE" \
  --create-bam=true \
  --output-dir="$OUT_DIR"

BAM_FILE="${OUT_DIR}/outs/possorted_genome_bam.bam"

# ==========================================
# STEP 1: Sierra Preparation & R Execution
# ==========================================
mkdir -p "$PEAK_DIR"

PEAK_FILE="${PEAK_DIR}/peak.txt"
PEAK_ANNOT="${PEAK_DIR}/peak.annot.txt"
PEAK_ANNOT_OVERLAP="${PEAK_DIR}/peak.annot.overlap.txt"
JUNC_FILE="${PEAK_DIR}/possorted_genome_bam.junc.bed"

echo "1a. Extracting junctions..."
regtools junctions extract "$BAM_FILE" -o "$JUNC_FILE" -s 0

echo "1b. Running Sierra FindPeaks in R..."
Rscript -e "
library(Sierra)
FindPeaks(output.file='${PEAK_FILE}', gtf.file='${GTF_FILE}', bamfile='${BAM_FILE}', junctions.file='${JUNC_FILE}', ncores=${THREADS}, min.jcutoff.prop = 0.0, min.cov.prop = 0.0, min.peak.prop = 0.0)
AnnotatePeaksFromGTF(peak.sites.file='${PEAK_FILE}', gtf.file='${GTF_FILE}', output.file='${PEAK_ANNOT}', transcriptDetails=TRUE)
AnnotatePeaksFromGTF(peak.sites.file='${PEAK_FILE}', gtf.file='${GTF_FILE}', output.file='${PEAK_ANNOT_OVERLAP}', transcriptDetails=TRUE, annotation_correction=FALSE)
"

# ==========================================
# STEP 2: Custom Peak Quantification
# ==========================================
SAF_FILE="${PEAK_FILE}.saf"
BED_FILE="${PEAK_FILE}.bed12"
echo "2a. Converting peaks to SAF and BED12 formats..."

python3 -c '
import sys

input_file = sys.argv[1]
saf_file = sys.argv[2]
bed_file = sys.argv[3]

seen_bed_regions = set()

with open(input_file, "r") as fin, \
     open(saf_file, "w") as fout_saf, \
     open(bed_file, "w") as fout_bed:

    # Skip header
    fin.readline()

    # Write SAF header
    fout_saf.write("GeneID\tChr\tStart\tEnd\tStrand\n")

    for line in fin:
        p = line.strip().split("\t")
        if len(p) < 14:
            continue

        # Extract fields based on the column indices
        chrom = p[1]
        strand_raw = p[2]
        strand = "+" if strand_raw in ["1", "+"] else "-"
        fit_start = int(p[5])
        fit_end = int(p[6])
        exon_intron = p[10]
        exon_pos = p[11]
        polyA_ID = p[13]

        # ---------------------------------------------------------
        # 1. SAF Format Logic
        # ---------------------------------------------------------
        if exon_intron == "across-junctions" and exon_pos != "NA":
            for b in exon_pos.strip("()").split(")("):
                s, e = b.split(",")
                fout_saf.write(f"{polyA_ID}\t{chrom}\t{s}\t{e}\t{strand}\n")
        else:
            fout_saf.write(f"{polyA_ID}\t{chrom}\t{fit_start}\t{fit_end}\t{strand}\n")

        # ---------------------------------------------------------
        # 2. BED12 Format Logic
        # ---------------------------------------------------------
        # Deduplication check
        bed_key = (chrom, fit_start, fit_end, strand)
        if bed_key in seen_bed_regions:
            continue
        seen_bed_regions.add(bed_key)

        if exon_intron == "no-junctions" or exon_pos == "NA":
            block_sizes = [fit_end - fit_start]
            block_starts = [0]
        else:
            # Parse the (start,end)(start,end) string
            blocks = exon_pos.strip("()").split(")(")
            exons = [list(map(int, b.split(","))) for b in blocks]

            # Update the first and last exon boundaries
            exons[0][0] = fit_start
            exons[-1][1] = fit_end

            block_sizes = []
            block_starts = []

            # Calculate sizes and relative starts, filtering out lengths <= 0
            for s, e in exons:
                size = e - s
                if size > 0:
                    block_sizes.append(size)
                    block_starts.append(s - fit_start)

        # Skip if no valid blocks remain after cleaning
        if not block_sizes:
            continue

        # Format BED12 columns
        b_count = len(block_sizes)
        b_sizes_str = ",".join(map(str, block_sizes))
        b_starts_str = ",".join(map(str, block_starts))

        # Write BED12 line (12 columns)
        fout_bed.write(
            f"{chrom}\t{fit_start}\t{fit_end}\t{polyA_ID}\t0\t{strand}\t"
            f"{fit_start}\t{fit_end}\t0\t{b_count}\t{b_sizes_str}\t{b_starts_str}\n"
        )
' "$PEAK_FILE" "$SAF_FILE" "$BED_FILE"

echo "2b. Running featureCounts..."
featureCounts -a "$SAF_FILE" -F SAF -f -s 1 -T "$THREADS" -R BAM -o "${PEAK_DIR}/featureCounts_summary.txt" "$BAM_FILE"
TAGGED_BAM="${PEAK_DIR}/$(basename $BAM_FILE).featureCounts.bam"

echo "2c. Sorting and indexing BAM..."
SORTED_BAM="${PEAK_DIR}/$(basename $BAM_FILE .bam).tagged.sorted.bam"
samtools sort -@ "$THREADS" -o "$SORTED_BAM" "$TAGGED_BAM"
samtools index "$SORTED_BAM"
rm "$TAGGED_BAM" # Clean up intermediate BAM

echo "2d. Counting UMIs..."
UMI_COUNTS="${PEAK_DIR}/Sierra_counts.tsv.gz"
umi_tools count --per-gene --gene-tag=XT --assigned-status-tag=XS --per-cell \
    --extract-umi-method=tag --umi-tag=UB --cell-tag=CB \
    -I "$SORTED_BAM" -S "$UMI_COUNTS"

# ==========================================
# STEP 3: Generate 10x-compatible .h5 Matrix
# ==========================================
echo "3. Generating 10x-compatible .h5 matrix..."
H5_OUT="${OUT_DIR}/outs/binned_outputs/square_002um/raw_probe_bc_matrix.h5"

# Check if already exists and remove if so
if [ -f "$H5_OUT" ]; then
    echo "Warning: $H5_OUT already exists. Will overwrite it."
    rm "$H5_OUT"
fi

python3 -c '
import sys, pandas as pd, numpy as np, h5py
from scipy.sparse import coo_matrix

counts_file, annot_file, output_h5 = sys.argv[1:4]

df = pd.read_csv(counts_file, sep="\t")
if df.empty:
    sys.exit("Error: No counts found.")

# Append cell suffix to ensure uniqueness and compatibility with 10x format
if not df["cell"].str.endswith("-1").all():
    df["cell"] = df["cell"].astype(str) + "-1"

df["cell"] = pd.Categorical(df["cell"])
df["gene"] = pd.Categorical(df["gene"])

barcodes = df["cell"].cat.categories.values.astype("S")
features = df["gene"].cat.categories.values.astype("S")
features_str = df["gene"].cat.categories.astype(str).tolist()

annot_df = pd.read_csv(annot_file, sep="\t", index_col=0).reindex(features_str).fillna("")

mat = coo_matrix((df["count"], (df["gene"].cat.codes, df["cell"].cat.codes)),
                 shape=(len(features), len(barcodes))).tocsc()

with h5py.File(output_h5, "w") as f:
    grp = f.create_group("matrix")
    grp.create_dataset("barcodes", data=barcodes)
    grp.create_dataset("data", data=mat.data)
    grp.create_dataset("indices", data=mat.indices)
    grp.create_dataset("indptr", data=mat.indptr)
    grp.create_dataset("shape", data=np.array(mat.shape, dtype=np.int32))

    feat_grp = grp.create_group("features")
    feat_grp.create_dataset("id", data=features)
    feat_grp.create_dataset("name", data=features)
    feat_grp.create_dataset("feature_type", data=np.array([b"Gene Expression"] * len(features)))
    feat_grp.create_dataset("genome", data=np.array([b""] * len(features)))

    for col in annot_df.columns:
        feat_grp.create_dataset(col, data=annot_df[col].astype(str).values.astype("S"))

print(f"Successfully created {output_h5} with {mat.shape[0]} peaks and {mat.shape[1]} cells.")
' "$UMI_COUNTS" "$PEAK_ANNOT_OVERLAP" "$H5_OUT"

# =======================================
# STEP 4: Generate peak-level SpatialData
# ========================================

ZARR_OUT="${OUT_DIR}/sdata_peak.filtered.zarr"
echo "4. Generating peak-level SpatialData in Zarr format..."

python3 -c '
import sys
from splisosm.io import load_visiumhd_probe

visium_hd_outs = sys.argv[1]
sdata_zarr = sys.argv[2]
bin_sizes = [2, 8, 16]

sdata = load_visiumhd_probe(
    path=visium_hd_outs,
    bin_sizes=bin_sizes,
    filtered_counts_file=True,
    load_all_images=False,
    var_names_make_unique=True,
    counts_layer_name="counts",
)
sdata.write(sdata_zarr, overwrite=True)
' "$OUT_DIR/outs" "$ZARR_OUT"

# Optional: Zip the Zarr folder for easier sharing
ZARR_ZIP="${OUT_DIR}/$(basename $ZARR_OUT).zip"
if [ -f "$ZARR_ZIP" ]; then
    echo "Warning: $ZARR_ZIP already exists. Will overwrite it."
    rm "$ZARR_ZIP"
fi

cd "$ZARR_OUT"
zip -0 -r "../$(basename $ZARR_OUT).zip" .
# unzip "$ZARR_ZIP" -d "path/to/zarr.zarr"

echo "Pipeline complete! Peak-level count matrix saved to $H5_OUT and SpatialData saved to $ZARR_OUT"