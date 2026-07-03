#!/bin/bash

# Runs 20-step profiling for every (backend x seq_len) combination and
# collects the median-iteration CSV into a single file for Google Sheets.
#
# Usage: bash examples/mamba/run_experiments.sh
# Output: experiments/results.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/atripathy/Megatron-LM"
LOG_DIR="${ROOT_DIR}/experiments/logs"
OUTPUT_FILE="${ROOT_DIR}/experiments/results.csv"

BACKENDS=(triton torch cudnn)
SEQ_LENS=(8192 16384 32768 65536 131072)
TRAIN_ITERS=20

mkdir -p "${LOG_DIR}"
> "${OUTPUT_FILE}"

HEADER_WRITTEN=false

for BACKEND in "${BACKENDS[@]}"; do
    for SEQ_LEN in "${SEQ_LENS[@]}"; do
        NAME="exp_${BACKEND}_seq${SEQ_LEN}"
        LOG_FILE="${LOG_DIR}/log_${NAME}.out"

        echo "=== Running: backend=${BACKEND} seq_len=${SEQ_LEN} ==="

        SEQ_LEN=${SEQ_LEN} TRAIN_ITERS=${TRAIN_ITERS} \
            bash "${SCRIPT_DIR}/train.sh" "${NAME}" 1 0 "${BACKEND}" \
            &> "${LOG_FILE}"

        echo "    Done. Extracting CSV..."

        # Extract the last CSV block (header + forward + backward lines)
        # The block looks like:
        #   [rank0] CSV (median fwd of 5):
        #   phase,label,...
        #   forward,...
        #   backward,...
        python3 - "${LOG_FILE}" "${BACKEND}" "${SEQ_LEN}" "${OUTPUT_FILE}" "${HEADER_WRITTEN}" <<'EOF'
import sys, re

log_file, backend, seq_len, out_file, header_written = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
header_written = header_written == "true"

with open(log_file) as f:
    content = f.read()

# Find all CSV blocks
blocks = re.findall(r"CSV \(median fwd of \d+\):\n(.*?)\n(.*?)\n(.*?)(?:\n|$)", content, re.DOTALL)
if not blocks:
    print(f"WARNING: no CSV block found in {log_file}", file=sys.stderr)
    sys.exit(0)

# Use the last block
header_line, fwd_line, bwd_line = blocks[-1]

with open(out_file, "a") as f:
    if not header_written:
        f.write(f"backend,seq_len,{header_line}\n")
    f.write(f"{backend},{seq_len},{fwd_line}\n")
    f.write(f"{backend},{seq_len},{bwd_line}\n")
EOF

        if [ $? -eq 0 ]; then
            HEADER_WRITTEN=true
        fi

        echo "    Appended to ${OUTPUT_FILE}"
    done
done

echo ""
echo "=== All done. Results in: ${OUTPUT_FILE} ==="
