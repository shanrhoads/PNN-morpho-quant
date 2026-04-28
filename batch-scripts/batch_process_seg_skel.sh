#!/bin/bash

#### GENERAL INFO ####
# Script to batch processing of image files using 'seg_skel_batch-parallel.py'
# This loops through a folder of files, and for each file, submits a batch job to run the Python script in parallel on a cluster using 'sbatch'


#### USER SPECIFIED INPUTS ####
# do NOT include spaces around the = or trailing "/" in directory paths
input_dir="/users/s/r/srhoads/PNN-morpho_Hayli-pilot-data/Male/Pair_5/3D_STED/WT"
output_dir="/users/s/r/srhoads/PNN-morpho_Hayli-pilot-data/Male/Pair_5/3D_STED/WT_seg-skel_20260427-sh-batch-test"
file_type=".tif"



#### BATCH SCRIPT ####
# Check if Python batch script exists in the same directory as this .sh script
script_dir="$(cd "$(dirname "$0")" && pwd)"
python_script="${script_dir}/seg_skel_batch-parallel.py"

if [ ! -f "${python_script}" ]; then
  echo "Python batch script not found: ${python_script}"
  exit 1
fi

# Get list of files from input_dir
set -- "${input_dir}"/*${file_type}

# Check if any ${file_type} files were found in the input_dir
if [ "$1" = "${input_dir}/*${file_type}" ]; then
  echo "No ${file_type} files found in input_dir: ${input_dir}"
  exit 0
fi

# Create output_dir if it doesn't exist
if [ ! -d "${output_dir}" ]; then
  echo "Output directory not found. Creating: ${output_dir}"
  mkdir -p "${output_dir}"
fi

# Loop through each .tif file and submit a batch job for processing
for tif_path in "$@"; do
  file="$(basename "${tif_path}")"
  file="${file%${file_type}}"

  echo "Initiating segmentation for: ${file}.tif"
  # echo "Input file: ${tif_path}"
  # echo "Output directory: ${output_dir}"
  # echo "Log file: ${output_dir}/${file}.log"

  # Run batch processing script for each file, output log to file-specific log file
  sbatch -t 2- --mem 64G --out "${output_dir}/${file}.log" \
    --wrap "python \"${python_script}\" \"${tif_path}\" \"${output_dir}\""
done


 
