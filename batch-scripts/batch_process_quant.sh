#!/bin/bash

#### GENERAL INFO ####
# Script to batch process image files using 'quant_batch-parallel.py'
# This loops through a folder of files, and for each file, submits a batch job to run the quantification Python script in parallel on a cluster using 'sbatch'


#### USER SPECIFIED INPUTS ####
# do NOT include spaces around the = or trailing "/" in directory paths
file_prefix="20260428-one-test"
rawfile_dir="/users/s/r/srhoads/PNN-morpho_Hayli-pilot-data/Male/Pair_5/3D_STED/WT-one-test"
seg_skel_dir="/users/s/r/srhoads/PNN-morpho_Hayli-pilot-data/Male/Pair_5/3D_STED/WT_seg-skel_20260427-one-test"
quant_dir="/users/s/r/srhoads/PNN-morpho_Hayli-pilot-data/Male/Pair_5/3D_STED/WT_quant_20260428-one-test_parallel"
raw_file_type=".tif"


#### BATCH SCRIPT ####
# Check if Python batch script exists in the same directory as this .sh script
script_dir="$(cd "$(dirname "$0")" && pwd)"
python_script="${script_dir}/quant_batch-parallel.py"

if [ ! -f "${python_script}" ]; then
  echo "Python batch script not found: ${python_script}"
  exit 1
fi

# Get list of files from input_dir
set -- "${rawfile_dir}"/*${raw_file_type}

# Check if any ${raw_file_type} files were found in the rawfile_dir
if [ "$1" = "${rawfile_dir}/*${raw_file_type}" ]; then
  echo "No ${raw_file_type} files found in rawfile_dir: ${rawfile_dir}"
  exit 0
fi

# Create quant_dir if it doesn't exist
if [ ! -d "${quant_dir}" ]; then
  echo "Output directory not found. Creating: ${quant_dir}"
  mkdir -p "${quant_dir}"
fi

csv_path="${quant_dir}/${file_prefix}-PNN_quantification.csv"

# if the quantification CSV already exists, raise an error to avoid overwriting existing data
if [ -f "${csv_path}" ]; then
  echo "The ${csv_path} file already exists."
  echo "Please choose a different quant_dir or file_prefix to avoid overwriting existing data."
  exit 1
fi

# create empty .csv file for storing quantification results
echo "Created empty .csv file for quantification results: ${csv_path}"
touch "${csv_path}"

# Loop through each .tif file and submit a batch job for processing
for tif_path in "$@"; do
  file="$(basename "${tif_path}")"
  file="${file%${raw_file_type}}"

  echo "Initiating segmentation for: ${file}"
  # echo "Input file: ${tif_path}"
  # echo "Segmentation and skeletonization directory: ${seg_skel_dir}"
  # echo "Output directory: ${quant_dir}"
  # echo "Log file: ${quant_dir}/${file}.log"
  # echo "Created empty .csv file for quantification results: ${quant_dir}/${file_prefix}-PNN_quantification.csv"

  # Run batch processing script for each file, output log to file-specific log file
  sbatch -t 2- --mem 64G --out "${quant_dir}/${file}.log" \
    --wrap "python \"${python_script}\" \"${file_prefix}\" \"${tif_path}\" \"${seg_skel_dir}\" \"${quant_dir}\""
done


 
