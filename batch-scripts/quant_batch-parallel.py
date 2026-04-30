#### GENERAL INFO ####
# This script runs the quantification of A SINGLE IMAGE and its associated segmentation and skeletonization files.
# It is intended to be run in parallel across multiple files using the batch_process_quant.sh script, which will call this script for each file to be processed.
# It will not work if run on its own, as it requires command line arguments for the repo path and quantification settings. 
# The batch_process_quant.sh script will handle passing the correct file paths to this script for each file to be processed in parallel.


#### USER SPECIFIED INPUTS ####
# specify the location of the PNN-morpho-quant repo
repo_path = '/users/s/r/srhoads/PNN-morpho-quant'

# specify settings for batch processing quantification
include_surface_area=True



#### IMPORTS ####
import sys
file_prefix = sys.argv[1]
raw_file_path = sys.argv[2]
seg_skel_path = sys.argv[3]
quant_out_path = sys.argv[4]

sys.path.insert(0, repo_path)
from src.quantification import batch_PNN_quant_PARALLEL

import warnings
warnings.filterwarnings("ignore")


#### BATCH PROCESS FUNCTION CALL ####
batch_PNN_quant_PARALLEL(file_out_prefix = file_prefix,
                         raw_file_path = raw_file_path,
                         seg_skel_path = seg_skel_path,
                         quant_out_path = quant_out_path,
                         include_surface_area = include_surface_area)