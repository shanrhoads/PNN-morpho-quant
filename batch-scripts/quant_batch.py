#### GENERAL INFO ####
# This script runs the PNN morphology quantification of A FOLDER OF IMAGES based on the input parameters specified below.
# It is intended to be run on its own, and will process all files in the specified input folder that match the specified file type.


#### USER SPECIFIED INPUTS ####
# specify the location of the PNN-morpho-quant repo
repo_path = '/users/s/r/srhoads/PNN-morpho-quant'

# inputs for batch_PNN_quant function
file_out_prefix=
raw_file_path = 
raw_file_type =
seg_skel_path = 
quant_out_path = 
include_surface_area =




#### IMPORTS ####
import sys
sys.path.insert(0, repo_path)
from src.quantification import batch_PNN_quant



#### BATCH PROCESS FUNCTION CALL ####
batch_PNN_quant(file_out_prefix = file_out_prefix,
                 raw_file_path = raw_file_path, 
                 raw_file_type = raw_file_type,
                 seg_skel_path = seg_skel_path,
                 quant_out_path = quant_out_path,
                 include_surface_area = include_surface_area)