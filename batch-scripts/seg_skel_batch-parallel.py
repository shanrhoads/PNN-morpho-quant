#### GENERAL INFO ####
# This script runs the segmentation and skeletonization of A SINGLE IMAGE based on the input parameters specified below.
# It is intended to be run in parallel across multiple files using the batch_process_seg_skel.sh script, which will call this script for each file to be processed.
# It will not work if run on its own, as it requires command line arguments for the file path and output path. 
# The batch_process_seg_skel.sh script will handle passing the correct file paths to this script for each file to be processed in parallel.


#### USER SPECIFIED INPUTS ####
# specify the location of the PNN-morpho-quant repo
repo_path = '/users/s/r/srhoads/PNN-morpho-quant'

# specify the location of the single image file to process, including the file name and extension
file_type=".tif"
gaus_sigma=2
med_size=8
manual_threshold_cutoff=0.045
auto_threshold_method=None
auto_threshold_adjust=None
auto_multiotsu_middle_to=None
local_threshold_method=None
local_threshold_adjust=None
local_threshold_size=None
local_gaussian_sigma=None
obj_min_diameter=10
obj_method='3D'
hole_min_diameter=0
hole_method='slices'
min_branch_len=0.3



#### IMPORTS ####
import sys
file_path = sys.argv[1]
out_path = sys.argv[2]
sys.path.insert(0, repo_path)
from src.image_processing import batch_PNN_seg_skel_PARALLEL

import warnings
warnings.filterwarnings("ignore")


#### BATCH PROCESS FUNCTION CALL ####
batch_PNN_seg_skel_PARALLEL(file_path=file_path,
                            out_path=out_path,
                            gaus_sigma=gaus_sigma,
                            med_size=med_size,
                            manual_threshold_cutoff=manual_threshold_cutoff,
                            auto_threshold_method=auto_threshold_method,
                            auto_threshold_adjust=auto_threshold_adjust,
                            auto_multiotsu_middle_to=auto_multiotsu_middle_to,
                            local_threshold_method=local_threshold_method,
                            local_threshold_adjust=local_threshold_adjust,
                            local_threshold_size=local_threshold_size,
                            local_gaussian_sigma=local_gaussian_sigma,
                            obj_min_diameter=obj_min_diameter,
                            obj_method=obj_method,
                            hole_min_diameter=hole_min_diameter,
                            hole_method=hole_method,
                            min_branch_len=min_branch_len)