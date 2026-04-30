#### GENERAL INFO ####
# This script runs the segmentation and skeletonization of A FOLDER OF IMAGES based on the input parameters specified below.
# It is intended to be run on its own, and will process all files in the specified input folder that match the specified file type.


#### USER SPECIFIED INPUTS ####
# specify the location of the PNN-morpho-quant repo
repo_path = '/users/s/r/srhoads/PNN-morpho-quant'

# inputs for batch_PNN_seg_skel function
file_path="/users/s/r/srhoads/PNN-morpho_Hayli-pilot-data/Male/Pair_5/3D_STED/WT"
file_type=".tif"
out_path="/users/s/r/srhoads/PNN-morpho_Hayli-pilot-data/Male/Pair_5/3D_STED/WT_seg-skel_20260422-py-test"
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
sys.path.insert(0, repo_path)
from src.image_processing import batch_PNN_seg_skel



#### BATCH PROCESS FUNCTION CALL ####
batch_PNN_seg_skel(file_path=file_path,
                    file_type=file_type,
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