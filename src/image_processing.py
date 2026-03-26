import skimage
import pandas as pd
import numpy as np
import skan
from bioio import BioImage
from typing import Union
from pathlib import Path
import time





# function to create skeleton objects from instance segmentation while maintaining the same labels as the instance segmentation
def skeletonize_plus(segmentation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ''' A function that generates punctate objects for the round organelle objects that lack a skeleton.

    As of 7/7/24 the original `skeletonize()` function from `skimage` labels the voxels of the skeleton with the same label,
    however the label serves no purpose to us as of now. 
    The purpose of this function is to: 

    1) Establish punctate objects in the center of organelle objects that for some reason lack a skeleton object (if necessary)
    * these objects are found to be round/spherical in every case this happens, so a punctate is appropriate
    2) Return a skeleton with float labels (due to skan requiring this) corresponding to its location in the original segmentation 
    3) Return a boolean skeleton for input in computation

    Parameters:
    -----------
    segmentation : np.ndarray
        A 3D numpy array containing the instance segmentation of the organelle objects.
    
    Returns:
    -----------
    lab_skel : np.ndarray
        A 3D numpy array containing the skeleton output where each skeleton object is labeled with the same label as the organelle object it belongs to.
    skeleton : np.ndarray
        A 3D boolean numpy array containing the boolean skeleton of the organelle objects.
    '''

    # This is the raw organelle skeleton, some fixing and relabeling has to be done before we can use the skeleton for computation
    skeleton = skimage.morphology.skeletonize(segmentation.astype(bool)).astype(bool)

    # All of the organelle object labels
    all_lab = set(pd.unique(segmentation.ravel()))

    # Applying the segmentation labels to the skeleton
    lab_skel = (skeleton * segmentation).astype(int)

    # Labels present in the skeleton
    skel_lab = set(pd.unique(lab_skel.ravel()))

    # Checker to see if there are any objects without a skeleton
    if all_lab == skel_lab:
        return lab_skel.astype(int), skeleton
    
    else:
        # gets a list of the missing labels
        mis_lab = all_lab - skel_lab

        for label in mis_lab:
            # list of coordinates of the object's voxels
            coord_list = np.nonzero(segmentation == label)

            # The coordinate closest to the middle of the object (due to rounding)
            av_coord = np.round(np.mean(coord_list,axis = 1)).astype(int)

            #checker and result
            if segmentation[tuple(av_coord)] == label:
                lab_skel[tuple(av_coord)] = label
            else:
                print("Apperently the centermost point is not in the object???? :(")
                break
        
        return lab_skel.astype(int), skeleton



# create batch processing function for segmentation/skeletonization
def batch_PNN_seg_skel(file_path: str,
                        file_type: str,
                        out_path: str,
                        gaus_sigma: Union[float, int],
                        med_size: Union[float, int],
                        manual_threshold_cutoff: Union[float, int, None] = None,
                        auto_threshold_method: Union[str, None] = 'otsu',
                        auto_threshold_adjust: Union[float, int, None] = 1,
                        auto_multiotsu_middle_to: Union[str, None] = None,
                        local_threshold_method: Union[str, None] = None,
                        local_threshold_adjust: Union[float, int, None] = 1,
                        local_threshold_size: Union[int, None] = None,
                        local_gaussian_sigma: Union[float, int, None] = None,
                        obj_min_diameter: int = 0,
                        obj_method: str = '3D',
                        hole_min_diameter: int = 0,
                        hole_method: str = '3D',
                        min_branch_len: Union[float, int] = 0):
    """
    This function segments perineuronal nets (PNN) from 3D STED images and generates skeletons for each segmented object.
    It processes all image files in the specified input directory and saves the segmentation/skeleton images to the specified output directory.
    Parameters:
    -----------
    file_path: str
        location of image files
    file_type: str
        file type of intput images (e.g., ".tif")
    out_path: str
        location where segmentation/skeleton images and data tables will be saved; if the path does not exist, it will be created for you
    gaus_sigma: float or int
        sigma value for Gaussian smoothing filter; set to 0 to skip this step
    med_size: float or int
        size value for Median smoothing filter; set to 0 to skip this step
    manual_threshold_cutoff: float, int, or None
        manual threshold value for segmentation; set to None to use global or local automatic thresholding
    auto_threshold_method: str or None
        method for global automatic thresholding
        options: 'otsu', 'li', 'yen', 'isodata', 'triangle', 'minimum', 'mean', 'multiotsu'
        set to None if using manual or local thresholding
    auto_threshold_adjust: float or int or None
        adjustment factor for global automatic thresholding; set to None if using manual or local
        set to 1 for no adjustment, <1 to make the threshold more lenient, >1 to make the threshold more stringent
    auto_multiotsu_middle_to: str or None
        if 'multiotsu' is selected for the auto_threshold_method, specify whether to select middle threshold towards 'foreground' or 'background'
        set to None if not using 'multiotsu' method
    local_threshold_method: str or None
        method for local automatic thresholding
        options: 'otsu', 'mean', 'gaussian'
        set to None if using manual or global automatic thresholding
    local_threshold_adjust: float or int or None
        adjustment factor for local automatic thresholding; set to None if using manual or global
        set to 1 for no adjustment, <1 to make the threshold more lenient, >1 to make the threshold more stringent
    local_threshold_size: int or None
        size value for local thresholding; must be odd integer
        set to None if using manual or global automatic thresholding
    local_gaussian_sigma: float or int or None
        sigma value for local Gaussian thresholding; only used if local_threshold_method is 'gaussian'
        set to None if not using local 'gaussian' method
    obj_min_diameter: int
        minimum acceptable diameter (in voxels) for segmented objects; anything smaller will be removed; set to 0 to skip this step
    obj_method: str
        method for object size filtering; options: 'slices' or '3D'
        'slice' considers objects in each 2D slice independently, while '3D' considers objects across the entire 3D volume
    hole_min_diameter: int
        minimum acceptable diameter (in voxels) for holes in segmented objects; any holes smaller will be removed; set to 0 to skip this step
    hole_method: str
        method for hole size filtering; options: 'slices' or '3D'
        'slice' considers holes in each 2D slice independently, while '3D' considers holes across the entire 3D volume
    min_branch_len: int
        minimum acceptable length (in real work units - usually um) for skeleton branches; set to 0 to skip this step

    Output: N/A; the segmentation and skeletonization files are saved to the specified output directory

    """
    # confirm file paths and files exist
    if not Path.exists(Path(file_path)):
        raise FileExistsError("Input file path does not exist.")
    else:
        file_list = sorted(Path(file_path).glob(f"*{file_type}"))
        if len(file_list) == 0:
            raise FileExistsError(f"Input file path does not have any {file_type} files.")  
        elif len(file_list) > 0:
            print(f"Found {len(file_list)} {file_type} files in {file_path}.")
        
    if not Path.exists(Path(out_path)):
        Path.mkdir(Path(out_path))
        print(f"Output file path not found. Creating: {out_path}")

    # keeping track of processing time
    count=0
    start=time.time()

    # loop through list of images and process
    for f in file_list:
        img_time_start=time.time()
        count=count+1
        if count==1: print("Processing first image:")
        elif count>1 and count<len(file_list): print("Processing next image:")
        elif count==len(file_list): print("Processing last image:")

        # read image
        raw_file = BioImage(str(f))
        raw_image = np.squeeze(raw_file.data)

        voxel_size_ZYX = (raw_file.physical_pixel_sizes.Z, raw_file.physical_pixel_sizes.Y, raw_file.physical_pixel_sizes.X)


        #### segmentation ####
        # rescale
        min_int = raw_image.min()
        max_int = raw_image.max()
        src_seg = (raw_image - min_int + 1e-8) / (max_int - min_int + 1e-8)

        # smooth
        if gaus_sigma != 0:
            src_seg = skimage.filters.gaussian(src_seg, sigma=gaus_sigma)
        if med_size != 0:
            # TODO: make the median filter footprint adjustable based on the image (XY images won't work here)
            fp = skimage.morphology.footprint_rectangle((round(med_size*(voxel_size_ZYX[0]/float(np.max(voxel_size_ZYX)))), 
                                                        round(med_size*(voxel_size_ZYX[1]/float(np.max(voxel_size_ZYX)))), 
                                                        round(med_size*(voxel_size_ZYX[2]/float(np.max(voxel_size_ZYX))))))
            src_seg = skimage.filters.median(src_seg, footprint=fp)

        # segment
        # check thresholding method selection
        if manual_threshold_cutoff is None and auto_threshold_method is None and local_threshold_method is None:
            raise ValueError("No thresholding method selected. Please specify either a manual_threshold_cutoff, an automatic_threshold_method, or a local_threshold_method.")
        elif sum(x is not None for x in [manual_threshold_cutoff, auto_threshold_method, local_threshold_method]) > 1:
            raise ValueError("Multiple thresholding methods selected. Please specify ONLY ONE of the following: manual_threshold_cutoff, automatic_threshold_method, or local_threshold_method.")
        
        # manual thresholding
        elif manual_threshold_cutoff is not None:
            print(f"Applying a manual threshold of {manual_threshold_cutoff}.")
            src_seg = src_seg >= manual_threshold_cutoff
        
        # automatic thresholding
        elif auto_threshold_method is not None:
            print(f"Applying an automatic threshold using {auto_threshold_method} method.")
            if auto_threshold_method == 'otsu':
                thresh_value = skimage.filters.threshold_otsu(src_seg)
            elif auto_threshold_method == 'multiotsu':
                thresholds = skimage.filters.threshold_multiotsu(src_seg, classes=3)
                if auto_multiotsu_middle_to == 'foreground':
                    thresh_value = thresholds[0]  # select the second highest threshold
                elif auto_multiotsu_middle_to == 'background':
                    thresh_value = thresholds[1]   # select the lowest threshold
                else:
                    raise ValueError(f"Unrecognized multiotsu middle to option: {auto_multiotsu_middle_to}")
            elif auto_threshold_method == 'li':
                thresh_value = skimage.filters.threshold_li(src_seg)
            elif auto_threshold_method == 'yen':
                thresh_value = skimage.filters.threshold_yen(src_seg)
            elif auto_threshold_method == 'isodata':
                thresh_value = skimage.filters.threshold_isodata(src_seg)
            elif auto_threshold_method == 'triangle':
                thresh_value = skimage.filters.threshold_triangle(src_seg)
            elif auto_threshold_method == 'minimum':
                thresh_value = skimage.filters.threshold_minimum(src_seg)
            elif auto_threshold_method == 'mean':
                thresh_value = skimage.filters.threshold_mean(src_seg)
            else:
                raise ValueError(f"Unrecognized threshold method: {auto_threshold_method}")
            src_seg = src_seg >= thresh_value*auto_threshold_adjust

        # local thresholding
        elif local_threshold_method is not None:
            print(f"Applying a local threshold using {local_threshold_method} method.")
            src_seg_8bit = (np.round((src_seg / src_seg.max())*255)//2).astype(np.uint8)
            if local_threshold_method == 'otsu':
                footprint = skimage.morphology.ball(local_threshold_size)
                local_threshold = skimage.filters.rank.otsu(src_seg_8bit, footprint)
            elif local_threshold_method == 'mean':
                local_threshold = skimage.filters.threshold_local(src_seg_8bit, block_size=local_threshold_size, method='mean')
            elif local_threshold_method == 'gaussian':
                local_threshold = skimage.filters.threshold_local(src_seg_8bit, block_size=local_threshold_size, method='gaussian', param=local_gaussian_sigma)
            else:
                raise ValueError(f"Unrecognized local threshold method: {local_threshold_method}")
            src_seg = src_seg_8bit >= local_threshold*local_threshold_adjust

        # refine segmentation
        if obj_method == 'slices' or hole_method == 'slices':
            out_seg = np.empty_like(src_seg, dtype=bool)  # allocate once
        else:
            out_seg = None

        if obj_min_diameter > 0:
            if obj_method == 'slices':
                for z in range(src_seg.shape[0]):
                    slice2d = src_seg[z, :, :]
                    filtered2d = skimage.morphology.remove_small_objects(slice2d, min_size=obj_min_diameter**2)
                    out_seg[z, :, :] = filtered2d
            elif obj_method == '3D':
                out_seg = skimage.morphology.remove_small_objects(src_seg, min_size=obj_min_diameter**3)
            else:
                raise SyntaxError("Unrecognized object filter method chosen. Options include: 'slices' or '3D'.")
        else:
            out_seg = src_seg

        if hole_min_diameter > 0:
            if hole_method == 'slices':
                for z in range(out_seg.shape[0]):
                    slice2d = out_seg[z, :, :]
                    filled2d = skimage.morphology.remove_small_holes(slice2d, area_threshold=hole_min_diameter**2, connectivity=8)
                    out_seg[z, :, :] = filled2d
            elif hole_method == '3D':
                out_seg = skimage.morphology.remove_small_holes(out_seg, area_threshold=hole_min_diameter**3, connectivity=26)
            else:
                raise SyntaxError("Unrecognized hole fill method chosen. Options include: 'slices' or '3D'.")
        else:
            out_seg = out_seg

        # create instance segmentation
        PNN_instance_seg = skimage.morphology.label(out_seg)


        #### skeletonization ####
        # create skeleton
        labeled_skel, skeleton = skeletonize_plus(PNN_instance_seg)
        skel_g = skan.Skeleton(labeled_skel, spacing=voxel_size_ZYX, value_is_height=False)

        if min_branch_len > 0:
            # calculate the degree of connectivity for each branch end point
            paths_table = skan.summarize(skel_g, separator='_')
            endpoints_src = skel_g.paths.indices[skel_g.paths.indptr[:-1]]
            endpoints_dst = skel_g.paths.indices[skel_g.paths.indptr[1:] - 1]
            paths_table['deg_src'] = skel_g.degrees[endpoints_src]
            paths_table['deg_dst'] = skel_g.degrees[endpoints_dst]

            # select only branches that have end points and are below the min size
            endpoint_indices = paths_table.index[(paths_table['deg_src'] == 1) | (paths_table['deg_dst'] == 1)]
            short_paths_indices = paths_table.index[paths_table['branch_distance'] < min_branch_len]
            indices_removed = pd.Index(set(endpoint_indices) & set(short_paths_indices))
            pruned_skeleton = skel_g.prune_paths(indices_removed)
        else:
            pruned_skeleton = skel_g

        # save segmentation and skeletonization outputs
        skimage.io.imsave(f"{out_path}/{f.stem}-PNN_instance_seg.tif", PNN_instance_seg, plugin="tifffile")
        del PNN_instance_seg
        skimage.io.imsave(f"{out_path}/{f.stem}-PNN_skeleton.tif", pruned_skeleton.skeleton_image, plugin="tifffile")
        del pruned_skeleton

        # stop time for this image
        end=time.time()
        print(f"Processed {f.name} in {round(end - img_time_start, 2)/60} minutes.")


    # stop total time
    total_end=time.time()
    print(f"Processed {len(file_list)} images in {round(total_end - start, 2)/60} minutes.")