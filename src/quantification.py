import skimage
from typing import Union
from pathlib import Path
import numpy as np
import time
from bioio import BioImage
import pandas as pd
import skan





# function for calculating surface area from regionprops output; based on function from infer-subc
def surface_area_from_props(labels, props, scale: Union[tuple,None]=None):
    surface_areas = np.zeros(len(props["label"]))

    for index, lab in enumerate(props["label"]):
        volume = labels[
            max(props["bbox-0"][index] - 1, 0) : min(props["bbox-3"][index] + 1, labels.shape[0]),
            max(props["bbox-1"][index] - 1, 0) : min(props["bbox-4"][index] + 1, labels.shape[1]),
            max(props["bbox-2"][index] - 1, 0) : min(props["bbox-5"][index] + 1, labels.shape[2]),
        ]
        volume = volume == lab
        if scale is None:
            scale=(1.0,) * labels.ndim
        verts, faces, _normals, _values = skimage.measure.marching_cubes(
            volume,
            method="lewiner",
            spacing=scale,
            level=0,
        )
        surface_areas[index] = skimage.measure.mesh_surface_area(verts, faces)

    return surface_areas


# create batch processing function for quantification
def batch_PNN_quant(dataset_name: str, 
                    raw_file_path: str,
                    raw_file_type: str,
                    seg_skel_path: str,
                    quant_out_path: str):
    """
    This function segments perineuronal nets (PNN) from 3D STED images, creates network graphs of the PNN structure, and quantifies morphological features of the PNN.

    Parameters:
    -----------
    dataset_name: str
        name of the dataset being processed (used for labeling outputs)
    raw_file_path: str
        location of raw image files
    raw_file_type: str
        file type of raw input images (e.g., ".tif")
    seg_skel_path: str
        location where segmentation/skeleton images are saved
    quant_out_path: str
        location where data tables will be saved; if the path does not exist, it will be created for you
    
    Output:
    -------
    summary_tab: pd.DataFrame
        Data table summarizing the quantification results for each PNN object and the entire PNN combined

    Analysis metrics include:
    -------------------------

    """
    # confirm file paths and files exist
    if not Path.exists(Path(raw_file_path)):
        FileExistsError("Input file path does not exist.")
    elif not Path.exists(Path(seg_skel_path)):
        FileExistsError("Segmentation/skeleton file path does not exist.")
    else:
        file_list = sorted(Path(raw_file_path).glob(f"*{raw_file_type}"))
        if len(file_list) == 0:
            FileExistsError(f"Input file path does not have any {raw_file_type} files.")
    
    if not Path.exists(Path(quant_out_path)):
        Path.mkdir(Path(quant_out_path))
        print(f"Making {quant_out_path}")
    elif Path.exists(Path(quant_out_path)):
        # check if output file already exists
        if Path.exists(Path(f"{quant_out_path}/{dataset_name}-PNN_quantification.csv")):
            raise FileExistsError("Quantification output file already exists. Please choose a different quant_out_path or dataset_name to avoid overwriting.")

    # keeping track of processing time
    count=0
    start=time.time()

    # loop through list of images and process
    for f in file_list:
        img_time_start=time.time()
        count=count+1
        if count==1:
            print("Quantifying first image:")
        if count>1:
            print("Quantifying next image:")
        if count==len(file_list):
            print("Quantifying last image:")
        
        # collect paths to the related seg and skel files based on file name
        filez = {name: str(Path(seg_skel_path) / f"{f.stem}-{name}.tif") for name in ['PNN_instance_seg', 'PNN_skeleton']}
        # for name in ['PNN_instance_seg', 'PNN_skeleton']:
        #     filez[name] = str(Path(seg_skel_path) / f"{f.stem}-{name}.tif")

        # read intensity image
        raw_file = BioImage(str(f))
        raw_image = np.squeeze(raw_file.data)

        voxel_size_ZYX = (raw_file.physical_pixel_sizes.Z, raw_file.physical_pixel_sizes.Y, raw_file.physical_pixel_sizes.X)
        rounded_scale = tuple(round(x, 4) for x in voxel_size_ZYX)

        # empty list to collect quantification tables for each image
        quant_tabs = []

        # loop through seg/skel files; read and quantify each
        for name, path in filez.items():
            if not Path.exists(Path(path)):
                raise FileExistsError(f"Expected file not found: {path}")
            
            seg = skimage.io.imread(path)
            
            # for segmentation files
            if name == 'PNN_instance_seg':
                # loop through both types of objects (PNN fragment and whole PNN)
                obj_quant_tabs = []
                obj_dict = {"PNN fragment": seg, 
                            "whole PNN": (seg>0).astype(np.uint8)}
                for obj_type, obj_seg in obj_dict.items():                    
                    properties = ['label', 'bbox', 'centroid', 'num_pixels', 'area', 'equivalent_diameter', 
                                'major_axis_length', 'minor_axis_length', 'extent', 'solidity', 'euler_number',
                                'min_intensity', 'max_intensity', 'mean_intensity', 'intensity_std']

                    props_obj = skimage.measure.regionprops_table(label_image=obj_seg, 
                                                                intensity_image=raw_image,
                                                                properties=properties,
                                                                spacing=voxel_size_ZYX)

                    props_obj_tab = pd.DataFrame(props_obj)
                    props_obj_tab.insert(0, 'object', obj_type)

                    surface_area_tab = pd.DataFrame(surface_area_from_props(obj_seg, props_obj, voxel_size_ZYX), columns=['surface_area'])
                    props_obj_tab.insert(13, 'surface_area', surface_area_tab['surface_area'])
                    obj_quant_tabs.append(props_obj_tab)
                
                # combine both tables and format
                combined_obj_quant = pd.concat(obj_quant_tabs, ignore_index=True)
                combined_obj_quant.insert(0, 'image_name', file_list[file_index].name)
                combined_obj_quant.rename(columns={'area':'volume'}, inplace=True)
                combined_obj_quant['intensity_sum'] = combined_obj_quant['mean_intensity'] * combined_obj_quant['num_pixels']
                combined_obj_quant.insert(15, "SA_to_volume_ratio", combined_obj_quant["surface_area"].div(combined_obj_quant["volume"]))
                rounded_scale = tuple(round(x, 2) for x in voxel_size_ZYX)
                combined_obj_quant.insert(1, "scale", str(rounded_scale))

                quant_tabs.append(combined_obj_quant)

            # for skeleton files
            if name == 'PNN_skeleton':
                skel_g = skan.Skeleton(seg, spacing=voxel_size_ZYX, value_is_height=False)
                del seg  # save memory

                # create initial skeleton branch table and add additional
                paths_table = skan.summarize(skel_g, separator='_')
                paths_table.insert(1, 'branch_id', np.arange(skel_g.n_paths))
                paths_table.insert(2, 'random_branch_id', np.random.default_rng().permutation(skel_g.n_paths))
                if not np.any(skel_g.path_stdev()): 
                    paths_table['skeleton_id'] = skel_g.path_means().astype(int)
                else:
                    raise ValueError("at least one branch came from different organelle objects")
                endpoints_src = skel_g.paths.indices[skel_g.paths.indptr[:-1]]
                endpoints_dst = skel_g.paths.indices[skel_g.paths.indptr[1:] - 1]
                deg_src = skel_g.degrees[endpoints_src]
                deg_dst = skel_g.degrees[endpoints_dst]
                paths_table['deg_src'] = deg_src
                paths_table['deg_dst'] = deg_dst
                paths_table.insert(0, 'combined_skeleton_id', 1)  # assign all branches to skeleton ID 1 for whole image summary

                # loop through both types of skeletons (PNN fragment and whole PNN)
                skel_quant_tabs = []
                skel_dict = {"PNN fragment": 'skeleton_id', 
                             'whole PNN': 'combined_skeleton_id'}
                for skel_type, skel_id in skel_dict.items():
                    skel_sum1 = paths_table[[skel_id, 'branch_id']].groupby(skel_id).agg(['count'])
                    skel_sum2 = paths_table[[skel_id, 'branch_type']].groupby(skel_id).agg(['mean', 'median', 'min', 'max', 'std'])
                    skel_sum3 = paths_table[[skel_id, 'branch_distance', 'euclidean_distance']].groupby(skel_id).agg(['sum', 'mean', 'median', 'min', 'max', 'std'])

                    skel_summary = pd.concat([skel_sum1, skel_sum2, skel_sum3], axis=1)
                    skel_summary.columns = ['_'.join(col).strip() for col in skel_summary.columns.values]
                    skel_summary.reset_index(inplace=True)
                    skel_summary.insert(0, 'image_name', file_list[file_index].name)
                    skel_summary.insert(1, "scale", str(rounded_scale))
                    skel_summary.insert(2, 'object', skel_type)
                    skel_summary.rename(columns={skel_id:'label',
                                                'branch_id_count':'branch_count'}, inplace=True)
                    
                    skel_quant_tabs.append(skel_summary)

                # combine both skel tables
                combined_skel_quant = pd.concat(skel_quant_tabs, axis=0)

                quant_tabs.append(combined_skel_quant)

        # combine regionprops and skel tables
        combo = pd.merge(quant_tabs[0], quant_tabs[1], on= ['image_name', 'scale', 'object', 'label'], how='outer')
        

        # write to csv
        combo.to_csv(f"{quant_out_path}/{dataset_name}-PNN_quantification.csv", index=False, mode='a')
        del combo  # save memory before repeating loop

        print(f"Quantified {f.name} and saved data table. Time taken: {(time.time() - img_time_start)/60} minutes.")

    print(f"Analysis finished!")
    print(f"Quantified {len(file_list)} images in {(time.time() - start)/60} minutes.")
    print("Output saved to:", f"{quant_out_path}/{dataset_name}-PNN_quantification.csv")