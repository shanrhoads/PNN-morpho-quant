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


# batch processing function for quantification of PNN morphology (for running a folder of data sequentially; not parallelized)
def batch_PNN_quant(file_out_prefix: str,
                     raw_file_path: str,
                     raw_file_type: str,
                     seg_skel_path: str,
                     quant_out_path: str,
                     include_surface_area: bool = True):
    """
    This function segments perineuronal nets (PNN) from 3D STED images, creates network graphs of the PNN structure, and quantifies morphological features of the PNN.

    Parameters:
    -----------
    file_out_prefix: str
        prefix to append to the quantification output file name when saving; a dash will be automatically added between the prefix and the rest of the file name 
        this allows for multiple unique rounds of quantification to be done and saved to the same location, if necessary.
        A good example of a prefix is the date of quantification and a brief note about the parameters used (e.g., "20260309_test")
    raw_file_path: str
        location of raw image files
    raw_file_type: str
        file type of raw input images (e.g., ".tif")
    seg_skel_path: str
        location where segmentation/skeleton images are saved
    quant_out_path: str
        location where data tables will be saved; if the path does not exist, it will be created for you
    include_surface_area: bool
        whether to include surface area in the quantification results; this is the most time intensive analysis step

    Output:
    -------
    summary_tab: pd.DataFrame
        Data table summarizing the quantification results for each PNN object and the entire PNN combined
        This table is automatically saved to a .csv file and not returned by the function

    Analysis metrics include:
    -------------------------

    """
    # confirm file paths and files exist
    if not Path.exists(Path(raw_file_path)):
        raise FileExistsError("Input file path does not exist.")
    elif not Path.exists(Path(seg_skel_path)):
        raise FileExistsError("Segmentation/skeleton file path does not exist.")
    else:
        file_list = sorted(Path(raw_file_path).glob(f"*{raw_file_type}"))
        if len(file_list) == 0:
            raise FileExistsError(f"Input file path does not have any {raw_file_type} files.")

    
    if not Path.exists(Path(quant_out_path)):
        Path(quant_out_path).mkdir(parents=True, exist_ok=True)
        print(f"Making {quant_out_path}")
    elif Path.exists(Path(quant_out_path)):
        # check if output file already exists
        if Path.exists(Path(f"{quant_out_path}/{file_out_prefix}-PNN_quantification.csv")):
            raise FileExistsError("Quantification output file already exists. Please choose a different quant_out_path or file_out_prefix to avoid overwriting.")
    out_csv = Path(quant_out_path) / f"{file_out_prefix}-PNN_quantification.csv"
    first_write = True

    # keeping track of processing time
    count=0
    start=time.time()

    # loop through list of images and process
    for f in file_list:
        img_time_start=time.time()
        count=count+1
        if count==1:
            print("Quantifying first image:")
        elif count==len(file_list):
            print("Quantifying last image:")
        else:
            print("Quantifying next image:")
        
        # collect paths to the related seg and skel files based on file name
        filez = {name: str(Path(seg_skel_path) / f"{f.stem}-{name}.tif") for name in ['PNN_instance_seg', 'PNN_skeleton']}


        # read intensity image
        raw_file = BioImage(str(f))
        raw_image = np.squeeze(raw_file.data)

        voxel_size_ZYX = (raw_file.physical_pixel_sizes.Z, raw_file.physical_pixel_sizes.Y, raw_file.physical_pixel_sizes.X)
        rounded_scale = tuple(round(x, 4) for x in voxel_size_ZYX)
        print("Quantiative metrics will be scaled according to voxel size (ZYX):", rounded_scale)

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
                    properties = ['label', 'bbox', 'centroid', 
                                  'num_pixels', 'area', 'equivalent_diameter', 
                                  'major_axis_length', 'minor_axis_length', 'extent', 'solidity', 'euler_number',
                                  'min_intensity', 'max_intensity', 'mean_intensity', 'intensity_std']

                    props_obj = skimage.measure.regionprops_table(label_image=obj_seg, 
                                                                intensity_image=raw_image,
                                                                properties=properties,
                                                                spacing=voxel_size_ZYX)

                    props_obj_tab = pd.DataFrame(props_obj)
                    props_obj_tab.insert(0, 'object', obj_type)

                    if include_surface_area:
                        surface_area_values = surface_area_from_props(obj_seg, props_obj, voxel_size_ZYX)
                        props_obj_tab.insert(13, 'surface_area', surface_area_values)
                    obj_quant_tabs.append(props_obj_tab)
                
                # combine both tables and format
                combined_obj_quant = pd.concat(obj_quant_tabs, ignore_index=True)
                combined_obj_quant.insert(0, 'image_name', f.name)
                combined_obj_quant.rename(columns={'area':'volume'}, inplace=True)
                combined_obj_quant['intensity_sum'] = combined_obj_quant['mean_intensity'] * combined_obj_quant['num_pixels']
                if include_surface_area:
                    combined_obj_quant.insert(15, "SA_to_volume_ratio", combined_obj_quant["surface_area"].div(combined_obj_quant["volume"]))
                rounded_scale = tuple(round(x, 2) for x in voxel_size_ZYX)
                combined_obj_quant.insert(1, "scale", str(rounded_scale))

                quant_tabs.append(combined_obj_quant)
                del obj_quant_tabs, combined_obj_quant, seg  # save memory

            # for skeleton files
            if name == 'PNN_skeleton':
                skel_g = skan.Skeleton(seg, spacing=voxel_size_ZYX, value_is_height=False)
                del seg  # save memory

                # create initial skeleton branch table and add additional
                paths_table = skan.summarize(skel_g, separator='_')
                paths_table.insert(1, 'branch_id', np.arange(skel_g.n_paths))
                if not np.any(skel_g.path_stdev()): 
                    paths_table['skeleton_id'] = skel_g.path_means().astype(int)
                else:
                    raise ValueError("at least one branch came from different organelle objects")

                paths_table.insert(0, 'combined_skeleton_id', 1)  # assign all branches to skeleton ID 1 for whole image summary
                del skel_g  # save memory

                # loop through both types of skeletons (PNN fragment and whole PNN)
                skel_quant_tabs = []
                paths_table_subset = paths_table[['skeleton_id', 'combined_skeleton_id', 'branch_id', 'branch_type', 'branch_distance', 'euclidean_distance']]
                skel_dict = {"PNN fragment": 'skeleton_id', 
                             'whole PNN': 'combined_skeleton_id'}
                for skel_type, skel_id in skel_dict.items():
                    skel_summary = paths_table_subset[[skel_id, 'branch_id', 'branch_type', 
                                                       'branch_distance', 'euclidean_distance']].groupby(skel_id).agg({'branch_id':'count',
                                                                                                                       'branch_type':['mean', 'median', 'min', 'max', 'std'],
                                                                                                                       'branch_distance':['sum', 'mean', 'median', 'min', 'max', 'std'],
                                                                                                                       'euclidean_distance':['sum', 'mean', 'median', 'min', 'max', 'std']})

                    skel_summary.columns = ['_'.join(col).strip() for col in skel_summary.columns.values]
                    skel_summary.reset_index(inplace=True)
                    skel_summary.insert(0, 'image_name', f.name)
                    skel_summary.insert(1, "scale", str(rounded_scale))
                    skel_summary.insert(2, 'object', skel_type)
                    skel_summary.rename(columns={skel_id:'label',
                                                'branch_id_count':'branch_count'}, inplace=True)
                    
                    skel_quant_tabs.append(skel_summary)

                # combine both skel tables
                combined_skel_quant = pd.concat(skel_quant_tabs, axis=0)

                quant_tabs.append(combined_skel_quant)
                del paths_table, paths_table_subset, skel_quant_tabs, combined_skel_quant  # save memory

        # combine regionprops and skel tables
        combo = pd.merge(quant_tabs[0], quant_tabs[1], on= ['image_name', 'scale', 'object', 'label'], how='outer')

        path_parts = list(Path(raw_file_path).parts[-4:])
        if len(path_parts) < 4:
            path_parts = [None] * (4 - len(path_parts)) + path_parts
        for i, label in enumerate(['experiment', 'sex', 'replicate', 'genotype']):
            combo.insert(i, label, path_parts[i])
        combo.insert(4, 'file_path', raw_file_path)

        # write to csv
        combo.to_csv(out_csv, index=False, mode='w' if first_write else 'a', header=first_write)
        first_write = False  # after first write, set to False so that header is not written again in next loop iteration
        del combo, quant_tabs, raw_image, raw_file, filez  # save memory before repeating loop

        print(f"Quantified {f.name} and saved data table. Time taken: {(time.time() - img_time_start)/60} minutes.")

    print(f"Analysis finished!")
    print(f"Quantified {len(file_list)} images in {(time.time() - start)/60} minutes.")
    print("Output saved as:", f"{quant_out_path}/{file_out_prefix}-PNN_quantification.csv")


def batch_PNN_quant_PARALLEL(file_out_prefix: str,
                            raw_file_path: str,
                            seg_skel_path: str,
                            quant_out_path: str,
                            include_surface_area: bool = True):
    """
    This function segments perineuronal nets (PNN) from 3D STED images, creates network graphs of the PNN structure, and quantifies morphological features of the PNN.

    Parameters:
    -----------
    file_out_prefix: str
        prefix to append to the quantification output file name when saving; a dash will be automatically added between the prefix and the rest of the file name 
        this allows for multiple unique rounds of quantification to be done and saved to the same location, if necessary.
        A good example of a prefix is the date of quantification and a brief note about the parameters used (e.g., "20260309_test")
    raw_file_path: str
        path to the single raw image file to be processed in this function call; this function is intended to be run in parallel across multiple files using the batch_process_quant.sh script, which will call this function for each file to be processed
    seg_skel_path: str
        location where segmentation/skeleton images are saved
    quant_out_path: str
        location where data tables will be saved; if the path does not exist, it will be created for you
    include_surface_area: bool
        whether to include surface area in the quantification results; this is the most time intensive analysis step

    Output:
    -------
    summary_tab: pd.DataFrame
        Data table summarizing the quantification results for each PNN object and the entire PNN combined
        This table is automatically saved to a .csv file and not returned by the function

    Analysis metrics include:
    -------------------------

    """
    # see if file path is a single file or a directory, if directory, raise an error
    if Path(raw_file_path).is_file():
        print(f"Found file: {raw_file_path}.")
    elif Path(raw_file_path).is_dir():
        raise NotADirectoryError("Input file path is a directory. Please specify a single file path for processing.")
    else:
        raise FileExistsError(f"Input file path does not exist: {raw_file_path}")

    # # confirm file paths and files exist
    # if not Path.exists(Path(raw_file_path)):
    #     raise FileExistsError("Input file path does not exist.")
    # elif not Path.exists(Path(seg_skel_path)):
    #     raise FileExistsError("Segmentation/skeleton file path does not exist.")
    # else:
    #     file_list = sorted(Path(raw_file_path).glob(f"*{raw_file_type}"))
    #     if len(file_list) == 0:
    #         raise FileExistsError(f"Input file path does not have any {raw_file_type} files.")
    
    if not Path(quant_out_path).exists():
        Path(quant_out_path).mkdir(parents=True, exist_ok=True)
        print(f"Making {quant_out_path}")

    # keeping track of processing time
    # count=0
    # start=time.time()

    # loop through list of images and process
    # for f in file_list:
    f = Path(raw_file_path)
    out_csv = Path(quant_out_path) / f"{file_out_prefix}-{f.stem}-PNN_quantification.csv"

    img_time_start=time.time()
    # count=count+1
    # if count==1:
    #     print("Quantifying first image:")
    # elif count>1:
    #     print("Quantifying next image:")
    # elif count==len(file_list):
    #     print("Quantifying last image:")
    
    # collect paths to the related seg and skel files based on file name
    filez = {name: str(Path(seg_skel_path) / f"{f.stem}-{name}.tif") for name in ['PNN_instance_seg', 'PNN_skeleton']}


    # read intensity image
    raw_file = BioImage(str(f))
    raw_image = np.squeeze(raw_file.data)

    voxel_size_ZYX = (raw_file.physical_pixel_sizes.Z, raw_file.physical_pixel_sizes.Y, raw_file.physical_pixel_sizes.X)
    rounded_scale = tuple(round(x, 4) for x in voxel_size_ZYX)
    print("Quantiative metrics will be scaled according to voxel size (ZYX):", rounded_scale)

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
                properties = ['label', 'bbox', 'centroid', 
                                'num_pixels', 'area', 'equivalent_diameter', 
                                'major_axis_length', 'minor_axis_length', 'extent', 'solidity', 'euler_number',
                                'min_intensity', 'max_intensity', 'mean_intensity', 'intensity_std']

                props_obj = skimage.measure.regionprops_table(label_image=obj_seg, 
                                                            intensity_image=raw_image,
                                                            properties=properties,
                                                            spacing=voxel_size_ZYX)

                props_obj_tab = pd.DataFrame(props_obj)
                props_obj_tab.insert(0, 'object', obj_type)

                if include_surface_area:
                    surface_area_values = surface_area_from_props(obj_seg, props_obj, voxel_size_ZYX)
                    props_obj_tab.insert(13, 'surface_area', surface_area_values)
                obj_quant_tabs.append(props_obj_tab)
            
            # combine both tables and format
            combined_obj_quant = pd.concat(obj_quant_tabs, ignore_index=True)
            combined_obj_quant.insert(0, 'image_name', f.name)
            combined_obj_quant.rename(columns={'area':'volume'}, inplace=True)
            combined_obj_quant['intensity_sum'] = combined_obj_quant['mean_intensity'] * combined_obj_quant['num_pixels']
            if include_surface_area:
                combined_obj_quant.insert(15, "SA_to_volume_ratio", combined_obj_quant["surface_area"].div(combined_obj_quant["volume"]))
            rounded_scale = tuple(round(x, 2) for x in voxel_size_ZYX)
            combined_obj_quant.insert(1, "scale", str(rounded_scale))

            quant_tabs.append(combined_obj_quant)
            del obj_quant_tabs, combined_obj_quant, seg  # save memory

        # for skeleton files
        if name == 'PNN_skeleton':
            skel_g = skan.Skeleton(seg, spacing=voxel_size_ZYX, value_is_height=False)
            del seg  # save memory

            # create initial skeleton branch table and add additional
            paths_table = skan.summarize(skel_g, separator='_')
            paths_table.insert(1, 'branch_id', np.arange(skel_g.n_paths))
            if not np.any(skel_g.path_stdev()): 
                paths_table['skeleton_id'] = skel_g.path_means().astype(int)
            else:
                raise ValueError("at least one branch came from different organelle objects")

            paths_table.insert(0, 'combined_skeleton_id', 1)  # assign all branches to skeleton ID 1 for whole image summary
            del skel_g  # save memory

            # loop through both types of skeletons (PNN fragment and whole PNN)
            skel_quant_tabs = []
            paths_table_subset = paths_table[['skeleton_id', 'combined_skeleton_id', 'branch_id', 'branch_type', 'branch_distance', 'euclidean_distance']]
            skel_dict = {"PNN fragment": 'skeleton_id', 
                            'whole PNN': 'combined_skeleton_id'}
            for skel_type, skel_id in skel_dict.items():
                skel_summary = paths_table_subset[[skel_id, 'branch_id', 'branch_type', 
                                                    'branch_distance', 'euclidean_distance']].groupby(skel_id).agg({'branch_id':'count',
                                                                                                                    'branch_type':['mean', 'median', 'min', 'max', 'std'],
                                                                                                                    'branch_distance':['sum', 'mean', 'median', 'min', 'max', 'std'],
                                                                                                                    'euclidean_distance':['sum', 'mean', 'median', 'min', 'max', 'std']})

                skel_summary.columns = ['_'.join(col).strip() for col in skel_summary.columns.values]
                skel_summary.reset_index(inplace=True)
                skel_summary.insert(0, 'image_name', f.name)
                skel_summary.insert(1, "scale", str(rounded_scale))
                skel_summary.insert(2, 'object', skel_type)
                skel_summary.rename(columns={skel_id:'label',
                                            'branch_id_count':'branch_count'}, inplace=True)
                
                skel_quant_tabs.append(skel_summary)

            # combine both skel tables
            combined_skel_quant = pd.concat(skel_quant_tabs, axis=0)

            quant_tabs.append(combined_skel_quant)
            del paths_table, paths_table_subset, skel_quant_tabs, combined_skel_quant  # save memory

    # combine regionprops and skel tables
    combo = pd.merge(quant_tabs[0], quant_tabs[1], on= ['image_name', 'scale', 'object', 'label'], how='outer')

    parent_parts = Path(raw_file_path).parent.parts
    if len(parent_parts) < 4:
        raise ValueError(
            f"Expected at least 4 parent directories in raw_file_path to derive "
            f"experiment/sex/replicate/genotype metadata, got: {raw_file_path}"
        )
    path_parts = parent_parts[-4:]
    for i, label in enumerate(['experiment', 'sex', 'replicate', 'genotype']):
        combo.insert(i, label, path_parts[i])
    combo.insert(4, 'file_path', raw_file_path)

    # write to csv - one file per image to avoid concurrent write conflicts in parallel jobs
    combo.to_csv(out_csv, index=False)
    # del combo, quant_tabs, raw_image, raw_file, filez  # save memory before repeating loop

    print(f"Quantified {f.name}. Time taken: {(time.time() - img_time_start)/60} minutes.")
    print("Output saved to:", str(out_csv))

    # print(f"Analysis finished!")
    # print(f"Quantified {len(file_list)} images in {(time.time() - start)/60} minutes.")
    # print("Output saved as:", f"{quant_out_path}/{file_out_prefix}-PNN_quantification.csv")


def batch_summarize_quant(out_file_prefix: str,
                          csv_path_list: List[str],
                          out_path: str):
    """
    out_file_prefix: str,
        prefix to append to the summary output file name when saving; a dash will be automatically added between the prefix and the rest of the file name 
        this allows for multiple unique rounds of summarization to be done and saved to the same location, if necessary.
        A good example of a prefix is the date of summarization and a brief note about the parameters used (e.g., "20260312_test")
    csv_path_list: List[str],
        A list of path for the .csv files to analyze. These should be the output quantification tables from the batch_PNN_quant function, 
        and should all have the same column structure.
    out_path: str,
        A path string where the summary data file will be output to

    """

    Path(out_path).mkdir(parents=True, exist_ok=True)
    out_file = Path(out_path) / f"{out_file_prefix}-PNN_quant_summary.csv"

    ###################
    # Read in the csv files and combine
    ###################
    quant_tabs = []
    for loc in csv_path_list:
        quant_tabs.append(pd.read_csv(Path(loc)))

    # assess if all tables have surface area included or not, and if they are consistent with each other
    sa_included = [("surface_area" in tab.columns and "SA_to_volume_ratio" in tab.columns) for tab in quant_tabs]
    include_sa = all(sa_included)

    if not include_sa:
        print("Not all tables have surface area included. Surface area will not be included in the summary table.")
        quant_tabs = [tab.drop(columns=['surface_area', 'SA_to_volume_ratio'], errors='ignore') for tab in quant_tabs]
    else:
        print("All tables have surface area included. Surface area will be included in the summary table.")

    raw_df = pd.concat(quant_tabs, axis=0, join='outer', ignore_index=True)

    ###################
    # summary stat group
    ###################
    group_by = ["experiment", "replicate", "genotype", "image_name", "object"]
    sharedcolumns = [
        "num_pixels", "equivalent_diameter", "major_axis_length", "minor_axis_length", "extent", "solidity", "euler_number",
        "min_intensity", "max_intensity", "mean_intensity", "intensity_std", "intensity_sum",
        "branch_count", "branch_type_mean", "branch_type_median", "branch_type_min", "branch_type_max", "branch_type_std",
        "branch_distance_sum", "branch_distance_mean", "branch_distance_median", "branch_distance_min", "branch_distance_max", "branch_distance_std",
        "euclidean_distance_sum", "euclidean_distance_mean", "euclidean_distance_median", "euclidean_distance_min", "euclidean_distance_max", "euclidean_distance_std"
    ]
    if include_sa:
        sharedcolumns += ["surface_area", "SA_to_volume_ratio"]
    ag_func_standard = ['mean', 'median', 'std']

    ###################
    # summarize measurements
    ###################
    tab1 = raw_df.groupby(group_by, dropna=False).size().to_frame('object_count')
    tab2 = raw_df.groupby(group_by, dropna=False)[['volume']].agg(['sum'] + ag_func_standard)
    tab3 = raw_df.groupby(group_by, dropna=False)[sharedcolumns].agg(ag_func_standard)

    summary_tab = pd.concat([tab1, tab2, tab3], axis=1).reset_index()

    summary_tab.columns = [
        '_'.join([str(x) for x in col if str(x) != '']).strip('_')
        if isinstance(col, tuple) else str(col)
        for col in summary_tab.columns.to_flat_index()
    ]

    summary_tab.to_csv(out_file, index=False)
    print(f"Summary table saved to: {out_file}")