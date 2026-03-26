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
    The output CSV contains one row per quantified object, where object is either:
      - "PNN fragment": individual connected component from instance segmentation
      - "whole PNN": all segmented voxels collapsed into one object

    Final table construction:
      - Segmentation-derived metrics and skeleton-derived metrics are merged with an
        outer join on ['image_name', 'scale', 'object', 'label'].
      - Some fields may be NaN if one side of the merge is missing for a given row.

    Metadata and grouping columns:
      - experiment:
          Parsed from raw_file_path (4th directory from the end).
      - sex:
          Parsed from raw_file_path (3rd directory from the end).
      - replicate:
          Parsed from raw_file_path (2nd directory from the end).
      - genotype:
          Parsed from raw_file_path (last directory).
      - file_path:
          Input raw_file_path used for this quantification run (folder-level provenance).
      - image_name:
          Name of the raw image file being quantified.
      - scale:
          Voxel spacing in (Z, Y, X), stored as a string tuple (rounded).
      - object:
          Object scope for the row: "PNN fragment" or "whole PNN".
      - label:
          Numeric object label ID used as merge key between segmentation and skeleton tables.

    Segmentation geometry columns (regionprops_table):
      - bbox-0, bbox-1, bbox-2:
          Minimum bounding-box indices in Z, Y, X.
      - bbox-3, bbox-4, bbox-5:
          Maximum bounding-box indices in Z, Y, X (exclusive upper bounds).
      - centroid-0, centroid-1, centroid-2:
          Object centroid coordinates in Z, Y, X (scaled by voxel spacing).
      - num_pixels:
          Number of voxels assigned to the object label.
      - volume:
          Physical object volume (renamed from regionprops "area").
      - equivalent_diameter:
          Diameter of a sphere with equivalent object volume.
      - major_axis_length:
          Longest principal-axis length of the fitted object shape.
      - minor_axis_length:
          Shortest principal-axis length of the fitted object shape.
      - extent:
          Fraction of the bounding-box volume occupied by object voxels.
      - solidity:
          Ratio of object volume to convex-hull volume.
      - euler_number:
          Topological descriptor of components/holes/tunnels in the labeled object.

    Intensity columns:
      - min_intensity:
          Minimum raw-image intensity in object voxels.
      - max_intensity:
          Maximum raw-image intensity in object voxels.
      - mean_intensity:
          Mean raw-image intensity in object voxels.
      - intensity_std:
          Standard deviation of raw-image intensity in object voxels.
      - intensity_sum:
          Integrated object intensity, computed as mean_intensity * num_pixels.

    Optional surface area columns (only when include_surface_area=True):
      - surface_area:
          Surface area estimated from marching-cubes mesh using voxel spacing.
      - SA_to_volume_ratio:
          Surface area to volume ratio (surface_area / volume).

    Skeleton summary columns:
      - branch_count:
          Number of skeleton branches assigned to the object.
      - branch_type_mean, branch_type_median, branch_type_min, branch_type_max, branch_type_std:
          Summary statistics of skeleton branch-type codes.
      - branch_distance_sum, branch_distance_mean, branch_distance_median,
        branch_distance_min, branch_distance_max, branch_distance_std:
          Summary statistics of geodesic branch path lengths.
      - euclidean_distance_sum, euclidean_distance_mean, euclidean_distance_median,
        euclidean_distance_min, euclidean_distance_max, euclidean_distance_std:
          Summary statistics of straight-line endpoint-to-endpoint branch distances.

    Branch type code reference (skan):
      - 0: endpoint-to-endpoint branch
      - 1: junction-to-endpoint branch
      - 2: junction-to-junction branch
      - 3: cycle/loop branch

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
            raise FileExistsError("Quantification output file already exists. Please choose a different quant_out_path or dataset_name to avoid overwriting.")
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
        if count>1:
            print("Quantifying next image:")
        if count==len(file_list):
            print("Quantifying last image:")
        
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
            print("imported segmentation successfully")
            
            # for segmentation files
            if name == 'PNN_instance_seg':
                # loop through both types of objects (PNN fragment and whole PNN)
                obj_quant_tabs = []
                obj_dict = {"PNN fragment": seg, 
                            "whole PNN": (seg>0).astype(np.uint8)}
                for obj_type, obj_seg in obj_dict.items():
                    print(f"quant seg: {obj_type}")
                    print("num of objs:", len(np.unique(obj_seg))-1)
                    properties = ['label', 'bbox', 'centroid', 
                                  'num_pixels', 'area', 'equivalent_diameter', 
                                  'major_axis_length', 'minor_axis_length', 'extent', 'solidity', 'euler_number',
                                  'min_intensity', 'max_intensity', 'mean_intensity', 'intensity_std']

                    props_obj = skimage.measure.regionprops_table(label_image=obj_seg, 
                                                                intensity_image=raw_image,
                                                                properties=properties,
                                                                spacing=voxel_size_ZYX)
                    print("finished regionprops")

                    props_obj_tab = pd.DataFrame(props_obj)
                    props_obj_tab.insert(0, 'object', obj_type)

                    if include_surface_area:
                        surface_area_values = surface_area_from_props(obj_seg, props_obj, voxel_size_ZYX)
                        props_obj_tab.insert(13, 'surface_area', surface_area_values)
                    obj_quant_tabs.append(props_obj_tab)
                    print(f"finished seg: {obj_type}")
                
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
                    print(f"quant skel: {skel_type}")
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
                    print(f"finished skel: {skel_type}")

                # combine both skel tables
                combined_skel_quant = pd.concat(skel_quant_tabs, axis=0)

                quant_tabs.append(combined_skel_quant)
                del paths_table, paths_table_subset, skel_quant_tabs, combined_skel_quant  # save memory

        # combine regionprops and skel tables
        combo = pd.merge(quant_tabs[0], quant_tabs[1], on= ['image_name', 'scale', 'object', 'label'], how='outer')
        print("merged together")
        display(combo)

        path_parts = raw_file_path.rsplit("/")[-4:]
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


# summarize quantitative data from batch_PNN_quant outputs; designed to be run after multiple rounds of batch_PNN_quant with the same or different datasets, as long as the output CSV schema is consistent across files
def batch_summarize_quant(out_file_prefix: str,
                          csv_path_list: List[str],
                          out_path: str):
    """
    Parameters:
    -----------
    out_file_prefix: str,
        prefix to append to the summary output file name when saving; a dash is
        automatically added between the prefix and the rest of the file name.
    csv_path_list: List[str],
        list of quantification CSV file paths produced by batch_PNN_quant.
    out_path: str,
        directory where the summary CSV will be written.

    Output table includes:
    ----------------------
    Grouping/index columns (copied to output rows):
        - experiment
        - sex
        - replicate
        - genotype
        - image_name
        - scale
        - object

    Count columns:
        - object_count: Count of rows/objects per group.

    Volume summary columns:
        - volume -> volume_sum, volume_mean, volume_median, volume_std

    Shape/intensity/skeleton metric summary columns:
        For each metric below, output includes: <metric>_mean, <metric>_median, <metric>_std

        - num_pixels
        - equivalent_diameter
        - major_axis_length
        - minor_axis_length
        - extent
        - solidity
        - euler_number
        - min_intensity
        - max_intensity
        - mean_intensity
        - intensity_std
        - intensity_sum
        - branch_count
        - branch_type_mean
        - branch_type_median
        - branch_type_min
        - branch_type_max
        - branch_type_std
        - branch_distance_sum
        - branch_distance_mean
        - branch_distance_median
        - branch_distance_min
        - branch_distance_max
        - branch_distance_std
        - euclidean_distance_sum
        - euclidean_distance_mean
        - euclidean_distance_median
        - euclidean_distance_min
        - euclidean_distance_max
        - euclidean_distance_std

        Optional surface metrics (included only when present in all input CSVs):
            - surface_area -> surface_area_mean, surface_area_median, surface_area_std
            - SA_to_volume_ratio -> SA_to_volume_ratio_mean, SA_to_volume_ratio_median, SA_to_volume_ratio_std

        Notes:
            - Output columns are flattened from pandas MultiIndex by joining with underscores.
            - Function is fail-fast for schema mismatches and missing required columns.

    """

    if not csv_path_list:
        raise ValueError("csv_path_list is empty. Provide at least one quantification CSV path.")

    Path(out_path).mkdir(parents=True, exist_ok=True)
    out_file = Path(out_path) / f"{out_file_prefix}-PNN_quant_summary.csv"

    quant_tabs = []
    for loc in csv_path_list:
        p = Path(loc)
        if not p.exists():
            raise FileNotFoundError(f"Quantification CSV not found: {p}")
        try:
            tab = pd.read_csv(p)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {p}. Error: {e}") from e
        quant_tabs.append(tab)

    # Fail fast on schema mismatch across files.
    ref_cols = set(quant_tabs[0].columns)
    for i, tab in enumerate(quant_tabs[1:], start=1):
        cols = set(tab.columns)
        if cols != ref_cols:
            missing_vs_ref = sorted(ref_cols - cols)
            extra_vs_ref = sorted(cols - ref_cols)
            raise ValueError(
                "Input CSV schema mismatch detected at index "
                f"{i}. Missing columns: {missing_vs_ref}. Extra columns: {extra_vs_ref}."
            )

    sa_included = [("surface_area" in tab.columns and "SA_to_volume_ratio" in tab.columns) for tab in quant_tabs]
    if not all(sa_included) and any(sa_included):
        raise ValueError(
            "Surface-area columns are inconsistently present across input CSV files. "
            "Use a consistent include_surface_area setting during batch_PNN_quant runs."
        )
    include_sa = all(sa_included)

    raw_df = pd.concat(quant_tabs, axis=0, join='outer', ignore_index=True)

    ###################
    # summary stat group
    ###################
    # Include sex and scale to prevent cross-group mixing.
    group_by = ["experiment", "sex", "replicate", "genotype", "image_name", "scale", "object"]
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

    required_cols = set(group_by + ['volume'] + sharedcolumns)
    missing_required = sorted(required_cols - set(raw_df.columns))
    if missing_required:
        raise ValueError(
            "Missing required columns for summarization: "
            f"{missing_required}"
        )

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