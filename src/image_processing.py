import skimage
import pandas as pd
import numpy as np



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