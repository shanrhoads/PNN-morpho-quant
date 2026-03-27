# if using longleaf start here
module load anaconda

# if on a personal computer, make sure you have anaconda installed and start here
conda create -n PNN-morpho python=3.12  # 3.13
conda activate PNN-morpho

pip install ipython ipykernel
pip install napari[all]
pip install skan
pip install git+https://github.com/SCohenLab/infer-subc.git@v2.0.0b1
pip install bioio bioio-ome-tiff bioio-tifffile bioio-lif

# pip install dask-image # only is using dask image reading
# pip install napari-ome-zarr # only if using zarr formatting