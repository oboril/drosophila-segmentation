# Automatic pipeline for segmentation of nuclei and cytoplasm
This python script automatically segments nuclei and cytoplasm from drosophila embryo.

## How to use
Install Python, required packages are: `scikit-image, tifffile, scipy, numpy`.

Prepare `.tif` file containing 3D scans with two channels - channel 1 contains cytoskeleton, channel 2 contains nuclei.

Change the desired input image and output folder in `config.json`.

Run `python segment.py`.

### Output
The output folder will contain `summary.csv` - summary of positions and volumes of all nuclei and cells.

The output folder will also contain `.tif` files with processed cytoskeleton, segmented nuclei, and individual segmented cells.

### Resolution
The images are automatically rescaled to 1x1x1 um voxels - I think this is good balance between speed and accuracy. If the input `.tif` file does not contain information about resolution, this will cause errors.

## Visualization
The 3D stacks can be visualized using 3D Viewer in ImageJ, but I personally prefer UCSF Chimera.

## Troubleshooting
The program is not very sophisticated, and will likely produce some errors. This section will hopefully show how to solve the common ones.

### Most runtime errors
Some input is probably not as expected - check the input image, config, etc.

### No nuclei are found
If the image is as expected, it is best to save some intermediary images and try to tweak some constants in `config.json`.

### More than 15 nuclei are found
This is likely to happen, especially if the image contains multiple embrya. The number of nuclei can be limited in config, and the smaller nuclei are then ommited.

This can, however, also ommit some nuclei in the embryo of interest. At this point, it is best to filter the nuclei and cells manually after segmentation.

