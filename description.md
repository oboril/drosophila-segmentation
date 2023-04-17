## Image is resized
Resized to 1px / um, this reduces image size (faster processing), the inaccuracy caused by this is negligible

```
resizing:
    resize: [bool] # whether to resize image, recommended True
    interpolation_order: [0-5] # order of interpolating polynomial, e.g. 3 corresponds to cubic interpolation, 1 (linear) seems to be just fine
    input_resolution: # voxel size in microns of the input image, this is compared with image metadata. If values do not match or metadata is missing, program will print warning message
        x: [num]
        y: [num]
        z: [num]
```

## Finding nuclei
The channel with nuclei is blurred, thresholded, and distance transform is used to locate centers of large blobs -> nuclei.

```
nuclei:
  center:
    blur: 5 # radius of gaussian blur for finding nuclei
    threshold: 0.3 # relative intensity (1 would select only brightest pixels) for finding nuclei
    min_radius: 7 # minimal radius of nuclei CENTRE in um
    dilation: 3 # how much can the nucleus centre be dilated before watershed
```

## Segmenting nuclei
Some minor preprocessing is done in hope to make the segmentation more reliable, morphological closing is applied to the raw channel. 

The centers of nuclei are then used as seeds in watershed segmentation.

```
nuclei: 
  threshold: 0.1 # relative intensity (1 would select only brightest pixels) for segmenting nuclei
  closing_radius: 2 # radius of morphological closing applied to nucleus channel
```

## Postprocessing segmented nuclei
Once the nuclei are segmented, it is possible to measure their volumes, locations, etc.

If more nuclei than expected are located, the nuclei are sorted by solidity (ratio of nucleus volume and bounding-box volume) and least solid nuclei are excluded.

```
nuclei:
    max_count: 15 # maximum number of nuclei
```

## Cytoskelet preprocessing
The lower z-layers are somewhat dimmer, there is a very crude correction step which enhaces voxel intensity based on intensity of above voxels.

```
cytoplasm:
  z_equalization_coef: 15 # smaller value amplifies bottom of the image more
```

Additional preprocessing includes minor blurring and morphological closing.

```
cytoplasm:
  nucleus_cell_gap: # minimum and maximum distance between nucleus surface and cell surface
    min: 5
    max: 80
  closing_iters: 3 # iterations and radius for morphological closing of the cytoskelet channel before watershed segmentation
  closing_radius: 3
```

## Cytoskelet segmentation
The segmented nuclei are used as seeds for watershed segmentation.

This step is the most error-prone, because sometimes there are gaps in the cytoskelet channel and part of cell might be incorrectly assigned to be part of another cell.

## Cytoskelet postprocessing
The volume of cytoskelet is measured.