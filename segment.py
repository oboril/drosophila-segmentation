print("PROGRAM STARTED")

import os
from time import time
import json

import utils

import numpy as np
from skimage.filters import gaussian
from scipy.ndimage import label, distance_transform_edt, binary_dilation
from skimage.morphology import closing, ball, dilation
from skimage.segmentation import watershed

print("Loading config")

config = json.load(open("config.json", "r"))

PATH = config["input-image"]
NUCLEI_CENTER_BLUR = config["nuclei"]["center"]["blur"]
NUCLEI_CENTER_THRESHOLD = config["nuclei"]["center"]["threshold"]
NUCLEI_CENTER_MIN_RADIUS = config["nuclei"]["center"]["min-radius"]

NUCLEI_THRESHOLD = config["nuclei"]["threshold"]
NUCLEI_CLOSING_RADIUS = config["nuclei"]["closing-radius"]
MAX_NUCLEI = config["nuclei"]["max-count"]

CYTOSKELET_EQUALIZATION_COEF = config["cytoplasm"]["z-equalization-coef"]
MAX_CYTOPLASM_GAP = config["cytoplasm"]["nucleus-cell-gap"]["max"]
MIN_CYTOPLASM_GAP = config["cytoplasm"]["nucleus-cell-gap"]["min"]
CYTOSKELET_CLOSING_ITERS = config["cytoplasm"]["closing-iters"]
CYTOSKELET_CLOSING_RADIUS = config["cytoplasm"]["closing-radius"]

SAVE_PATH = config["output-folder"]

start = time()

if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)

print(f"Loading image {PATH}")
img, scale = utils.load_image(PATH)

print("Resizing image to 1um/voxel")
img = utils.resize_1um(img, scale)

nuclei = img[:,:,:,1]
cytoskelet = img[:,:,:,0]

print("Finding centers of nuclei")
blurred = gaussian(nuclei, NUCLEI_CENTER_BLUR)

threshold = np.min(blurred)*(1-NUCLEI_CENTER_THRESHOLD) + np.max(blurred)*NUCLEI_CENTER_THRESHOLD
mask = blurred > threshold

dist = distance_transform_edt(mask)

mask = np.where(dist > NUCLEI_CENTER_MIN_RADIUS, 1, 0)

labelled, num_areas = label(mask)

print(f"Found {num_areas} nuclei")

print("Preparing nuclei for segmentation")
nuclei = (nuclei*255/np.max(nuclei)).astype(np.uint8)
nuclei = closing(nuclei, ball(NUCLEI_CLOSING_RADIUS))

threshold = np.min(nuclei)*(1-NUCLEI_THRESHOLD) + np.max(nuclei)*NUCLEI_THRESHOLD
mask = nuclei > threshold

dist = distance_transform_edt(mask)

print("Segmenting nuclei")
segmented = watershed(-dist, labelled, mask=mask)

print("Processing segmented nuclei")
volumes = []
centres = []
x = np.arange(segmented.shape[0])
y = np.arange(segmented.shape[1])
z = np.arange(segmented.shape[2])
x,y,z = np.meshgrid(x,y,z, indexing="ij")
for nucleus in range(1,num_areas+1):
  mask_i = (segmented==nucleus)
  volume_i = np.count_nonzero(mask_i)
  x_i = np.sum(np.where(mask_i, x, 0))//volume_i
  y_i = np.sum(np.where(mask_i, y, 0))//volume_i
  z_i = np.sum(np.where(mask_i, z, 0))//volume_i
  volumes.append(volume_i)
  centres.append((x_i,y_i,z_i))

indeces = range(1, len(volumes)+1)

volumes, centres, indeces = zip(*sorted(zip(volumes, centres, indeces), reverse=True))

if len(volumes) > MAX_NUCLEI:
  print(f"Limiting the nuclei to {MAX_NUCLEI} largest out of {len(volumes)}")
  volumes = volumes[:MAX_NUCLEI]
  centres = centres[:MAX_NUCLEI]
  indeces = indeces[:MAX_NUCLEI]

ordered_segmented = np.zeros(segmented.shape, dtype=np.uint8)

print("Sorting nuclei by volume")
for new_idx, old_idx in enumerate(indeces):
  new_idx += 1
  ordered_segmented = np.where(segmented == old_idx, new_idx, ordered_segmented)

print("Saving segmented nuclei")

with open(SAVE_PATH + "/nuclei.csv", "w+") as f:
  f.write("{},{},{},{},{}\n".format("N", "x", "y", "z", "Volume"))
  for idx,(v, (x,y,z)) in enumerate(zip(volumes, centres)):
    f.write("{},{},{},{},{}\n".format(idx+1, x, y, z, v))

utils.tf.imwrite(SAVE_PATH + "/segmented_nuclei.tif", np.where(ordered_segmented.transpose([2,1,0]) > 0, 255, 0).astype(np.uint8))

print(f"NUCLEI SEGMENTED (elapsed {time()-start:0.0f} s)")

print("Equalizing cytoskelet channel along z axis")
cytoskelet = cytoskelet.astype(float)/np.max(img)
equalized = utils.equalize_depth_intensity(cytoskelet, CYTOSKELET_EQUALIZATION_COEF)

print("Preparing seeds for cytoplasm segmentation")
seed_points = dilation(ordered_segmented, ball(MIN_CYTOPLASM_GAP))

not_outside = ordered_segmented > 0

not_outside = binary_dilation(not_outside, iterations = MAX_CYTOPLASM_GAP)

outside_idx = len(volumes)+1
seed_points = np.where(not_outside, seed_points, outside_idx)

print("Preprocessing cytoskelet channel")
cytoskelet = gaussian(cytoskelet, 1)
cytoskelet = (cytoskelet*255/np.max(cytoskelet)).astype(np.uint8)
for _ in range(CYTOSKELET_CLOSING_ITERS):
  cytoskelet = closing(cytoskelet, ball(CYTOSKELET_CLOSING_RADIUS))
cytoskelet = gaussian(cytoskelet, 1)
cytoskelet = (cytoskelet*255/np.max(cytoskelet)).astype(np.uint8)

print("Saving preprocessed cytoskelet channel")
utils.tf.imwrite(SAVE_PATH + "/preprocessed_cytoskelet.tif", cytoskelet)

print("Segmenting cytoplasm")
segmented = watershed(cytoskelet, seed_points)

cyt_volumes = []
cyt_centres = []
x = np.arange(segmented.shape[0])
y = np.arange(segmented.shape[1])
z = np.arange(segmented.shape[2])
x,y,z = np.meshgrid(x,y,z, indexing="ij")
for cell in range(1,len(volumes)+1):
  mask_i = (segmented==cell)
  volume_i = np.count_nonzero(mask_i)
  x_i = np.sum(np.where(mask_i, x, 0))//volume_i
  y_i = np.sum(np.where(mask_i, y, 0))//volume_i
  z_i = np.sum(np.where(mask_i, z, 0))//volume_i
  cyt_volumes.append(volume_i)
  cyt_centres.append((x_i,y_i,z_i))

print("Saving all results")

with open(SAVE_PATH + "/summary.csv", "w+") as f:
  f.write("{},{},{},{},{},{},{},{},{}\n".format("N", "x nuc", "y nuc", "z nuc", "volume", "x cyt", "y cyt", "z cyt", "volume cyt"))
  for idx,(v, (x,y,z), cv, (cx,cy,cz)) in enumerate(zip(volumes, centres, cyt_volumes, cyt_centres)):
    f.write("{},{},{},{},{},{},{},{},{}\n".format(idx+1, x, y, z, v, cx,cy,cz,cv))

for idx in range(1,len(volumes)+1):
  cell = (segmented == idx).astype(np.uint8)*255
  utils.tf.imwrite(SAVE_PATH+f"/segmented_cell_{idx:0>2}.tif", cell)

print("DONE!")
print(f"Elapsed: {time()-start:0.0f}s")