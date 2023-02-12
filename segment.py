print("PROGRAM STARTED")

import os
from time import time
import json

import utils
import resize

import numpy as np
from skimage.filters import gaussian
from scipy.ndimage import label, distance_transform_edt, binary_dilation, find_objects
from skimage.morphology import closing, ball, dilation
from skimage.segmentation import watershed

print("Loading config")
config = utils.load_yaml("segment_config.yaml")

print("Loading image", config.input_file)
if config.resizing.resize:
  print("Resizing image")
  inp_res = config.resizing.input_resolution
  inp_res = [inp_res.x, inp_res.y, inp_res.z]
  img, metadata = resize.resize_complete(config.input_file, inp_res, [1., 1., 1.], config.resizing.interpolation_order)
else:
  img, scale, metadata = utils.load_image(config.input_file, dtype=float)
  print("Image was not resized, the resolution in .tif metadata is", scale) 

start = time()

if not os.path.exists(config.output_folder):
  os.makedirs(config.output_folder)

nuclei = img[:,:,:,1]
cytoskelet = img[:,:,:,0]

print("Saving resized nuclei")
utils.tf.imwrite(config.output_folder + "/nuclei.tif", nuclei.transpose([2,1,0]).astype(np.uint8))

print("Finding centers of nuclei")
blurred = gaussian(nuclei, config.nuclei.center.blur)

threshold = np.min(blurred)*(1-config.nuclei.center.threshold) + np.max(blurred)*config.nuclei.center.threshold
mask = blurred > threshold

dist = distance_transform_edt(mask)

mask = np.where(dist > config.nuclei.center.min_radius, 1, 0)

mask = dilation(mask, ball(config.nuclei.center.dilation))

labelled, num_areas = label(mask)

print(f"Found {num_areas} nuclei")

print("Preparing nuclei for segmentation")
nuclei = (nuclei*255/np.max(nuclei)).astype(np.uint8)
nuclei = closing(nuclei, ball(config.nuclei.closing_radius))

threshold = np.min(nuclei)*(1-config.nuclei.threshold) + np.max(nuclei)*config.nuclei.threshold
mask = nuclei > threshold

dist = distance_transform_edt(mask)

print("Segmenting nuclei")
segmented = watershed(-dist, labelled, mask=mask)

print("Processing segmented nuclei")
bounding_boxes = find_objects(segmented)
volumes = []
centres = []
bb_volume_frac = []
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

  bb = bounding_boxes[nucleus-1]
  dx = int(bb[0].stop - bb[0].start)
  dy = int(bb[1].stop - bb[1].start)
  dz = int(bb[2].stop - bb[2].start)
  bb_volume_frac.append(volume_i/(dx*dy*dz))

indeces = range(1, len(volumes)+1)

bb_volume_frac, volumes, centres, indeces = zip(*sorted(zip(bb_volume_frac, volumes, centres, indeces), reverse=True))

if len(volumes) > config.nuclei.max_count:
  print(f"Limiting the nuclei to {config.nuclei.max_count} largest out of {len(volumes)}")
  bb_volume_frac = bb_volume_frac[:config.nuclei.max_count]
  volumes = volumes[:config.nuclei.max_count]
  centres = centres[:config.nuclei.max_count]
  indeces = indeces[:config.nuclei.max_count]

ordered_segmented = np.zeros(segmented.shape, dtype=np.uint8)

print("Sorting nuclei by volume")
volumes, centres, indeces = zip(*sorted(zip(volumes, centres, indeces), reverse=True))
for new_idx, old_idx in enumerate(indeces):
  new_idx += 1
  ordered_segmented = np.where(segmented == old_idx, new_idx, ordered_segmented)

print("Saving segmented nuclei")

#with open(SAVE_PATH + "/nuclei.csv", "w+") as f:
#  f.write("{},{},{},{},{}\n".format("N", "x", "y", "z", "Volume"))
#  for idx,(v, (x,y,z)) in enumerate(zip(volumes, centres)):
#    f.write("{},{},{},{},{}\n".format(idx+1, x, y, z, v))


utils.tf.imwrite(config.output_folder + "/segmented_nuclei.tif", np.where(ordered_segmented.transpose([2,1,0]) > 0, 255, 0).astype(np.uint8))

print(f"NUCLEI SEGMENTED (elapsed {time()-start:0.0f} s)")

print("Equalizing cytoskelet channel along z axis")
cytoskelet = cytoskelet.astype(float)/np.max(img)
cytoskelet = utils.equalize_depth_intensity(cytoskelet, config.cytoplasm.z_equalization_coef)

print("Preparing seeds for cytoplasm segmentation")
seed_points = dilation(ordered_segmented, ball(config.cytoplasm.nucleus_cell_gap.min))

not_outside = ordered_segmented > 0

not_outside = binary_dilation(not_outside, iterations = config.cytoplasm.nucleus_cell_gap.max)

outside_idx = len(volumes)+1
seed_points = np.where(not_outside, seed_points, outside_idx)
seed_points[:,:,0] = outside_idx
seed_points[:,:,-1] = outside_idx
seed_points[:,0,:] = outside_idx
seed_points[:,-1,:] = outside_idx
seed_points[0,:,:] = outside_idx
seed_points[-1,:,:] = outside_idx

print("Preprocessing cytoskelet channel")
cytoskelet = gaussian(cytoskelet, 1)
cytoskelet = (cytoskelet*255/np.max(cytoskelet)).astype(np.uint8)
for _ in range(config.cytoplasm.closing_iters):
  cytoskelet = closing(cytoskelet, ball(config.cytoplasm.closing_radius))
cytoskelet = gaussian(cytoskelet, 1)
cytoskelet = (cytoskelet*255/np.max(cytoskelet)).astype(np.uint8)

print("Saving processed cytoskelet")
utils.tf.imwrite(config.output_folder + "/cytoskelet.tif", cytoskelet.transpose([2,1,0]).astype(np.uint8))

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

with open(config.output_folder + "/summary.csv", "w+") as f:
  f.write("{},{},{},{},{},{},{},{},{}\n".format("N", "x nuc", "y nuc", "z nuc", "volume", "x cyt", "y cyt", "z cyt", "volume cyt"))
  for idx,(v, (x,y,z), cv, (cx,cy,cz)) in enumerate(zip(volumes, centres, cyt_volumes, cyt_centres)):
    f.write("{},{},{},{},{},{},{},{},{}\n".format(idx+1, x, y, z, v, cx,cy,cz,cv))

for idx in range(1,len(volumes)+1):
  cell = (segmented == idx).astype(np.uint8)*255
  utils.tf.imwrite(config.output_folder+f"/segmented_cell_{idx:0>2}.tif", cell.transpose([2,1,0]))

print("DONE!")
print(f"Elapsed: {time()-start:0.0f}s")