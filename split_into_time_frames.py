import utils
import numpy as np

INPUT_IMAGE = "../3d_video/images/source.tif"
OUTPUT_FILENAMES = "../3d_video/images/frame_"

with utils.tf.TiffFile(INPUT_IMAGE) as tif:
  img = tif.asarray()
  axes = tif.series[0].axes
  metadata = tif.imagej_metadata

#print(metadata)

#scale = np.array([float(s.split("=")[1].strip()) for s in metadata["Info"].split("\n") if s.startswith("length") and "#1" in s])

print("The resolution is set manually in the code!!!")
scale=(1,0.4143,0.4143)

print(f"Loaded images, the axes are {axes} and the resolution is {scale}")

print("Resizing to 1um")
scale = np.array([1,*scale])
target_shape = (np.array(img.shape)*scale).astype(int)
print(f"Current shape: {img.shape}, target shape: {target_shape}")

img = utils.resize(img, np.array(img.shape)*scale)
img = (img/np.max(img)*255).astype(np.uint8)

for idx, frame in enumerate(img):
  #print(frame.dtype, frame.shape)
  utils.tf.imwrite(OUTPUT_FILENAMES+f"{idx:0>3}.tif", frame, metadata=({'axes':'ZYX'}))