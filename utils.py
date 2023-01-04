import tifffile as tf
from skimage.transform import resize
import numpy as np
from skimage.filters import gaussian

def load_image(path, size=None, dtype=float):
  # returns XYZC
  with tf.TiffFile(path) as tif:
     img = tif.asarray()
     axes = tif.series[0].axes
     metadata = tif.imagej_metadata

  img = img.transpose([axes.index("X"), axes.index("Y"), axes.index("Z"), axes.index("C")])

  scale = np.array([float(s.split("=")[1].strip()) for s in metadata["Info"].split("\n") if s.startswith("length") and "#1" in s])

  if size is not None:
    img = resize(img.astype(float), size)
  return img.astype(dtype), scale

def save_image(path, image, axes="XYZC"):
  # transposes to ZCYX
  image = image.transpose([axes.index("Z"), axes.index("C"), axes.index("Y"), axes.index("X")])
  tf.imwrite(path,image,metadata=({'axes':'ZCYX'}))

def resize_1um(img, scale):
  return resize(img, [*(np.array(img.shape)[:-1]*scale).astype(int),img.shape[-1]])

def equalize_depth_intensity(img, coef, sigma=3):
  blurred = gaussian(img, (sigma, sigma, 0))
  blurred_sum = np.cumsum(blurred, 2)
  fact = np.exp(blurred_sum/coef)
  return img * fact