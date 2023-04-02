import tifffile as tf
from skimage.transform import resize
import numpy as np
from skimage.filters import gaussian
import yaml
import json

def load_image(path, dtype=None):
  # returns XYZC
  with tf.TiffFile(path) as tif:
     img = tif.asarray()
     axes = tif.series[0].axes
     metadata = tif.imagej_metadata

  img = img.transpose([axes.index("X"), axes.index("Y"), axes.index("Z"), axes.index("C")])

  try:
    scale = np.array([float(s.split("=")[1].strip()) for s in metadata["Info"].split("\n") if s.startswith("length") and "#1" in s])
    if len(scale) != 3:
      scale = None
  except:
    scale = None

  if dtype is not None:
    img = img.astype(dtype)

  return img, scale, metadata

def check_resolution(meta_res, config_res):
  if meta_res is None:
    print("WARNING: Could not read image resolution from metadata")
    print("Using resolution from config:",config_res)
  elif sum([abs(x-y) for x,y in zip(config_res, meta_res)])  > 1e-3:
    print("WARNING: Resolution in metadata and in config do not match")
    print("Image metadata:", meta_res)
    print("Config:", config_res)
    print("Resolution from config will be used.")
  return config_res

def save_image(path, image, axes="XYZC", resolution=None, metadata={}):
  # transposes to ZCYX
  image = image.transpose([axes.index("Z"), axes.index("C"), axes.index("Y"), axes.index("X")])

  metadata["axes"] = 'ZCYX'

  if resolution is not None:
    resolution = [1/r for r in resolution]
    metadata["spacing"] = 1/resolution[2]
    tf.imwrite(path,image,imagej=True, resolutionunit="MICROMETER",resolution=resolution[:2], metadata=metadata)
  else:
    tf.imwrite(path,image,imagej=True, resolutionunit="MICROMETER", metadata=metadata)

def equalize_depth_intensity(img, coef, sigma=3):
  blurred = gaussian(img, (sigma, sigma, 0))
  blurred_sum = np.cumsum(blurred, 2)
  fact = np.exp2(blurred_sum/coef)
  return img * fact

def load_yaml(path):
  with open(path, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

  class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)
    def __repr__(self):
      dic = json.loads(json.dumps(self, default=lambda o: o.__dict__))
      return yaml.dump(dic, default_flow_style=False)
  
  config = json.loads(json.dumps(config), object_hook=obj)

  return config
