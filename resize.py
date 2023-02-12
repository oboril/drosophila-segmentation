import yaml
import utils
from scipy.ndimage import zoom
import numpy as np

def resize_complete(path, inp_res, out_res, interpolation_order):
  # Load image
  img, scale, metadata = utils.load_image(path, dtype=float)

  # Check resolution
  inp_res = utils.check_resolution(scale, inp_res)
  res_ratio = [i/o for i,o in zip(inp_res, out_res)] + [1.]*(len(img.shape) - 3)

  print("The input image has shape", img.shape)

  # Resize image
  resized = zoom(img, res_ratio, mode="constant", cval=0., prefilter=True, order=interpolation_order)

  print("The resized image has shape", resized.shape)

  return resized, metadata

if __name__ == "__main__":
  config = utils.load_yaml("resize_config.yaml")
  inp_res = [config.input_resolution.x, config.input_resolution.y, config.input_resolution.z]
  out_res = [config.output_resolution.x, config.output_resolution.y, config.output_resolution.z]

  resized, metadata = resize_complete(config.input_file, inp_res, out_res, config.interpolation_order)

  utils.save_image(config.output_file, resized.astype(np.uint16), resolution=out_res, metadata=metadata)
  print("Output image saved!")


