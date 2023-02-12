import yaml
import utils
from scipy.ndimage import zoom
import numpy as np

config = utils.load_yaml("resize_config.yaml")
inp_res = [config.input_resolution.x, config.input_resolution.y, config.input_resolution.z]
out_res = [config.output_resolution.x, config.output_resolution.y, config.output_resolution.z]

img, scale, metadata = utils.load_image(config.input_file, dtype=float)

inp_res = utils.check_resolution(scale, inp_res)

res_ratio = [i/o for i,o in zip(inp_res, out_res)] + [1.]*(len(img.shape) - 3)

print("The input image has shape", img.shape)

resized = zoom(img, res_ratio, mode="constant", cval=0., prefilter=True, order=config.interpolation_order)
resized = resized.astype(np.uint16)

print("The output image has shape", resized.shape)

resized[0,0,0,0] = 0
resized[0,0,0,1] = 65535

utils.save_image(config.output_file, resized, resolution=out_res, metadata=metadata)

print("Output image saved!")


