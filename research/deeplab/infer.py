import os
import argparse
import tensorflow as tf
import tarfile
import numpy as np
from PIL import Image
import ntpath # for base path


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

# make output images
def makeOutputs(resized_im, seg_map):
    '''
    post-processes the model output
    '''
    # make bg removed
    resized_im_np = np.array(resized_im)
    mask = np.logical_not(seg_map==15)
    image_fg = resized_im_np
    image_fg[mask,:] = 0
    image_fg = Image.fromarray(image_fg)
    # plt.imshow(image_fg)
    # plt.show()

    # make blended image
    seg_image = Image.fromarray(label_to_color_image(seg_map).astype(np.uint8))
    image_blended = Image.blend(resized_im, seg_image, alpha=.6)
    # plt.imshow(image_blended)
    # plt.show()

    return image_fg,image_blended 
   
def getModel():
  	model = DeepLabModel(config.model_path)
  	print('Model loaded: {}'.format(config.model_path))
  	return model

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type = str, help = "input directory containing images to process")
    parser.add_argument("--output_dir", type = str, help = "output directory")
    parser.add_argument("--model_path", type = str, help = "path to trained model")


    return parser.parse_args()

config = parseArguments()

# image reading
def readImage(_image_addr):
    '''
    reads the image from image_addr
    '''
    try:
        image = Image.open(_image_addr)
        return image
    except IOError:
        print('Cannot retrieve image. Please check address: ' + _image_addr)
        return

def getImagelist():
    dataset_dir = config.input_dir
    image_list = os.listdir(dataset_dir)

    image_path_list = [os.path.join(dataset_dir,x) for x in image_list]
    return image_path_list

# saveing images
def saveImage(_output_dir, _filename, _image):
    '''
    saves the image to output_dir with filename
    ''' 
    file_addr = os.path.join(_output_dir,_filename)
    _image.save(file_addr)    

def getBasePath(_input_path):
    head, tail = ntpath.split(_input_path)
    return tail or ntpath.basename(head)

def main():

    # get list of images to process
    image_list = getImagelist()

    # read the model
    model = getModel()

    for image_addr in image_list:
        # read the image

        image = readImage(image_addr)

        # produce inference
        # resized_im, seg_map = model.run(image)

        # make output
        image_fg, image_blended = makeOutputs(resized_im, seg_map)

        # save images
        filename = getBasePath(image_addr)
        saveImage(os.path.join(config.output_dir,'fg'), filename, image_fg)
        saveImage(os.path.join(config.output_dir,'blend'), filename, image_blended)

        # send messages

        # logging
    

if __name__ == '__main__':
	
	print("TAG1")
	main()