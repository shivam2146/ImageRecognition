from PIL import Image
import numpy as np

img_height = 28
img_width = 28

def conv_mnist(filename):
  img = Image.open(filename).convert('L')   # converting to grayscale image
  height, width = img.size                  # height and width of the image

  if height != img_height and width != img_width:   # check the height and width of the image and set it for our model
    img = img.resize([img_width,img_height])
  img = np.asarray(img)                             # creating numpy array from image
  img = img.astype(np.float32)                      # changing type of pixel values
  img = np.multiply(img,1.0/255.0)                  # change the values in range 0 to 1
  #print(img)
  img = img.reshape([1,img_height*img_width])
  return img
  #img.show()
  #print(img.size)
  ######################
  #if img.format != 'png':
  #  name = filename.split('.')[0]
  #  im.save(name,'png')
  ######################
