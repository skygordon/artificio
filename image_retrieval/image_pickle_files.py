"""

 image_pickle_files.py  (author: Skylar Gordon / git: skygordon)

 We collect outputs using transfer learning on a pre-trained
 VGG or ResNet image classifier for data collection for CNN research. 

"""
import os
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from src.CV_IO_utils import read_imgs_dir
from src.CV_transform_utils import apply_transformer
from src.CV_transform_utils import resize_img, normalize_img
from src.CV_plot_utils import plot_query_retrieval, plot_tsne, plot_reconstructions
from src.autoencoder import AutoEncoder

# Run mode: (autoencoder -> simpleAE, convAE) or (transfer learning -> vgg19)
# We only care about transfer learning with CNNs for our data collection
modelName = "ResNet"  # try: "vgg19", "ResNet"
trainModel = True
parallel = True  # use multicore processing

# Make paths
dataTrainDir = os.path.join(os.getcwd(), "data", "train")
dataTestDir = os.path.join(os.getcwd(), "data", "test")
outDir = os.path.join(os.getcwd(), "output", modelName)
if not os.path.exists(outDir):
    os.makedirs(outDir)

# Read images
extensions = [".jpg", ".jpeg"]
print("Reading train images from '{}'...".format(dataTrainDir))
imgs_train = read_imgs_dir(dataTrainDir, extensions, parallel=parallel)
print("Reading test images from '{}'...".format(dataTestDir))
imgs_test = read_imgs_dir(dataTestDir, extensions, parallel=parallel)
shape_img = imgs_train[0].shape
print("Image shape = {}".format(shape_img))
print("Model Name: {}".format(modelName))

# Build models
if modelName in ["vgg19", "ResNet"]: # used to be elif
    if modelName == "vgg19":
        # Load pre-trained VGG19 model + higher level layers
        print("Loading VGG19 pre-trained model...")
        model = tf.keras.applications.VGG19(weights='imagenet', include_top=False,
                                        input_shape=shape_img)
        model.summary()
    elif modelName == "ResNet": # Load pre-trained ResNet50 model + higher level layers
        print("Loading ResNet50 pre-trained model...")
        model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,  
                                        input_shape=shape_img) 
        model.summary()
    
    shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
    input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
    output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
    n_epochs = None

else:
    raise Exception("Invalid modelName!")

# Print some model info
print("input_shape_model = {}".format(input_shape_model))
print("output_shape_model = {}".format(output_shape_model))

# Apply transformations to all images
class ImageTransformer(object):

    def __init__(self, shape_resize):
        self.shape_resize = shape_resize

    def __call__(self, img):
        img_transformed = resize_img(img, self.shape_resize)
        img_transformed = normalize_img(img_transformed)
        return img_transformed

transformer = ImageTransformer(shape_img_resize)
print("Applying image transformer to training images...")
imgs_train_transformed = apply_transformer(imgs_train, transformer, parallel=parallel)
print("Applying image transformer to test images...")
imgs_test_transformed = apply_transformer(imgs_test, transformer, parallel=parallel)

# Convert images to numpy array
X_train = np.array(imgs_train_transformed).reshape((-1,) + input_shape_model)
X_test = np.array(imgs_test_transformed).reshape((-1,) + input_shape_model)
print(" -> X_train.shape = {}".format(X_train.shape))
print(" -> X_test.shape = {}".format(X_test.shape))

# Create embeddings using model
print("Inferencing embeddings using pre-trained model...")
E_train = model.predict(X_train)
E_train_flatten = E_train.reshape((-1, np.prod(output_shape_model)))
E_test = model.predict(X_test)
E_test_flatten = E_test.reshape((-1, np.prod(output_shape_model)))
print(" -> E_train.shape = {}".format(E_train.shape))
print(" -> E_test.shape = {}".format(E_test.shape))
print(" -> E_train_flatten.shape = {}".format(E_train_flatten.shape))
print(" -> E_test_flatten.shape = {}".format(E_test_flatten.shape))

########## Pickling ############ I ran this twice, with modelName = "ResNet" and modelName = "vgg19"
import pickle

for i, image in enumerate(E_train):
    etraincurr = 'saved_outputs/{}/E_train/retrieval_{}.pkl'.format(modelName, i)
    with open(etraincurr,'wb') as f:
        pickle.dump(image, f)

for i, image in enumerate(E_test):
    etraincurr = 'saved_outputs/{}/E_test/retrieval_{}.pkl'.format(modelName, i)
    with open(etraincurr,'wb') as f:
        pickle.dump(image, f)
