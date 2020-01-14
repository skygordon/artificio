"""

 cifar10_pickle_files.py  (author: Skylar Gordon / git: skygordon)

 We collect outputs using transfer learning on a pre-trained
 VGG or ResNet image classifier for data collection for CNN research utilizing the cifar10 dataset. 

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

# Makes output files
outDir = os.path.join(os.getcwd(), "pickledcifar10", modelName)
if not os.path.exists(outDir):
    os.makedirs(outDir)

print("Model Name:")
print(modelName)

# Import Cifar10 dataset 
from keras.datasets import cifar10
# Load the CIFAR10 data.
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# Normalize data.
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
# Input image dimensions.
shape_img = X_train.shape[1:]

# Build models
if modelName in ["vgg19", "ResNet"]:
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

print(" -> X_train.shape = {}".format(X_train.shape))
print(" -> X_test.shape = {}".format(X_test.shape))

# Train (if necessary)
if modelName in ["simpleAE", "convAE"]:
    if trainModel:
        model.compile(loss="binary_crossentropy", optimizer="adam")
        model.fit(X_train, n_epochs=n_epochs, batch_size=256)
        model.save_models()
    else:
        model.load_models(loss="binary_crossentropy", optimizer="adam")

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

########## Pickling ############
import pickle

for i, image in enumerate(E_train):
    etraincurr = 'pickledcifar10/{}/E_train/retrieval_{}.pkl'.format(modelName, i)
    with open(etraincurr,'wb') as f:
        pickle.dump(image, f)

for i, image in enumerate(E_test):
    etraincurr = 'pickledcifar10/{}/E_test/retrieval_{}.pkl'.format(modelName, i)
    with open(etraincurr,'wb') as f:
        pickle.dump(image, f)