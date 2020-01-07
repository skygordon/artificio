import pickle
import os

modelName = = "ResNet"  # try "vgg19" or "ResNet"
# added to store pickled files
outDir = os.path.join(os.getcwd(), "saved_data", modelName)
if not os.path.exists(outDir):
    os.makedirs(outDir)

# where we access original files
filename1 = "output/ResNet/E_test"
filename2 = "output/ResNet/E_train"
# where we store pickled files
filename3 = "saved_data/ResNet/pickled_E_test"
filename4 = "saved_data/ResNet/pickled_E_train"

# pickling E test data
outfile = open(filename1,'wb')
pickle.dump(filename3, outfile)
outfile.close()

# pickling E train data
outfile = open(filename2,'wb')
pickle.dump(filename4, outfile)
outfile.close()