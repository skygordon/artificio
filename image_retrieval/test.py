import pickle
import numpy as np

# arr = numpy.array([[1, 2], [3, 4]])
# pickle.dump(arr, open("sample.pkl", "wb"))

arr = np.zeros((15,2))
with open('test/sample.pkl','wb') as f:
    pickle.dump(arr, f)

with open('test/sample.pkl','rb') as f:
    x = pickle.load(f)
    print(x.shape)

# print (pickle.load(open("sample.pkl")))



# import os

# modelName = = "ResNet"  # try "vgg19" or "ResNet"
# # added to store pickled files
# outDir = os.path.join(os.getcwd(), "saved_data", modelName)
# if not os.path.exists(outDir):
#     os.makedirs(outDir)

# # where we access original files
# filename1 = "output/ResNet/E_test"
# filename2 = "output/ResNet/E_train"
# # where we store pickled files
# filename3 = "saved_data/ResNet/pickled_E_test"
# filename4 = "saved_data/ResNet/pickled_E_train"

# # pickling E test data
# outfile = open(filename1,'wb')
# pickle.dump(filename3, outfile)
# outfile.close()

# # pickling E train data
# outfile = open(filename2,'wb')
# pickle.dump(filename4, outfile)
# outfile.close()

# with open('test.pkl','wb') as f:
#     pickle.dump(arr, f)