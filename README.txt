Dataset used is PASCAL VOC2010 : https://drive.google.com/open?id=1XNX_2oqoJqa6wJAAhcetPOM6du9MT38r

*********************BUILDING CLASSIFIER****************************

Extracting data:
From the xml file in the dataset extract image_path, class labels and
bounding box parameters. Functions get_name(a) and read_img(img_path)
Rescale the image pixels and resize the images :
Convert each image into a matrix of size 28 x 28 x 3 which is fed into the
network.
Rescale the pixel values in range 0 - 1 inclusive.
Processing labels:
20 classes are to be formed based on the 20 labels that are given in the
input file
Class labels must be converted to one-hot encoding format
Now split the data sets (11,323 images) into two different parts, one designed
for training (10,000) and another for testing.

BUILIDNG THE MODEL:
The model is built using three convolution layers and three max-pooling layer. 
The first layer will have 32-3 x 3 filters, the second layer will have 64-3 x 3 filters and the third layer will have 128-3 x 3 filters.
The first convolution layer is added by Conv2D () . Next Leaky ReLU is added.
Add the max-pooling layer with MaxPooling2D( ) and so on.
The last layer is a Dense layer that has a softmax activation function
with 20 units(i.e., number of classes)

COMPILE AND TRAIN THE MODEL
Compile the model with binary crossentropy and adam() optimiser.Train the model with Keras' fit() function. The model trains for 20
epochs and batch size of 64.

EVALUATE AGAINST TEST DATA
Evaluate the model against test dataset and then record the accuracy and loss for the classification of the PASCAL-VOC 2010 dataset.

OUTPUT
  ACCURACY : 81%

*********************OBJECT LOCALISATION*******************************

DATA PREPROCESSING
Extract image_path, bounding box, classes of images and process it in the same format as in classification
But instead of resize padding (500 by 500) is done to all images.
This formatted dataset is split into training(1000) and testing datasets. The train dataset is fed to the neural network.

BUILDING THE MODEL
The model is built using two convolution layers and two max-pooling layer. Both the layers will have 32-3 x 3 filters.
The model is built in the same way as classification with first layer as convolution layer added by Conv2D () .The Leaky ReLU activation function. Add the max-pooling layer with MaxPooling2D( ) and so on.
The last layer is a Dense layer that has a linear activation function with 4 units.

COMPILE AND TRAIN THE MODEL
Compile the model with loss function as mean-squared error and adam( ) optimiser.
Train the model with Keras' fit() function. The model trains for 5 epochs and batch size of 1.

EVALUATE AGAINST TEST DATA
Evaluate the model against test dataset and check for the proper detection
of images in PASCAL_VOC DATASETS.

OUTPUT
  ACCURACY : 86%
All the test datasets are localised perfectly. The test input was first 20 images and then the test input size was increased.

********************TRANSFER LAEARNING THROUGH VGGNET********************

IMAGE PARSING TO VGGNET
After trying with our own classifier, we use transfer learning for performance comparison.
Here we use one of the pre-trained models such as VGGNET.
We have included all the libraries and then model is built by removing the top layers and replace them with a 20 layer Softmax.
The VGG16 is already pre-trained on ImageNet datasets, now we are using the pre-trained model against our dataset by setting the required
output layer units on specified list of inputs on VGG architecture.

OUTPUT :
     ACCURACY : 93%
