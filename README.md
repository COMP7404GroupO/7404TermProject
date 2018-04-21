# Butterfly Detection And Classification
=========================
## Demo instructions
* The demo file is realized in predict.py. This file will load the model from /fine\_tune\_for\_rcnn/model and do the detection and classification on the images in file /demoimages. In order to run the demo code, you should install some dependencies.
* Install tensorflow: we use the tensorflow-gpu. Follow the instructions of [installing tensorflow](https://www.tensorflow.org/install/).
* Install numpy: pip install python-numpy
* Install matplotlib: pip install python-matplotlib
* Install selectivesearch: pip install selectivesearch
* Install skimage: pip install scikit-image
* Install PIL: pip install Pillow
* After install all the dependencies, just run the predict.py to see the demo. If there is any problem when doing demo, please connect to <longtaoz@connect.hku.hk>


## Project  progress
### Goal identification
* Using the R-CNN algorithm for buttfly detection and classification.

### Progress
#### Regions proposals
* This part is doing in the preprocessing_rcnn.py. The program will load the information of the images in butterfly\_list.txt and do the regions proposals for all images under /JPEGImages and store the proposal regions data in /dataset.(<i>The upload data file is empty because of the size limitation</i>)

#### CNN network
* This part is finished in fine\_tune\_for\_rcnn.py. The program load the data in /dataset and training the AlexNet network defined in AlexNet.py.
* We get a [pre-trained AlexNet](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) ,bvlc\_alexnet.npy, and doing the fine tuning for our own data. For this part we learned from [a gentle man](http://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html). Thank you!
* The trained model is stored in /fine\_tune\_for\_rcnn/model

#### Detection and classification
* This part is same as the demo. In predict.py, we load the model and images to finish the detection and classification.
* Finally, we use the softmax values of the network output to do the classification and detection.

#### Continuous work: SVM and Bbox Regression
* The program file preprocessing_rcnn.py is to load the information of images under /SVMtrainClass and store the proposal regions data in /datasetsvm.(<i>The upload data file is empty because of the size limitation</i>)
* The program svm_train.py is to train the svms for all 94 classes.
* And we still hope to finished the Bbox Regression to improve the detection performance.
* This part work is still in progress. We will continue our work.

## Improvement
### Faster R-CNN
* Under consideration
