# Installation

## Prerequisites

Python 3
 
[Tensorflow](https://www.tensorflow.org/install/pip) *keras included

[OpenCV](https://pypi.org/project/opencv-python/)


# Usage
### Separate Group Pictures
Include group pictures on Unknown Group Pictures

Press Separate Group Pictures

Individual Faces are then transferred to Unknown Individual Faces
### Label Unknown Faces
Individual faces on Unknown Individual Faces will be manually labeled by the user

Labeled faces are moved to Known Faces/{label}/

### Train CNN/RNN
Checks if previously existing model exists, if not creates a model.


Model trains using Known Faces


### Classify CNN/RNN

Checks if model exists

Checks Testing Pictures/{all labeled folders}

Outputs prediction of image one by one