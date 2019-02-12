# Truesight
Flipkart grid challenge 2019 repository (second round) by Pranoy Ghosh, Harshit Chebolu and Akanksha Kaushal.
Contains a simple object detection model which predicts bounding boxes for images provided.


# Problem Statement:
• The 2019 GRiD challenge for students is to leverage a predefined data-set from Flipkart to enable ‘Vertical Classification’ using images.

• A dataset was provided which has images and a metadata file containing name of image and bounding box coordinates around the object of image.

# Libraries used:
1. Keras
2. Tensor Flow
3. Pandas
4. Numpy
5. Open CV
6. Scikit-learn

# Model
A convolutional neural network has been used as a feature map extractor for the image and a regression head has been used to predict bounding box values. The model has been trained locally on a Lenovo Legion Y530, which has a Nvidia Graphics card belonging to the GeForce family, a 6GB 1060X.
