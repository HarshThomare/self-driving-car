# Overview
This project is built using a CNN based on the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This model has shown to be effective in training the car to predict the steering angle using the input image.
# Preprocessing
Images that were collected were first normalized to reduce weightage of a single steering angle
Then the images were cropped to only show the road
Finally, to reduce overfitting, the following augmentation were made:
- Tilting the image
- Changing the brightness
- Flipping the image
- Zooming in
# Model Architecture
The model is a Multilayer Convolutional Neural Network connected to fully connected neural network.
The Structure of the model looks as follows:
- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output)

The optimizer used was ADAM and the loss was Mean Squared Error.
# Files
- `model.py` The script for the model and training
- `selfdriving.py` The script for image preprocessing
- `main.py` Runs all the individual scripts
- `model.h5` The model weights
- `drive.py` The script to test the model on the simulator

## Demo
https://youtu.be/yF8PFNI-X0A

## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
