# Custom Object Detection Model

## Overview

This repository contains a targeted object detection model implemented in Python using OpenCV, NumPy, and TensorFlow. The model has been tested with three different architectures: EfficientDet D4, RCNN Inception ResNet V2, and SSD MobileNet V2.

## Objective

The primary objective of this project is to provide a versatile and easily deployable targeted object detection model, capable of accurately identifying objects in images using state-of-the-art architectures. The model is designed to offer flexibility by supporting distinct architectures like: EfficientDet D4, RCNN Inception ResNet V2, and SSD MobileNet V2. Users can seamlessly integrate this object detection solution into their projects or use it as a standalone tool.

## Features

- **Multiple Architectures**: The model supports various popular object detection architectures like EfficientDet D4, RCNN Inception ResNet V2, and SSD MobileNet V2.

- **Easy Integration**: The codebase is designed to be easily integrated into existing projects or used as a standalone object detection solution.

- **Dependency Management**: The model relies on widely-used Python libraries such as OpenCV, NumPy, and TensorFlow, making it easy to manage dependencies.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- NumPy
- TensorFlow

## Performance

- **SSD MobileNet V2**: It detected less objects with less confidence.
- **EfficientDet D4**: Detected more objects but the confidence was still low.
- **RCNN Inception ResNet V2**: Detected less objects but with higher confidence.

## Libraries Used
<li>NumPy</li>
<li>OpenCV</li>
<li>Tensorflow</li>

## Result
<li>SSD MobileNet V2</li>

![ssd_mobilenet_v2_320x320_coco17_tpu-8](https://github.com/nadgawd/Custom-Object-Detection/assets/130206961/cc95bad8-67cb-46cb-bfc0-73915c445a7e)

<li>EfficientDet D4</li>

![efficientdet_d4_coco17_tpu-32](https://github.com/nadgawd/Custom-Object-Detection/assets/130206961/154e607c-1b9c-4b0d-99cc-047974bcd6e1)

<li>RCNN Inception ResNet V2</li>

![faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8](https://github.com/nadgawd/Custom-Object-Detection/assets/130206961/368c1440-277f-429a-b82c-ac7af9edfbd4)

