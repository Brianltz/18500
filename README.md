This is a project that aims to count the capacity of people entering, exiting rooms in a hallway thorugh computer vision. We set up cameras that stream video feed to google colab through ngrok, and processes the videofeed with object detection and tracking to count number of people entering/exiting rooms

yolov5 documentation:
https://github.com/ultralytics/yolov5
clone this repo and install requirements with following:

Requirements:
$ need openCV version 4.6.0
$ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
$ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Deepsort realtime tracker:
https://pypi.org/project/deep-sort-realtime/

install:
$ pip install deep-sort-realtime

then you should be able to run the code