# Car-Damages
This project contains a flask application that detects damages in cars when a user uploads an image. This was built with a Mask-RCNN from the Matterport RCNN repo. In order to run the file, export the run_keras_server_edits-3.py file as a flask application. The Templates folder contains the necessary HTML files and the Static folder contains the CSS file and an Uploads folder containing sample images with car damages. 

The Car_Damages.ipynb file contains all steps for training, testing and visualizing predictions made by the detection model when supplied with trianing and testing data. We have also used occlusion and saliency maps to determine areas that contribute most towards recognition and detection.
