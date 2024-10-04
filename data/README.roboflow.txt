
AI Drone Lifeguard - v6 2024-04-04 6:48pm
==============================

This dataset was exported via roboflow.com on April 4, 2024 at 10:50 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1309 images.
Human are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 3 pixels
* Salt and pepper noise was applied to 0.98 percent of pixels

The following transformations were applied to the bounding boxes of each image:
* Randomly crop between 0 and 20 percent of the bounding box


