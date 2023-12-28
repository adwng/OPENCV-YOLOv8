# OPENCV-YOLOv8
Task 1 - Predict Number of Legs on an IC.
Task 2 - Using Roboflow for Annotation & Yolov8 as Training and Prediction Model

Task 1:
  Using OpenCV, an IC's legs can be detected and summated to find out if the IC has complete legs or missing. It makes use of estimating the body size of the IC and predict the amount of legs it should have. However, this is only applicable for IC bodys of 16 and thus making it inefficient for IC Legs Identification

Task 2:

1. Using Roboflow, a dataset can be done by uploading video frames of the ICs into the project base. Roboflow is an free annotating tool software whereby classes can be created.
2. A sample website for reference can be found: https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/

3. Furthermore, after the annotation process is done, the dataset can be downloaded via zip to your personal laptop.
4. In the downloaded folder, there is a file called 'data.yaml', inside the file, make sure to edit the paths of:
     - /train/images
     - /valid/images
     - /test/images
5. To the absolute paths of the train, valid, and test folders in the downloaded zip.
6. Install yolov8 via https://github.com/ultralytics/ultralytics.
7. In their repository, get sample code for training and prediction to reference to your own code to conduct training on downloaded dataset and prediction
