# ThinkerFarmTrainer


### ThinkerFarmTrainer V1.0.0

## Introduction  

ThinkerFarmTrainer is a toolset for training ssd object detection models. Originally i made this toolset for myself aiming to ease my custom object detection model training process so i'm sharing here and hope you will find useful. I use transfer learning method on ssd mobilenet v2 quantized 300x300 coco. Model performance is quite good for variety of mobile and edge projects.

## Features    

[✓] - TensorFlow Object Detection Module  
[✓] - Train and test cvs generator  
[✓] - Train and test record generator  
[✓] - Model converter to TFLite  


## Usage and Installation  
This is not a tutorial but i will try to explain whole training process step by step and as brief as possible.

### Need to Know
For testing i put together Mobilephone detection files "label_map.pbtxt train_labels.csv test_labels.csv and train_images test_images" gathered from OpenImage dataset V5 so after Installation you can immediately train model and generate tfrecord using 6. Start Training "5. Generate TFRecords" this train and generate a Mobilephone detection model.  

When training steps good enough convert last check point to TFLite model file and test your newly trained model with [ThinkerFarm](https://github.com/erkansirin/ThinkerFarm) mobile framework (iOS only)  

If you want use another model weights you have to change files inside "trainer/pre_trained_mobilenet" folder.  

### Clone repository  
```
$ git clone https://github.com/erkansirin/ThinkerFarmTrainer.git  

$ cd ThinkerFarmTrainer  

$ bash ./run.sh  
```
### After initialize run.sh script you will land following menu :  
```
~~~~~~~~~~~~~~~~~~~~~
 M A I N - M E N U
~~~~~~~~~~~~~~~~~~~~~
1. Install Dependencies
2. Clean Training Data
3. Run LabelImg by Tzutalin and Create bounding boxes
4. Generate train and test labels csv
5. Generate TFRecords
6. Start Training
7. Convert Model to TFLite
8. Change Train Images Path
9. Change Test Images Path
10. Open Tensorboard
11. Quit
~~~~~~~~~~~~~~~~~~~~~
```
### 1. Install Dependencies  
This will set you up and install all required dependencies. I tested installation on Ubuntu 18.04.3 LTS and macOS Mojave

### 2. Clean Training Data  
This option will dele train and test record file along with check points and TFLite converted model files.   

### 3. Run LabelImg by Tzutalin and Create bounding boxes  
Before generate csv files and tfrecords and start training you need to create bounding boxes on your training images. It's simply telling to ssd mobilenet where your object located in image so it can extract features from it. Creating bounding boxes is very easy with this amazing tool [LabelImg created by Tzutalin](https://github.com/tzutalin/labelImg) LabelImg tool should be installed by now in step "1. Install Dependencies"

### 4. Generate train and test labels csv  
After you create all bounding boxes with LabelImg this tool will combine all xml files into one single csv file.  

### 5. Generate TFRecords  
This tool generate tfrecord by using csv file created by previous tool.  

### 6. Start Training
This tool will start training by using tfrecord file generated with previous tools  

### 7. Convert Model to TFLite  
Training periodically record training summary into check point file. It's actually your model file and this tool automatically generate TFLite model from latest check point file.   

### 8. Change Train Images Path & 9. Change Test Images Path   
Sometimes you may need to change your image paths this tool will change all image path location in entire project.  

### 11. Open Tensorboard  
Opens tensorboard to investigate your progress

### 11. Quit   
...


## [Licence](https://github.com/erkansirin/ThinkerFarmTrainer/blob/master/LICENSE)  
