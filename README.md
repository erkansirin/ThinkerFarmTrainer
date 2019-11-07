# ThinkerFarmTrainer


### ThinkerFarmTrainer V1.0.0

## Introduction  

ThinkerFarmTrainer is a toolset for training Deep Learning Machine Learning models. Originally i made this toolset for myself in order to ease custom object detection model training process so i'm sharing here and hope you will find useful. I use transfer learning method on ssd mobilenet v2 quantized 300x300 coco. Model performance is quite good for variety of mobile and edge projects.

## Features    

[✓] - TensorFlow Object Detection Module  
[✓] - Train and test cvs generator  
[✓] - Train and test record generator  
[✓] - Model converter to TFLite  


## Usage and Installation  
I will try to explain whole training process step by step and as brief as possible.

### Need to Knows  
In this repo in sake of testing i put together Mobilephone detection files "label_map.pbtxt train_labels.csv test_labels.csv and train_images test_images" gathered from OpenImage dataset V5 so after Installation you can immediately generate tfrecord using "4. Generate TFRecords" menu and then start training for Mobilephone detection model. Whenever you feel enough use main menu to convert last check point to TFLite model file and test your newly trained model by using [ThinkerFarm](https://github.com/erkansirin/ThinkerFarm) mobile framework (iOS only)  

### Clone repository  
```
$ git clone https://github.com/erkansirin/ThinkerFarmTrainer.git  

$ cd ThinkerFarmTrainer  

$ bash ./run.sh  
```
### After initialize run.sh script you will land following menu :  
```
~~~~~~~~~~~~~~~~~~~~~
 M A I N - M E N U"
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
10. Quit
~~~~~~~~~~~~~~~~~~~~~
```
### 1. Install Dependencies  
This will set you up and install all required dependencies. I tested installation on Ubuntu 18.04.3 LTS and Mac OS.  

### 2. Clean Training Data  
This option will dele train and test record file along with check points and TFLite converted model files  

### 3. Generate train and test labels csv  


## [Licence](https://github.com/erkansirin/ThinkerFarmTrainer/blob/master/LICENSE)  
