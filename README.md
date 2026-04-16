# Gesture Detector
Using mediapipe library this project detects 21 point in one's hand and matches them with letters according to [this](##Execution-and-Usage) diagram. Enables user to write a whole sentences with only his hands. 

## Data Collection 
Data were collected by using DataCollection project. It uses Mediapipe library that locates 21 points on hand and counts their cordinates ([x, y, z] landmarks) from zero point on a wrist. They were collected by recording these points and loading them into csv file. These files includes cordinates of alphabetical, numerical, few special characters and rock, paper, scissors game gestures. 

### Data processing
Data were cleaned and transformed in Jupyter colab notebook using Pandas. They were reduced of duplicate records and null lines. 
[Clearing Data - Jupyter Notebook](https://colab.research.google.com/drive/1Es2mXxtfxMs5MvX3xGpQhv7P-3ra6VXZ#scrollTo=1_l9p4xuMnyD)

### Data Usage - Model 
Three Random forest models were trained and used. Each for different gesture modes. All models reached accuracy of 99%.
[Random Forest Classifier Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Requirements 
* **Python 3.11.x or 3.10.x**
* Libraries: [here](##Used-Libraries)
* Webcamera

## Execution and usage
Basicly install .exe file in Releases section. The newer version the better.
Then you simply run the application. Available gestures are displayed in an images bellow.

<img width="559" height="900" alt="prstováAbeceda" src="https://github.com/user-attachments/assets/46b98458-ed40-4352-9a8c-f45430de000d" />

> [!TIP]
> If you have problems with recognition, take your hand further.

<img width="523" height="315" alt="cisla" src="https://github.com/user-attachments/assets/517c4e0e-914d-4551-960b-5642cdd2268d" />

## Used Libraries
* MediaPipe
* Opencv
* Joblib
* NumPy
* Pandas
* Scikit-learn

## Version and Issues
* Version: 1.0 
  * good gesture recognition
  * two recognition modes
    * rock, paper, scissors
    * english alphabet (25 letters)
  * modes switch by pressing m key 
* Version: 2.0
  * three modes
    * rock, paper, scissors
    * english alphabet (25 letters) + special characters (.,?space)
    * numerical gestures (0-9)
  * modes switch by gesture or m key
  * automatic writing after holding one gesture for 3 seconds        
              
## Author
**Name:** Nikola Poláchová 
**Subject:** Programming 
**Year:** 2026
