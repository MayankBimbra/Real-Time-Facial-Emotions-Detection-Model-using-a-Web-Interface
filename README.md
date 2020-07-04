# Data Mining and Analysis Project
# Real Time Facial Expressions/Emotions Detection using a Web Interface in Python

# Description
The Human facial expressions are important for visually expressing a lot more information. Facial expression recognition is essential in the field of human-machine interaction. Automated facial recognition systems have many applications, including understanding of human behavior, diagnosing mental disorders, and synthetic human expression. Identifying facial expressions through computers with high detection rates is still a challenging task.

Computer animated agents and robots bring new dimension in human computer interaction which makes it vital as how computers can affect our social life in day-to-day activities. Face to face communication is a real-time process operating at a a time scale in the order of milliseconds. The level of uncertainty at this time scale is considerable, making it necessary for humans and machines to rely on sensory rich perceptual primitives rather than slow symbolic inference processes.

In this project we are presenting the real time facial expression recognition of seven most basic human expressions.We have used a variety of intensive deep learning techniques (convolutional neural networks) to identify the main seven universal human emotions: 

I. Neutral II. Angry III. Disgust IV. Fear V. Happy VI. Sadness VII. Surprise

# Problem
#### For any real time image taken from our web camera, our goal is to predict the expression of the face in that image out of seven basic human expression.
 i.e. CLASSIFY THE EXPRESSION OF FACE IN IMAGE OUT OF SEVEN BASIC HUMAN EXPRESSION
# Project Formulation

The hands on building this project of Facial Expression Recognition is divided into following tasks/steps:-

#### A.	Task 1: Introduction 
•	Introduction to the dataset

•	Import essential modules and helper functions from NumPy, Matplotlib, and Keras.

#### B.	Task 2: Exploring the Dataset
•	Display some images from every expression type in the Emotion FER dataset.

#### C.	Task 3: Generating Training and Validation Batches
•	Generate batches of tensor image data with real-time data augmentation.

•	Specify paths to training and validation image directories and generates batches of augmented data.

#### D.	Task 4: Creating a Convolutional Neural Network (CNN) Model
•	Design a convolutional neural network with 4 convolution layers and 2 fully connected layers to predict 7 types of facial expressions.

•	Used Adam as the optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.

#### E.	Task 5: Training and Evaluating Model
•	Training the CNN by invoking the model.fit() method.

•	Used ModelCheckpoint() to save the weights associated with the higher validation accuracy.

•	Observed live training loss and accuracy plots in Jupyter Notebook for Keras.

#### F.	Task 6: Saving and Serializing Model as JSON String
•	Used to_json(), which uses a JSON string, to store the model architecture.

#### G.	Task 7: Creating a Flask App to Serve Predictions
•	We used the open-source code from "Video Streaming with Flask Example" to create a flask app to serve the model's prediction images directly to a web interface.

#### H.	Task 8: Creating a Class to Output Model Predictions
•	Created a FacialExpressionModel class to load the model from the JSON file, load the trained weights into the model, and predict facial expressions.

#### I.	Task 9: Designed an HTML Template for the Flask App
•	Designed a basic template in HTML to create the layout for the Flask app.

#### J.	Task 10: Used Model to Recognize Facial Expressions at the Real Time using laptops webcamera
•	We than run the main.py script to create the Flask app and serve the model's predictions to a web interface.

•	Applied the model for real time recognition of facial expresssions of users using webcam of the Laptop.


# Dataset Description
The dataset used in this project work has been taken from the Kaggle.com available at (https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column.

The training set consists of 28,709 examples. The public test set used for the leaderboard consists of 3,589 examples. The final test set, which was used to determine the winner of the competition, consists of another 3,589 examples.

# Prerequisites
You need to have installed following softwares and libraries in your machine before running this project.

1.Python 3

2.Anaconda: It will install jupyter notebook and most of the libraries which are needed like sklearn, pandas, seaborn, matplotlib, numpy, PIL.

3.OpenCV

4.keras

5.flask

6. Atom: Atom is a free and open-source text and source code editor for macOS, Linux, and Microsoft Windows. 

# Project Structure 
1. Locating faces in the scene 

2. Extracting facial features from the detected face region 

3. Analyzing the motion of facial features and/or the changes in the appearance of facial features 

# Steps followed for building the project

As per various surveys it is found that for implementing this project four basic steps are required to be performed.

i.) Preprocessing

ii.) Face registration 

iii.) Facial feature extraction 

iv.) Emotion classification

# Author
Mayank Bimbra



