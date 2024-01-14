# Diabetes Classification
In this project, I have used various classification techniques to predict whether a patient has diabetes or not, based on certain diagnostic measurements.

Dataset Used - https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset.<br>
#### Attribute description 

<b>Pregnancies:</b> To express the Number of pregnancies

<b>Glucose:</b> To express the Glucose level in blood

<b>BloodPressure:</b> To express the Blood pressure measurement

<b>SkinThickness:</b> To express the thickness of the skin

<b>Insulin:</b> To express the Insulin level in blood

<b>BMI:</b> To express the Body mass index

<b>DiabetesPedigreeFunction:</b> To express the diabetes likelihood depending on the subject's age and his/her diabetic family history

<b>Age:</b> To express the age

<b>Outcome:</b> To express the final result 1 is Yes and 0 is No

#### Classification techniques used - 
1) Linear Regression
2) Artificial Neural Networks
3) Decision Tree Classifier
4) Bagging Classifier
5) Random Forest Classifier
6) AdaBoost Classifier
7) GradientBoost Classifier

<b> There are a lot of missing values in the Glucose, BloodPressure, SkinThickness, Insulin, and BMI attributes. To handle these missing values, I have used imputation using Random Forests (https://www.youtube.com/watch?v=sQ870aTKqiM&t=631s)</b> 

The performance is significantly better than other versions that I saw online. This, in my understanding, is due to a more sophisticated imputation technique. Other versions simply use mean or median of the particular attribute to fill the missing values.

## Content
- Diabetes_Classification.ipynb - The main jupyter notebook
- DT_framework.py - decision tree and ensemble framework (as written in https://github.com/shashwatgupta42/tree_ensemble_framework)
- NN_Framework.py - feed forward neural net framework (as written in https://github.com/shashwatgupta42/neural_net)
- RF_imputer.py - the script containing the function for imputation using random forests
- datasets - folder containing the dataset
