#load libraries
import numpy as np         # linear algebra
import pandas as pd        # data processing, CSV file I/O (e.g. pd.read_csv)

# Read the file "data.csv" and print the contents.
df = pd.read_csv('data.csv', index_col=False)
df.info
# df.isnull().any()

---------------
df.shape
-----------------
print(df.head())   # View the first few rows

-------------------
print(df.info())   # Get a summary of the DataFrame

------------------

print(df.describe())  # Get summary statistics

--------------------

missing_values = df.isnull().sum()
print(missing_values)

----------------

#### Feature engineering 

#1/ finding the area perimeter ratio = check whether the cell is regular or irregular [high or low]
#2/ finding the radius and texture ratio= check whether the cell is larger or smaller [high or low]

df['size_cell']=df['radius_mean']/df['texture_mean']

# we are doing 2nd feature enginerring
#identify the shape of the cancer

df['shape_cancer']=df['area_mean']/df['perimeter_mean']

df.drop(['radius_mean','texture_mean','area_mean','perimeter_mean'],axis=1,inplace=True)

print(df)

----------------------------------------------------------

# Encode the 'diagnosis' column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

-----------------------------------------

#       WE ARE SPLITTING THE DATA IN THE DATASET TO TRAINING AND AND TESTING 

from sklearn.model_selection import train_test_split

x=df[['shape_cancer','size_cell','smoothness_mean','compactness_mean']]
y=df['diagnosis']


# Splitting the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

--------------------------------------------------


## USING RANDOM FOREST ALGORITHM TO TRAIN THE MODEL

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(max_depth=2,random_state=100)

#max_depth means the depth of the tree 
#the data splitting will be deterministic, meaning that every time you run the code with the same value--random_state=100

rf.fit(x_train,y_train)

#we need to fit only the training data to the forest 
--------------------------------------------------------------

# PREDICTING THE X_TRAIN AND X_TEST PERFORMANCE

# y_train_predict=rf.predict(x_train)
# y_test_predict=rf.predict(x_test)
# print(y_train_predict)
# print(y_test_predict)



# Convert the predicted labels from continuous to categorical
y_train_predict_categorical = y_train_predict.round().astype(int)
y_test_predict_categorical = y_test_predict.round().astype(int)


# Compute accuracy score for training and testing predictions
train_accuracy = accuracy_score(y_train, y_train_predict_categorical)
test_accuracy = accuracy_score(y_test, y_test_predict_categorical)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

-----------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Compute precision, recall, and F1-score
precision = precision_score(y_test, y_test_predict_categorical)
recall = recall_score(y_test, y_test_predict_categorical)
f1 = f1_score(y_test, y_test_predict_categorical)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_predict_categorical)
print("Confusion Matrix:")
print(conf_matrix)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_predict)
roc_auc = roc_auc_score(y_test, y_test_predict)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

-------------------------------------------------
#EVALUATING MODEL PERFOMANCE

from sklearn.metrics import mean_squared_error,r2_score

#comparing the prediciton on the training data with the actual target

rf_train_mse=mean_squared_error(y_train,y_train_predict)
rf_train_r2=r2_score(y_train,y_train_predict)
print(rf_train_mse)
print(rf_train_r2)

----------------------------------------------

# Creating a DataFrame to store the results
rf_result = pd.DataFrame({'Model': ['RandomForestRegressor'],
                          'Mean Squared Error (MSE)': [rf_train_mse],
                          'R-squared (R2)': [rf_train_r2]})

print(rf_result)

--------------------------------------------

