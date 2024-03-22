# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

 
# Function to display confusion matrix
def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

# Split the dataset into train, validation, and test sets 
X_train = X[:5000]
y_train = y[:5000]
X_val = X[5000:6000]
y_val = y[5000:6000]

X_test = X[6000:7000]
y_test = y[6000:7000]

# Standardizing the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)   
X_test_scaled = scaler.transform(X_test)  
joblib.dump(scaler, "scaler.pkl")

##EDA

print(X_train_scaled.shape)
print(y_train.shape)

# Plotting a sample image from the dataset
some_digit = X[150]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)

print(y[150])

# Create model 1: Support Vector Machine Classifier with hyperparameter tuning
svm_clf = SVC()
hyper_param_grid = [
    {'kernel': ['rbf', 'poly'], 'gamma': [0.25, 0.5, 5], 'C': [0.5, 1, 1,5]}
]
grid_search = GridSearchCV(svm_clf, hyper_param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
y_val_pred_grid = grid_search.predict(X_val_scaled)

# Visualize the confusion matrix for SVM model
display_confusion_matrix(y_val, y_val_pred_grid)
plt.title("Confusion Matrix SVM validation data")
plt.show()

# Create model 2: K-Nearest Neighbors Classifier
knn_clf = KNeighborsClassifier()
hyper_param_grid_knn = {'n_neighbors': [3]}  # You can adjust the number of neighbors as needed
grid_search_knn = GridSearchCV(knn_clf, hyper_param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train_scaled, y_train)

# Find the best KNN model
y_val_pred_knn = grid_search_knn.predict(X_val_scaled)

# Visualize the confusion matrix for SVM model
display_confusion_matrix(y_val, y_val_pred_knn)
plt.title("Confusion Matrix KNN validation data")
plt.show()

# Create model 3: Decision Trees Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
hyper_param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(dt_clf, hyper_param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train_scaled, y_train)

# Find the best Decision Trees model
y_val_pred_dt = grid_search_dt.predict(X_val_scaled)

# Visualize the confusion matrix for SVM model
display_confusion_matrix(y_val, y_val_pred_dt)
plt.title("Confusion Matrix DT validation data")
plt.show()

# Print out the best parameters found by grid search
print("Best parameters:", grid_search.best_params_)


# Calculate accuracy for each model

accuracy_grid_search = accuracy_score(y_val, y_val_pred_grid)
accuracy_grid_search_knn = accuracy_score(y_val, y_val_pred_knn)
accuracy_grid_search_dt = accuracy_score(y_val, y_val_pred_dt)

# Print accuracies

print("SVM with RBF or polynomial kernel accuracy on validation set:", accuracy_grid_search)
print("K-Nearest Neighbors Classifier accuracy on validation set:", accuracy_grid_search_knn)
print("Decision Trees Classifier accuracy on validation set:", accuracy_grid_search_dt)

# Store accuracies in a dictionary
accuracies = {
    "SVM with RBF or polynomial kernel": accuracy_grid_search,
    "K-Nearest Neighbors Classifier": accuracy_grid_search_knn,
    "Decision Trees Classifier": accuracy_grid_search_dt
}


# Initialize the best SVM model with the best parameters found by grid search
svm_model3 = SVC(kernel='poly', gamma=0.25, C=0.5)
svm_model3.fit(X_train_scaled, y_train)

# Make predictions on the preprocessed test data
y_test_pred = svm_model3.predict(X_test_scaled)

# Visualize the confusion matrix for SVM model
display_confusion_matrix(y_test, y_test_pred)
plt.title("Confusion Matrix test data")
plt.show()


print(classification_report(y_test, y_test_pred))

# Evaluate the performance of the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test set accuracy:", test_accuracy)



# Save the best SVM model
joblib.dump(svm_model3, "test_model3.pkl")

# Function to preprocess an image before prediction
def preprocess_image(file_path, target_size=(28, 28), lower_pixel=80, upper_pixel=235):
    # Read image
    test_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        raise FileNotFoundError("Image not found or unable to read")

    # Resize image
    img_resized = cv2.resize(test_image, target_size, interpolation=cv2.INTER_LINEAR)
    img_resized = cv2.bitwise_not(img_resized)

    # Thresholding
    img_processed = np.where((img_resized <= lower_pixel), 0, img_resized)
    img_processed = np.where((img_processed > upper_pixel), 255, img_processed)

    return img_processed

# Function to display original and processed images
def display_images(original, processed):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Image")
    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title("Processed Image")
    plt.show()

# Main function
def main():
    file_path = r"C:\Users\Jacob\Documents\Machine Learning\Kunskapskontroll\handwritten 3.PNG"
    
    # Preprocess image
    processed_image = preprocess_image(file_path)

    # Display original and processed images
    original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    display_images(original_image, processed_image)

    # Reshape image for prediction
    processed_image_flat = processed_image.reshape(-1, 784)

    # Make prediction
    prediction = grid_search.predict(processed_image_flat)
    print("Number prediction:", prediction)

if __name__ == "__main__":
    main()

