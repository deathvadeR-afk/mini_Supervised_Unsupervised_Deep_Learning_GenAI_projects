# Import necessary libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the breast cancer dataset
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])  # Print first data sample
print(breast_cancer_data.feature_names)  # Print feature names
print(breast_cancer_data.target)  # Print target values
print(breast_cancer_data.target_names)  # Print target class names

# Split the data into training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=42)
print(len(training_data))  # Print number of training samples
print(len(training_labels))  # Print number of training labels

# List to store accuracy for each k value
accuracies = []
# Test k values from 1 to 100
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)  # Create KNN classifier with k neighbors
    classifier.fit(training_data, training_labels)  # Train classifier
    accuracies.append(classifier.score(validation_data, validation_labels))  # Store validation accuracy

# Plot accuracy vs. k
k_list = range(1, 101)
plt.plot(k_list, accuracies)
plt.xlabel('No. of k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()