#Q2.1
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings as w
w.filterwarnings('ignore')

lfw_data = fetch_lfw_people(min_faces_per_person=110, resize=0.7)

# Fetch LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_data.images.shape
X = lfw_data.data
y = lfw_data.target
target_names = lfw_data.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_classes: %d" % n_classes)

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train_tensor = torch.tensor(X1_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X1_test, dtype=torch.float32)

# Center the data(subtract the mean)
mean_tensor = torch.mean(X_train_tensor, dim=0)
X_train_centered = X_train_tensor - mean_tensor
X_test_centered = X_test_tensor - mean_tensor

# Computing the SVD
num_of_components = 150
U, S, V = torch.svd(X_train_centered)
components = V[:, :num_of_components]
eigenfaces1 = components.T.reshape(num_of_components, h, w)

# Projecting into the PCA space
X_train_transformed1 = torch.mm(X_train_centered, components)
X_test_transformed1 = torch.mm(X_test_centered, components)

# Converting back to numpy array for compatibility with scikit-learn
X_train_transformed_np = X_train_transformed1.numpy()
X_test_transformed_np = X_test_transformed1.numpy()

# Continue with RandomForestClassifier as before
rf = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
rf.fit(X_train_transformed_np, y1_train)
predictions = rf.predict(X_test_transformed_np)

predictions1 = rf.predict(X_test_transformed_np)
correct = predictions1 == y1_test
total_test = len(X_test_transformed_np)

print("Total Testing", total_test)
print("Predictions", predictions)
print("Which Correct:", correct)
print("Total Correct:", np.sum(correct))
print("Accuracy:", np.sum(correct)/total_test)
print(classification_report(y1_test, predictions, target_names=target_names))

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces1.shape[0])]
plot_gallery(eigenfaces1, eigenface_titles, h, w)
plt.show()

