import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

def rbf_kernel(x, c, gamma=0.1):
  """Compute the RBF kernel between a sample and a centroid."""
  return np.exp(-gamma * np.linalg.norm(x - c) ** 2)

def initial_projection(X, centroids, gamma=0.1):
  """Project the input data X onto the space defined by centroids using the RBF kernel."""
  return np.array([[rbf_kernel(x, c, gamma) for c in centroids] for x in X])

class WLS_SVC:
  def __init__(self, C=100, max_iter=5000, tol=1e-2, learning_rate=0.01):
    self.C = C
    self.max_iter = max_iter
    self.tol = tol
    self.learning_rate = learning_rate

  def fit(self, X, y): #fit and predict function / maybe we should change them
    self.w = np.zeros(X.shape[1])
    self.b = 0



    grad_w = None
    grad_b = None
    for iteration in range(self.max_iter):
      """
      loss>0 , loss[loss <0]
      """ # what is this exactly?
      margins = y * (np.dot(X, self.w) + self.b)
      loss = 1 - margins
      loss[loss < 0] = 0

      grad_w = self.w - self.C * np.dot((loss > 0) * y, X) / len(X)
      grad_b = -self.C * np.mean((loss > 0) * y)

      self.w -= self.learning_rate * grad_w
      self.b -= self.learning_rate * grad_b

      #print(grad_w , grad_b)

    #print(grad_w , grad_b)
    # Check for convergence after all iterations
    if np.linalg.norm(grad_w) < self.tol and abs(grad_b) < self.tol: # 2 tolerances? (a different one for the matrix)
      print(f"Converged after {iteration + 1} iterations")
    else:
      print(f"Did not converge after {self.max_iter} iterations")

  def predict(self, X):
    return np.sign(np.dot(X, self.w) + self.b).astype(int)

# Example usage
'''X = np.array([[1, 2], [2, 3], [3, 1], [4, 4], [5, 2]])
y = np.array([1, 1, -1, -1, 1])

model = WLS_SVC()
model.fit(X, y)
predictions = model.predict(X)
print(predictions)'''

# Load and preprocess your datasets
# ... (replace with your data loading and preprocessing code)

# Create a list of datasets to process
datasets = [
  ('image', 'segment', 'class'),
  #('pima', 'diabetes', 'class'),
  ('waveform', 'waveform-5000', 'class')
]

for dataset_name, data_name, target_name in datasets:
  print(f"Processing dataset: {dataset_name}")

  try:
    # Load data
    data = fetch_openml(name=data_name, version=1, as_frame=True, parser='auto')
    X, y = data.data, data.target

    # Convert to binary classification if needed
    if dataset_name in ['abalone', 'image', 'waveform']:
      y = (y == y.unique()[0]).astype(int) * 2 - 1

    # Preprocess data
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=[object]).columns.tolist()

    preprocessor = ColumnTransformer(
      transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
      ]
    )

    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Ensure data dimensions match
    if X_train.shape[1] != X_test.shape[1]:
      raise ValueError("Number of features in training and testing data must match.")

    # Train the WLS-SVC model
    model = WLS_SVC(C=100, max_iter=5000, learning_rate=0.01)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

  except Exception as e:
    print(f"Error processing dataset {dataset_name}: {e}")