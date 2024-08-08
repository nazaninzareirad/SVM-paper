import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def rbf_kernel(x, c, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - c) ** 2)

def initial_projection(X, centroids, gamma=0.1):
    return np.array([[rbf_kernel(x, c, gamma) for c in centroids] for x in X])

class WLS_SVC:
    def __init__(self, C=1.0, max_iter=100, tol=1e-2, learning_rate=0.001):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
    
    def fit(self, X_proj, y):
        l, M = X_proj.shape
        self.w = np.zeros(M)
        self.b = 0
        y = y.astype(np.float64)
        
        for iteration in range(self.max_iter):
            margins = y * (np.dot(X_proj, self.w) + self.b)
            loss = 1 - margins
            loss[loss < 0] = 0
            hinge_loss = self.C * np.mean(loss)
            
            grad_w = self.w - self.C * np.dot((loss > 0) * y, X_proj) / l
            grad_b = -self.C * np.mean((loss > 0) * y)
            
            if np.linalg.norm(grad_w) < self.tol and abs(grad_b) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b
        
        else:
            print(f"Did not converge after {self.max_iter} iterations")
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b).astype(int)

def find_new_centroids(X, y, model, centroids, gamma=0.1):
    X_proj = initial_projection(X, centroids, gamma)
    predictions = model.predict(X_proj)
    errors = np.abs(y - predictions) > 0.1
    return X[errors]

def iterative_poker(X, y, initial_centroids, max_iterations=10, gamma=0.1):
    centroids = initial_centroids
    wls_svc = WLS_SVC(C=1.0, learning_rate=0.0001)  # Reduced learning rate
    
    for iteration in range(max_iterations):
        projection = initial_projection(X, centroids, gamma)
        wls_svc.fit(projection, y)
        
        new_centroids = find_new_centroids(X, y, wls_svc, centroids, gamma)
        if len(new_centroids) == 0:
            print(f"No new centroids found after {iteration + 1} iterations")
            break
        
        centroids = np.vstack([centroids, new_centroids])[:initial_centroids.shape[0] + 10]
    
    return wls_svc, centroids

def experiment_with_centroids(X_train, y_train, X_test, y_test, num_centroids_list, gamma=0.1):
    results = []
    for num_centroids in num_centroids_list:
        kmeans = KMeans(n_clusters=num_centroids, random_state=42, n_init=10).fit(X_train)
        initial_centroids = kmeans.cluster_centers_
        
        poker_model, final_centroids = iterative_poker(X_train, y_train, initial_centroids, gamma=gamma)
        poker_projection = initial_projection(X_test, final_centroids, gamma=gamma)
        
        poker_predictions = poker_model.predict(poker_projection)
        poker_accuracy = accuracy_score(y_test, poker_predictions)
        results.append((num_centroids, poker_accuracy))
        print(f"Num centroids: {num_centroids}, POKER Accuracy: {poker_accuracy}")
    
    return results

def visualize_features(X, y, centroids, model, gamma=0.1):
    X_proj = initial_projection(X, centroids, gamma)
    phi_plus = np.zeros(X_proj.shape[0])
    phi_minus = np.zeros(X_proj.shape[0])
    
    for i, w in enumerate(model.w):
        if w > 0:
            phi_plus += w * X_proj[:, i]
        elif w < 0:
            phi_minus += w * X_proj[:, i]
    
    y = np.array(y)
    
    if len(y) != len(phi_plus):
        print("Mismatch between y and phi_plus length")
        return
    
    plt.figure(figsize=(10, 8))
    plt.scatter(phi_plus[y == 1], phi_minus[y == 1], c='b', label='Class 1')
    plt.scatter(phi_plus[y == -1], phi_minus[y == -1], c='r', label='Class -1')
    plt.xlabel('φ+')
    plt.ylabel('φ-')
    plt.legend()
    plt.title('Projection onto (φ+, φ-) space')
    plt.show()

def visualize_decision_boundary(X, y, centroids, model, gamma=0.1):
    if X.shape[1] != 2:
        print("Decision boundary visualization is only available for 2D input space.")
        return
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    X_proj = initial_projection(X_mesh, centroids, gamma)
    Z = model.predict(X_proj)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

def visualize_contour(X, centroids, model, gamma=0.1, positive=True):
    if X.shape[1] != 2:
        print("Contour visualization is only available for 2D input space.")
        return
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    X_proj = initial_projection(X_mesh, centroids, gamma)
    
    Z = np.zeros(X_proj.shape[0])
    for i, w in enumerate(model.w):
        if positive and w > 0:
            Z += w * X_proj[:, i]
        elif not positive and w < 0:
            Z += w * X_proj[:, i]
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1], c='k', s=10)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Contour Plot of ' + ('Positive' if positive else 'Negative') + ' Contributions')
    plt.show()

try:
    # Generate concentric circles dataset
    X_circles, y_circles = make_circles(n_samples=500, noise=0.1, factor=0.2, random_state=42)
    y_circles = 2 * y_circles - 1  # Convert to {-1, 1}

    # Initial centroids for POKER
    kmeans_circles = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_circles)
    initial_centroids_circles = kmeans_circles.cluster_centers_

    # Run POKER on circles dataset
    poker_model_circles, final_centroids_circles = iterative_poker(X_circles, y_circles, initial_centroids_circles)

    # Visualization of the results
    visualize_decision_boundary(X_circles, y_circles, final_centroids_circles, poker_model_circles)
    visualize_features(X_circles, y_circles, final_centroids_circles, poker_model_circles)
    visualize_contour(X_circles, final_centroids_circles, poker_model_circles, positive=True)
    visualize_contour(X_circles, final_centroids_circles, poker_model_circles, positive=False)

except Exception as e:
    print(f"An error occurred during visualization: {e}")

# Further datasets and experiments
datasets = {
    'abalone': ('abalone', 'target'),
    'image': ('segment', 'class'),
    'pima': ('diabetes', 'class'),
    'waveform': ('waveform-5000', 'class')
}

results_list = []

for dataset_name, (data_name, target_name) in datasets.items():
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
        
        # Experiment with different numbers of centroids
        num_centroids_list = [5, 10, 20]
        results = experiment_with_centroids(X_train, y_train, X_test, y_test, num_centroids_list)
        
        for num_centroids, accuracy in results:
            results_list.append({
                "dataset": dataset_name,
                "num_centroids": num_centroids,
                "accuracy": accuracy
            })
    
    except Exception as e:
        print(f"An error occurred while processing {dataset_name}: {e}")

# Convert results list to DataFrame
results_df = pd.DataFrame(results_list)

# Save results to a CSV file
results_df.to_csv('poker_results.csv', index=False)
