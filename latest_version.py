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
    def __init__(self, C=1.0, max_iter=10, tol=1e-3, learning_rate=0.01):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
    
    def fit(self, X_proj, y):
        """Fit the WLS-SVC model to the projected data."""
        l, M = X_proj.shape
        self.w = np.zeros(M)
        self.b = 0
        y = y.astype(np.float64)
        
        for iteration in range(self.max_iter):
            margins = y * (np.dot(X_proj, self.w) + self.b)
            loss = 1 - margins
            loss[loss < 0] = 0
            
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
        """Predict the class labels for the input data."""
        return np.sign(np.dot(X, self.w) + self.b).astype(int)

def find_new_centroids(X, y, model, centroids, gamma=0.1, num_new_centroids=2):
    """Identify new centroids based on support vectors and their distance from existing centroids."""
    X_proj = initial_projection(X, centroids, gamma)
    margins = y * (np.dot(X_proj, model.w) + model.b)
    
    # Select support vectors close to the decision boundary
    support_vectors = X[(margins > 0) & (margins < 1)]
    
    if len(support_vectors) == 0:
        return np.array([])
    
    # Calculate distances from support vectors to existing centroids
    distances = np.array([np.min([np.linalg.norm(sv - c) for c in centroids]) for sv in support_vectors])
    
    # Select the farthest support vectors
    farthest_indices = np.argsort(distances)[::-1]
    farthest_support_vectors = support_vectors[farthest_indices]
    
    # Return top candidates for new centroids
    return farthest_support_vectors[:num_new_centroids]

def select_initial_centroids(X, y, M=2):
    """Select initial centroids from the training data."""
    unique_classes = np.unique(y)
    centroids = []
    
    for cls in unique_classes:
        class_samples = X[y == cls]
        selected_centroids = class_samples[np.random.choice(class_samples.shape[0], M//2, replace=False)]
        centroids.append(selected_centroids)
    
    return np.vstack(centroids)

def iterative_poker(X, y, initial_centroids, max_iterations=10, gamma=0.1):
    """Iteratively refine the POKER model by adding centroids and updating the SVC."""
    centroids = initial_centroids
    wls_svc = WLS_SVC(C=1.0, learning_rate=0.01, max_iter=10, tol=1e-3)
    
    centroids_history = [centroids.copy()]
    support_vectors_history = []
    
    for iteration in range(max_iterations):
        projection = initial_projection(X, centroids, gamma)
        wls_svc.fit(projection, y)
        
        # Store support vectors
        margins = y * (np.dot(projection, wls_svc.w) + wls_svc.b)
        support_vectors = X[(margins > 0) & (margins < 1)]
        support_vectors_history.append(support_vectors)
        
        # Visualize the decision boundary and feature space
        if (iteration + 1) % 5 == 0:
            visualize_decision_boundary(X, y, centroids, wls_svc, gamma)
            visualize_features(X, y, centroids, wls_svc, gamma)
        
        # Find and add new centroids
        new_centroids = find_new_centroids(X, y, wls_svc, centroids, gamma, num_new_centroids=2)
        if len(new_centroids) == 0:
            print(f"No new centroids found after {iteration + 1} iterations")
            break
        
        centroids = np.vstack([centroids, new_centroids])
        centroids_history.append(centroids.copy())
    
    return wls_svc, centroids, centroids_history, support_vectors_history

def visualize_centroids_and_support_vectors(centroids_history, support_vectors_history):
    """Visualize the evolution of centroids and support vectors during iterations."""
    num_iterations = len(centroids_history)
    
    for iteration in range(num_iterations):
        centroids = centroids_history[iteration]
        support_vectors = support_vectors_history[iteration]
        
        plt.figure(figsize=(12, 6))
        
        # Plot centroids
        plt.subplot(1, 2, 1)
        if centroids.shape[1] == 2:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='g', marker='x', label='Centroids')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Centroids at Iteration {iteration}')
            plt.legend()
        
        # Plot support vectors
        plt.subplot(1, 2, 2)
        if support_vectors.shape[1] == 2:
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='orange', marker='o', label='Support Vectors')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Support Vectors at Iteration {iteration}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

def visualize_features(X, y, centroids, model, gamma=0.1):
    """Visualize the data projected onto the (φ+, φ-) feature space."""
    X_proj = initial_projection(X, centroids, gamma)
    phi_plus = np.sum([w * X_proj[:, i] for i, w in enumerate(model.w) if w > 0], axis=0)
    phi_minus = np.sum([w * X_proj[:, i] for i, w in enumerate(model.w) if w < 0], axis=0)
    
    if np.isscalar(phi_plus) or np.isscalar(phi_minus):
        print("phi_plus or phi_minus is scalar, skipping plot")
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
    """Visualize the decision boundary in a 2D input space."""
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

# Load datasets
datasets = {
    'abalone': ('Class_number_of_rings'),
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
        
        # Experiment with different numbers of initial centroids
        M_list = [2, 4, 8]
        for M in M_list:
            initial_centroids = select_initial_centroids(X_train, y_train, M=M)
            
            poker_model, final_centroids, centroids_history, support_vectors_history = iterative_poker(
                X_train, y_train, initial_centroids, gamma=0.1
            )
            
            # Plot changes in centroids and support vectors
            visualize_centroids_and_support_vectors(centroids_history, support_vectors_history)
            
            poker_projection = initial_projection(X_test, final_centroids, gamma=0.1)
            poker_predictions = poker_model.predict(poker_projection)
            poker_accuracy = accuracy_score(y_test, poker_predictions)
            
            results_list.append({
                'Dataset': dataset_name,
                'M': M,
                'Accuracy': poker_accuracy
            })
            print(f"M: {M}, POKER Accuracy: {poker_accuracy}")
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")

results_df = pd.DataFrame(results_list)
print(results_df)
