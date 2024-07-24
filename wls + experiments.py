import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

Train_df = pd.DataFrame(np.column_stack([X_train, y_train]), columns=iris.feature_names + ['target'])

# select initial centroids using k-means clustering
def kmeans_initial_centroids(X, n_clusters=6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X)
    return kmeans.cluster_centers_

# Experiment with different numbers of initial centroids
num_centroids = 6
initial_centroids = kmeans_initial_centroids(X_train, n_clusters=num_centroids)

# RBF Kernel
def rbf_kernel(x, c, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x - c)**2)

# Projects the data X into a new space defined by the RBF kernel with respect to the centroids.
def initial_projection(X, centroids, gamma=0.1):
    M = len(centroids)
    projection = np.zeros((X.shape[0], M))
    for i, x in enumerate(X):
        for j, c in enumerate(centroids):
            projection[i, j] = rbf_kernel(x, c, gamma)
    return projection

# weighted least squares support vector classifier
class WLS_SVC:
    def __init__(self, C=1.0):
        self.C = C
        self.alpha = None
        self.w = None
        self.b = 0
    
    def fit(self, X_proj, y):
        l, M = X_proj.shape
        self.alpha = np.zeros(l)
        self.w = np.zeros(M)
        self.b = 0

        for _ in range(10):
            for i in range(l):
                e_i = y[i] - (X_proj[i] @ self.w + self.b)
                a_i = 2 * self.alpha[i] * y[i] / (1 + y[i] * (X_proj[i] @ self.w + self.b))
                self.w += a_i * y[i] * X_proj[i]
                self.b += a_i * y[i]
                self.alpha[i] = min(max(self.alpha[i] + e_i, 0), self.C)
    
    def predict(self, X):
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(f"Dimension mismatch: X.shape[1]={X.shape[1]}, self.w.shape[0]={self.w.shape[0]}")
        return np.sign(X @ self.w + self.b)

def find_new_centroids(X, y, model, centroids, gamma=0.1):
    X_proj = initial_projection(X, centroids, gamma)
    #print(f"X_proj shape: {X_proj.shape}")

    predictions = model.predict(X_proj)
    support_vectors = X[np.abs(y - predictions) > 0.1]
    #print(f"Support vectors shape: {support_vectors.shape}")

    return support_vectors

def iterative_poker(X, y, initial_centroids, iterations=10, gamma=0.1):
    centroids = initial_centroids
    wls_svc = WLS_SVC(C=1.0)
    
    for iteration in range(iterations):
        projection = initial_projection(X, centroids, gamma)
        #print(f"Iteration {iteration} projection shape: {projection.shape}")
        wls_svc.fit(projection, y)
        
        new_centroids = find_new_centroids(X, y, wls_svc, centroids, gamma)

        if len(new_centroids) == 0:
            break
        
        total_centroids = np.vstack([centroids, new_centroids])
        centroids = total_centroids[:initial_centroids.shape[0] + 10]
        #print(f"Updated centroids shape: {centroids.shape}")

    return wls_svc, centroids

X_train_ = Train_df[iris.feature_names].values
y_train_ = Train_df['target'].values
y_train_ = 2 * y_train_ - 1
y_test_ = 2 * y_test - 1

poker_model, final_centroids = iterative_poker(X_train_, y_train_, initial_centroids)
#print(f"Final centroids shape: {final_centroids.shape}")

poker_projection = initial_projection(X_test, final_centroids)
#print(f"Test projection shape: {poker_projection.shape}")

if poker_projection.shape[1] != poker_model.w.shape[0]:
    raise ValueError(f"Dimension mismatch: poker_projection.shape[1]={poker_projection.shape[1]}, poker_model.w.shape[0]={poker_model.w.shape[0]}")

poker_predictions = poker_model.predict(poker_projection)
poker_accuracy = accuracy_score(y_test_, poker_predictions)

svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_, y_train_)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test_, svm_predictions)

print(f"POKER Accuracy: {poker_accuracy}")
print(f"SVM Accuracy: {svm_accuracy}")

def experiment_with_centroids(X_train, y_train, X_test, y_test, num_centroids_list, gamma=0.1):
    results = []
    for num_centroids in num_centroids_list:
        initial_centroids = kmeans_initial_centroids(X_train, n_clusters=num_centroids)
        poker_model, final_centroids = iterative_poker(X_train, y_train, initial_centroids, gamma=gamma)
        poker_projection = initial_projection(X_test, final_centroids, gamma=gamma)
        
        if poker_projection.shape[1] != poker_model.w.shape[0]:
            raise ValueError(f"Dimension mismatch: poker_projection.shape[1]={poker_projection.shape[1]}, poker_model.w.shape[0]={poker_model.w.shape[0]}")
        
        poker_predictions = poker_model.predict(poker_projection)
        poker_accuracy = accuracy_score(y_test, poker_predictions)
        results.append((num_centroids, poker_accuracy))
        print(f"Num centroids: {num_centroids}, POKER Accuracy: {poker_accuracy}")
    
    return results

num_centroids_list = [4, 6, 8, 10, 12]
experiment_results = experiment_with_centroids(X_train_, y_train_, X_test, y_test_, num_centroids_list)

for num_centroids, acc in experiment_results:
    print(f"Num centroids: {num_centroids}, POKER Accuracy: {acc}")
