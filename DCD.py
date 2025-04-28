import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def generate_synthetic_data(n_samples=300, n_features=2, n_centers=4, cluster_std=0.60, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    return data

# 2. Preprocess Data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# 3. Train Clustering Model
def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels

# 4. Evaluate Clustering
def evaluate_clustering(data, labels):
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score

# 5. Visualize Clusters
def visualize_clusters(data, labels, model):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='Set2', s=60)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.title('Cluster Visualization with KMeans')
    plt.legend()
    plt.show()

# 6. Main Pipeline
def main():
    data = generate_synthetic_data(n_samples=300, n_features=2, n_centers=4)
    print("Synthetic Data Sample:")
    print(data.head())

    # Step 2: Preprocess
    processed_data = preprocess_data(data)

    # Step 3: Train KMeans
    model, labels = train_kmeans(processed_data, n_clusters=4)

    # Step 4: Evaluate
    evaluate_clustering(processed_data, labels)

    # Step 5: Visualize
    visualize_clusters(processed_data, labels, model)

if __name__ == "__main__":
    main()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def generate_synthetic_data(n_samples=300, n_features=2, n_centers=4, cluster_std=0.60, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    return data

# 2. Preprocess Data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# 3. Train Clustering Model
def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels

# 4. Evaluate Clustering
def evaluate_clustering(data, labels):
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score

# 5. Visualize Clusters
def visualize_clusters(data, labels, model):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='Set2', s=60)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.title('Cluster Visualization with KMeans')
    plt.legend()
    plt.show()

# 6. Main Pipeline
def main():
   
    data = generate_synthetic_data(n_samples=300, n_features=2, n_centers=4)
    print("Synthetic Data Sample:")
    print(data.head())

    # Step 2: Preprocess
    processed_data = preprocess_data(data)

    # Step 3: Train KMeans
    model, labels = train_kmeans(processed_data, n_clusters=4)

    # Step 4: Evaluate
    evaluate_clustering(processed_data, labels)

    # Step 5: Visualize
    visualize_clusters(processed_data, labels, model)

if __name__ == "__main__":
    main()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def generate_synthetic_data(n_samples=300, n_features=2, n_centers=4, cluster_std=0.60, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    return data

# 2. Preprocess Data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# 3. Train Clustering Model
def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels

# 4. Evaluate Clustering
def evaluate_clustering(data, labels):
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score

# 5. Visualize Clusters
def visualize_clusters(data, labels, model):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='Set2', s=60)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.title('Cluster Visualization with KMeans')
    plt.legend()
    plt.show()

# 6. Main Pipeline
def main():
   
    data = generate_synthetic_data(n_samples=300, n_features=2, n_centers=4)
    print("Synthetic Data Sample:")
    print(data.head())

    # Step 2: Preprocess
    processed_data = preprocess_data(data)

    # Step 3: Train KMeans
    model, labels = train_kmeans(processed_data, n_clusters=4)

    # Step 4: Evaluate
    evaluate_clustering(processed_data, labels)

    # Step 5: Visualize
    visualize_clusters(processed_data, labels, model)

if __name__ == "__main__":
    main()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def generate_synthetic_data(n_samples=300, n_features=2, n_centers=4, cluster_std=0.60, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    return data

# 2. Preprocess Data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# 3. Train Clustering Model
def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels

# 4. Evaluate Clustering
def evaluate_clustering(data, labels):
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score

# 5. Visualize Clusters
def visualize_clusters(data, labels, model):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='Set2', s=60)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.title('Cluster Visualization with KMeans')
    plt.legend()
    plt.show()

# 6. Main Pipeline
def main():
  
    data = generate_synthetic_data(n_samples=300, n_features=2, n_centers=4)
    print("Synthetic Data Sample:")
    print(data.head())

    # Step 2: Preprocess
    processed_data = preprocess_data(data)

    # Step 3: Train KMeans
    model, labels = train_kmeans(processed_data, n_clusters=4)

    # Step 4: Evaluate
    evaluate_clustering(processed_data, labels)

    # Step 5: Visualize
    visualize_clusters(processed_data, labels, model)

if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def generate_synthetic_data(n_samples=300, n_features=2, n_centers=4, cluster_std=0.60, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    return data

# 2. Preprocess Data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# 3. Train Clustering Model
def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels

# 4. Evaluate Clustering
def evaluate_clustering(data, labels):
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score

# 5. Visualize Clusters
def visualize_clusters(data, labels, model):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='Set2', s=60)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.title('Cluster Visualization with KMeans')
    plt.legend()
    plt.show()

# 6. Main Pipeline
def main():
    data = generate_synthetic_data(n_samples=300, n_features=2, n_centers=4)
    print("Synthetic Data Sample:")
    print(data.head())

    # Step 2: Preprocess
    processed_data = preprocess_data(data)

    # Step 3: Train KMeans
    model, labels = train_kmeans(processed_data, n_clusters=4)

    # Step 4: Evaluate
    evaluate_clustering(processed_data, labels)

    # Step 5: Visualize
    visualize_clusters(processed_data, labels, model)

if __name__ == "__main__":
    main()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def generate_synthetic_data(n_samples=300, n_features=2, n_centers=4, cluster_std=0.60, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    return data

# 2. Preprocess Data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# 3. Train Clustering Model
def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels

# 4. Evaluate Clustering
def evaluate_clustering(data, labels):
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score

# 5. Visualize Clusters
def visualize_clusters(data, labels, model):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='Set2', s=60)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.title('Cluster Visualization with KMeans')
    plt.legend()
    plt.show()

# 6. Main Pipeline
def main():
    data = generate_synthetic_data(n_samples=300, n_features=2, n_centers=4)
    print("Synthetic Data Sample:")
    print(data.head())

    # Step 2: Preprocess
    processed_data = preprocess_data(data)

    # Step 3: Train KMeans
    model, labels = train_kmeans(processed_data, n_clusters=4)

    # Step 4: Evaluate
    evaluate_clustering(processed_data, labels)

    # Step 5: Visualize
    visualize_clusters(processed_data, labels, model)

if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def generate_synthetic_data(n_samples=300, n_features=2, n_centers=4, cluster_std=0.60, random_state=42):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    return data

# 2. Preprocess Data
def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# 3. Train Clustering Model
def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return model, labels

# 4. Evaluate Clustering
def evaluate_clustering(data, labels):
    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score

# 5. Visualize Clusters
def visualize_clusters(data, labels, model):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='Set2', s=60)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.title('Cluster Visualization with KMeans')
    plt.legend()
    plt.show()

# 6. Main Pipeline
def main():
   
    data = generate_synthetic_data(n_samples=300, n_features=2, n_centers=4)
    print("Synthetic Data Sample:")
    print(data.head())

    # Step 2: Preprocess
    processed_data = preprocess_data(data)

    # Step 3: Train KMeans
    model, labels = train_kmeans(processed_data, n_clusters=4)

    # Step 4: Evaluate
    evaluate_clustering(processed_data, labels)

    # Step 5: Visualize
    visualize_clusters(processed_data, labels, model)

if __name__ == "__main__":
    main()
