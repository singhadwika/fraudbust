import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, LatentDirichletAllocation as LDA, TruncatedSVD as LSA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model, Sequential
from keras.layers import Dense, Input

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Preprocessing
X = df.drop(['Time', 'Amount', 'Class'], axis=1)  # Use V1 to V28 features
y = df['Class']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Autoencoder Model
input_dim = X_train.shape[1]
encoding_dim = 14  # compression factor of 2

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="tanh")(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

# Get latent space representation
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_train_encoded = encoder_model.predict(X_train)
X_test_encoded = encoder_model.predict(X_test)

# Clustering on Latent Space
# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train_encoded)
y_kmeans = kmeans.predict(X_test_encoded)
print(f"K-Means Accuracy: {accuracy_score(y_test, y_kmeans)}")
print(f"K-Means Silhouette Score: {silhouette_score(X_test_encoded, y_kmeans)}")

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.3, min_samples=10)
y_dbscan = dbscan.fit_predict(X_test_encoded)
y_dbscan = np.where(y_dbscan == -1, 1, 0)  # Treat noise points as frauds
print(f"DBSCAN Accuracy: {accuracy_score(y_test, y_dbscan)}")
print(f"DBSCAN Silhouette Score: {silhouette_score(X_test_encoded, y_dbscan)}")

# Refining Latent Space with LSA and LDA
# LSA
lsa = LSA(n_components=2)
X_lsa = lsa.fit_transform(X_train_encoded)

# LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_train_encoded, y_train)
