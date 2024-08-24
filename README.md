# fraudbust

## Overview

This project develops a Financial Fraud Detection System that identifies fraudulent transactions in financial data using autoencoders and clustering algorithms. The system trains an autoencoder on legitimate transactions to capture normal patterns and detects anomalies in the latent space that could indicate fraudulent activities.

## Key Features

- **Autoencoders**: Trains an autoencoder to learn normal transaction patterns and reconstruct data.
- **Clustering Algorithms**: Applies K-Means and DBSCAN clustering on the latent space to detect anomalies.
- **Latent Space Refinement**: Uses Latent Semantic Analysis (LSA) and Latent Discriminant Analysis (LDA) to improve clustering results.
- **Visualization**: Implements PCA and t-SNE to visualize high-dimensional data and clustering results.

## Dataset

- **Dataset Used**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.
- **Description**: Contains transactions made by credit cards in September 2013 by European cardholders.

## Usage

### Clone the Repository

```bash
git clone https://github.com/yourusername/repository-name.git
cd repository-name ```

### Run Code

```bash
python fraud_detection.py
