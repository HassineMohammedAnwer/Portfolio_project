# Project Structure

## 1. Data Exploration and Cleaning (`data_exploration.ipynb`)
In this notebook, data related to customers, products, and orders are explored, cleaned, and visualized. This step helps handle irrelevant or missing data and extract important features. After cleaning, the processed data is saved for future use:

```python
df_customers.to_csv('data/cleaned_data/customers.csv', index=False)
df_products.to_csv('data/cleaned_data/products.csv', index=False)
```

## 2. Data Preparation (data_preparation.ipynb)
In this notebook, the cleaned data is preprocessed for modeling. The process includes normalizing numerical features, encoding categorical features, and generating matrices for the recommendation models. The preprocessed data is saved:

```python
df_products.to_csv('data/prepared_data/products.csv', index=False)
df_orders.to_csv('data/prepared_data/orders.csv', index=False)
```

## 3. Content-Based Modeling (final_modeling_content-based.ipynb)
A content-based filtering approach is applied using product features and customer interactions. PCA (Principal Component Analysis) is used to reduce data dimensionality. Different similarity measures are evaluated:

Cosine Similarity
Euclidean Distance
Pearson Correlation
Mahalanobis Distance
The performance is evaluated using Precision@5, Recall@5, and F1-Score.

## 4. Collaborative Filtering Modeling (collaborative_filtering_modeling_final.ipynb)
This notebook covers collaborative filtering techniques, such as:

User-Based Collaborative Filtering
Item-Based Collaborative Filtering
Singular Value Decomposition (SVD)
Alternating Least Squares (ALS)
Evaluation metrics include Precision, Recall, and F1-Score.

## 5. Autoencoder-Based Modeling (autoencoder.ipynb)
Deep learning models using autoencoders are explored in this notebook. Two types of autoencoders are implemented:

Basic Autoencoder
Learns compressed representations of user-product interactions and reconstructs the user-item matrix.

```python
input_layer = Input(shape=(n_inputs,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(latent)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(n_inputs, activation='sigmoid')(decoded)
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
```

Variational Autoencoder (VAE)
A more complex model with regularization to ensure a continuous latent space for better recommendations.

Both models are evaluated using Precision@K and Recall@K.

# How to Run the Project
## Data Exploration and Cleaning:
Run data_exploration.ipynb to load, clean, and explore the dataset.
Save the cleaned data to the data/cleaned_data/ directory.
## Data Preparation:
Run data_preparation.ipynb to preprocess the cleaned data.
Save the preprocessed data to data/prepared_data/.

## Content-Based Modeling:
Run final_modeling_content-based.ipynb to test content-based filtering techniques.
Evaluate performance using metrics such as Precision@K and Recall@K.

## Collaborative Filtering:
Run collaborative_filtering_modeling_final.ipynb to test collaborative filtering approaches (SVD, ALS, etc.).
Measure the performance using precision, recall, and F1 scores.

## Autoencoder-Based Modeling:
Run autoencoder.ipynb to train both basic and variational autoencoder models.
Evaluate performance using Precision@K and Recall@K.
Results
Each method was evaluated using Precision@K, Recall@K, and F1-Score. Below are the final performance results:

Model	Precision@5	Recall@5	F1-Score
Content-Based (Cosine)	0.085	0.0233	0.0113
Collaborative Filtering (SVD)	0.6143	0.4378	0.511
Autoencoder	Varies	Varies	Moderate

# Conclusion
This project demonstrates a variety of recommendation techniques, including content-based filtering, collaborative filtering, and deep learning models like autoencoders. Each method has its strengths and limitations, and further work can focus on combining these methods into a hybrid recommendation system for improved accuracy and personalization.

# Author
This project was developed as part of my portfolio, demonstrating my expertise in machine learning, data preprocessing, and recommendation system development. Connect with me on LinkedIn(link) to discuss this project or explore collaboration opportunities!

# License
This project is licensed under the MIT License.
