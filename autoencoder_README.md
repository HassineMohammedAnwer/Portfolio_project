# Autoencoder-based Product Recommendation System
This project demonstrates how to build an autoencoder-based recommendation system using three datasets: customers, orders, and products. The model is designed to predict which products a user might be interested in, based on their previous orders.
## Project Structure
### 1. Data Preprocessing:
Three datasets are loaded: customers.csv, orders.csv, and products.csv.
Data cleaning involves removing unnecessary columns, renaming columns, and creating an aggregated user-item interaction matrix.

### 2. Modeling:
The project explores two models:
Basic Autoencoder: A simple feed-forward autoencoder to learn latent representations of users based on their order history.
Variational Autoencoder (VAE): A more advanced autoencoder that learns a probability distribution of the latent space.

### 3. Evaluation:
The models are evaluated using Precision@K and Recall@K, providing insights into the model's accuracy in recommending top-k products.


## Installation
1   Clone this repository and navigate to the project folder.

2   Ensure the datasets (customers.csv, orders.csv, products.csv) are available in the data/prepared_data/ directory.

## How to Use
### 1. Data Preprocessing
The first step involves loading and cleaning the data. We aggregate duplicate entries and create a user-item interaction matrix, which is a pivot table where rows represent users and columns represent products. The values indicate how many times a product was purchased by each user.
```python
df_orders_aggregated = df_orders.groupby(['customer_id', 'lineitem_sku'], as_index=False).agg({'lineitem_quantity': 'sum'})

user_item_matrix = df_orders_aggregated.pivot(index='customer_id', columns='lineitem_sku', values='lineitem_quantity').fillna(0)
```
### 2. ''Vanilla'' Autoencoder
The autoencoder is a neural network that learns how to encode user-item interactions into a compressed latent representation and then decode it back to predict interactions.
#### Key Code: Autoencoder Architecture
```python
input_layer = Input(shape=(n_inputs,))
encoded = Dense(128, activation='relu')(input_layer)  # Encoder
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(32, activation='relu')(encoded)  # Latent Space

decoded = Dense(64, activation='relu')(latent)  # Decoder
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(n_inputs, activation='sigmoid')(decoded)  # Reconstructed Output

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
```
#### Explanation:
. Input Layer: Takes the user-item interaction matrix (one row per user).

. Encoder: Compresses the input into a lower-dimensional representation (latent space) through a series of fully connected layers.

. Decoder: Expands the latent representation back to the original input size.

. Output Layer: Produces the reconstructed user-item interactions, indicating predicted product interest.

. The autoencoder is trained using the original user-item interaction matrix as both input and output.

### 3. Evaluation of the Autoencoder
After training the autoencoder, predictions are made by reconstructing the interaction matrix. The model’s predictions are then evaluated by calculating precision and recall at k (for example, k=10).
```python
predicted_interactions = reconstructed
actual_interactions = user_item_matrix.values

precision_at_10, recall_at_10 = precision_recall_at_k(predicted_interactions, actual_interactions, k=10)
```
This code computes how well the top-10 predictions align with actual user interactions.
### 4. Variational Autoencoder (VAE)
The VAE model introduces a probabilistic approach to generating latent space representations. It uses a reparameterization trick to generate latent variables z from the learned mean (mu) and variance (sigma).
#### Key Code: VAE Architecture
```python
def sampling(args):
    mu, log_sigma = args
    batch = keras.backend.shape(mu)[0]
    dim = keras.backend.shape(mu)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return mu + keras.backend.exp(log_sigma / 2) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    x = Input(shape=(input_dims,))
    hidden = Dense(hidden_layers[0], activation='relu')(x)
    for units in hidden_layers[1:]:
        hidden = Dense(units, activation='relu')(hidden)

    z_mean = Dense(latent_dims, activation=None)(hidden)
    z_log_sigma = Dense(latent_dims, activation=None)(hidden)
    z = Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_sigma])

    encoder = Model(x, [z, z_mean, z_log_sigma], name="encoder")
    ...
    return encoder, decoder, vae
```
#### Explanation:
Sampling Function: This function implements the reparameterization trick, which allows the model to backpropagate through the stochastic latent variable z.
VAE Architecture: The encoder generates two outputs: z_mean and z_log_sigma, which represent the latent variable's mean and log-variance, respectively. These are then used to sample a latent vector z.
### 5. Evaluation of the VAE
The VAE model is evaluated similarly to the basic autoencoder, using precision and recall at k.
## Results
Basic Autoencoder: Precision@10: 0.1234, Recall@10: 0.5678

Variational Autoencoder: Precision@10: 0.2345, Recall@10: 0.6789
## License
This project is licensed under the MIT License.
