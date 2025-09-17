
# Instacart Market Basket Recommender System

This project aims to build and compare several recommendation algorithms to predict which products a user will reorder in their next shopping cart. The project uses the publicly available Instacart 2017 dataset from a Kaggle competition.

## Project Goal

The primary objective is to develop a model that, given a user's purchase history, can accurately rank the products they are most likely to reorder. This project explores a progression of models, starting with a strong Factorization Machine baseline and moving towards more complex deep learning architectures.

-----

## Dataset

This project uses the ["Instacart Market Basket Analysis" dataset](https://www.google.com/search?q=https://www.kaggle.com/c/instacart-market-basket-analysis), which contains over 3 million anonymized grocery orders from more than 200,000 users. The data is relational and split across six CSV files detailing users, products, orders, and their relationships.

-----

## Methodology

The recommendation task is framed as a supervised, pointwise Learning-to-Rank (LTR) problem.

1.  **Data Preprocessing**:

      * The raw CSV files (`orders.csv`, `products.csv`, `order_products__prior.csv`, etc.) were merged to create a unified DataFrame of all historical user-product interactions.
      * Positive samples were identified as products a user reordered in their final `train` order.
      * Negative samples were created from products a user had purchased in the past but did *not* reorder in their final `train` order.

2.  **Train/Validation Split**:

      * A **user-based split** (80% train, 20% validation) was performed.
      * This ensures that no user in the validation set appears in the training set, providing a robust test of the model's ability to handle the "user cold-start" problem.

-----

## Models Implemented

### 1\. Baseline: Factorization Machine (Pointwise LTR)

A Factorization Machine was implemented as the baseline model. This model is designed to learn the interaction effects between users and products in a high-dimensional, sparse feature space.

  * **Framework**: PyTorch
  * **Architecture**:
      * `torch.nn.Embedding` layers were used to create dense latent vectors for users and products.
      * The model predicts a score for each `(user, product)` pair based on the FM formula (global bias + linear terms + dot product of latent vectors).
  * **Training Objective**: The model was trained as a binary classifier using `BCEWithLogitsLoss` to predict the probability of a product being reordered.

-----

## Current Results

The baseline Factorization Machine was trained for 5 epochs and evaluated on the held-out validation set of users.

| Metric          | Score  |
| --------------- | :----: |
| **Mean Precision@10** | 0.2089 |
| **Mean Recall@10** | 0.1528 |
| **MAP@10** | 0.1183 |
| **NDCG@10** | 0.2082 |

These results serve as the benchmark for all future, more complex models.

-----

## Future Work

The next steps in this project are to implement and compare more advanced models:

  * [ ] **Pairwise FM**: Adapt the current model to use a pairwise loss (like BPR) to directly optimize ranking.
  * [ ] **Two-Towers Model**: Build a two-towers architecture for efficient retrieval.
  * [ ] **DIN (Deep Interest Network)**: Implement an attention mechanism to model user interests based on their purchase history.
  * [ ] **Transformer-based Model**: Use a sequential model like SASRec to capture the order of user purchases.

-----

## Setup and Usage

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/Dhyanesh18/instacart-recommender-benchmark
    cd instacart-recommender-benchmark
    ```

2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the data**:

      * Download the Instacart dataset from [Kaggle](https://www.google.com/search?q=https://www.kaggle.com/c/instacart-market-basket-analysis).
      * Unzip and place the six CSV files into a folder named `./archive/`.

4.  **Run the scripts**:
      * **Training**: `python train.py` (This will train the baseline model and print training metrics).
      * **Validation**: `python validation.py` (This will run inference on the model and print evaluation metrics).
