import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from factorization_machine import FactorizationMachineModel
from instacart_dataset import InstacartDataset
from torch.utils.data import DataLoader

K_FACTORS = 32

orders = pd.read_csv('./archive/orders.csv')
products = pd.read_csv('./archive/products.csv')
order_products_prior = pd.read_csv('./archive/order_products__prior.csv')
order_products_train = pd.read_csv('./archive/order_products__train.csv')

# Merge orders with prior order products on 'order_id' -> Link users with their prior orders
merged_df = pd.merge(orders, order_products_prior, on="order_id")
merged_df

# Merge the result dataframe with products on 'product_id' -> link orders with product details
merged_df = pd.merge(merged_df, products, on='product_id')

train_orders = orders[orders['eval_set']=="train"]
train_df = pd.merge(train_orders, order_products_train, on='order_id')

user_history = merged_df.groupby('user_id')['product_id'].apply(set).reset_index()
user_history.rename(columns={"product_id":"all_products"}, inplace=True)

# List of products that were in each user's training cart
train_products = train_df.groupby('user_id')['product_id'].apply(set).reset_index()
train_products.rename(columns={"product_id":"train_products"}, inplace=True)

user_data = pd.merge(user_history, train_products, on="user_id")

# Products that are not in training order but are in history -> negative samples
user_data['negative_samples'] = user_data.apply(lambda row: row['all_products'] - row['train_products'], axis=1)


positive_samples = [] # Products that were reordered
negative_samples = [] # Products that weren't reordered

for index, row in user_data.iterrows():
    for product in row['train_products']:
        positive_samples.append({'user_id': row['user_id'], 'product_id':product, 'y':1})

    for product in row['negative_samples']:
        negative_samples.append({'user_id': row['user_id'], 'product_id':product, 'y':0})

positive_df = pd.DataFrame(positive_samples)
negative_df = pd.DataFrame(negative_samples)

final_train_df = pd.concat([positive_df, negative_df]).reset_index(drop=True)

global_user_map = {user_id: i for i, user_id in enumerate(final_train_df['user_id'].unique())}
global_product_map = {product_id: i for i, product_id in enumerate(final_train_df['product_id'].unique())}

all_user_ids = final_train_df['user_id'].unique()

train_user_ids, test_user_ids = train_test_split(all_user_ids, test_size=0.2, random_state=42)

# Creating the test_dataloader
test_data = final_train_df[final_train_df['user_id'].isin(test_user_ids)]
test_dataset = InstacartDataset(test_data, user_map=global_user_map, product_map=global_product_map)
test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FactorizationMachineModel(
    num_users=len(global_user_map),
    num_products=len(global_product_map),
    k=K_FACTORS
)
model.load_state_dict(torch.load("fm_model.pth"))
model.to(device)
model.eval()

print("Predicting on Test set")
predictions = []

# Inference loop
with torch.no_grad():
    for batch in test_dataloader:
        users = batch['user'].to(device)
        products = batch['product'].to(device)
        
        # Get the raw scores (logits) from the model
        outputs = model(users, products)
        
        # Store the original IDs and the predictions
        # We need the original IDs for grouping later
        original_user_ids = test_dataset.df['user_id'].iloc[batch['user'].cpu().numpy()]
        original_product_ids = test_dataset.df['product_id'].iloc[batch['product'].cpu().numpy()]
        
        for i in range(len(outputs)):
            predictions.append({
                'user_id': original_user_ids.iloc[i],
                'product_id': original_product_ids.iloc[i],
                'score': outputs[i].item(),
                'label': batch['label'][i].item()
            })

pred_df = pd.DataFrame(predictions)
print("Finished getting predictions")

def calculate_average_precision_at_k(k, ranked_list, true_set):
    """Calculates Average Precision at K."""
    hits = 0
    precision_sum = 0
    for i, p_id in enumerate(ranked_list):
        if p_id in true_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            precision_sum += precision_at_i
    
    if not true_set:
        return 0.0
        
    return precision_sum / min(len(true_set), k)


def ndcg_at_k(predictions, labels, k=10):
    """
    Calculates Normalized Discounted Cumulative Gain at K.
    Args:
        predictions (np.array): Array of predicted scores.
        labels (np.array): Array of true labels (0 or 1).
        k (int): The cutoff for the metric.
    """
    k = min(k, len(predictions))
    # Convert to PyTorch tensors for calculation
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    # Sort labels by prediction score
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_labels = labels[sorted_indices]
    
    # DCG@k
    discounts = torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    dcg = torch.sum(sorted_labels[:k] / discounts)
    
    # IDCG@k
    ideal_labels = torch.sort(labels, descending=True)[0]
    idcg = torch.sum(ideal_labels[:k] / discounts)
    
    if idcg == 0:
        return 0.0
    return (dcg / idcg).item()


# EVALUATION LOOP
print("Calculating ranking metrics...\n")

K = 10
user_metrics = []

# Group by user to evaluate each user's ranked list
for user_id, group in pred_df.groupby('user_id'):
    true_reorders = set(group[group['label'] == 1]['product_id'])
    if not true_reorders:
        continue
        
    group = group.sort_values('score', ascending=False)
    top_k_preds = list(group.head(K)['product_id'])
    
    hits = len(true_reorders.intersection(set(top_k_preds)))
    
    # --- ADD NDCG CALCULATION HERE ---
    user_ndcg = ndcg_at_k(group['score'].values, group['label'].values, k=K)

    user_metrics.append({
        'precision_at_k': hits / K,
        'recall_at_k': hits / len(true_reorders),
        'map_at_k': calculate_average_precision_at_k(K, top_k_preds, true_reorders),
        'ndcg_at_k': user_ndcg # Add the new metric
    })

metrics_df = pd.DataFrame(user_metrics)
mean_metrics = metrics_df.mean()

print("--- Evaluation Results ---")
print(f"Mean Precision@{K}: {mean_metrics['precision_at_k']:.4f}")
print(f"Mean Recall@{K}:    {mean_metrics['recall_at_k']:.4f}")
print(f"MAP@{K}:              {mean_metrics['map_at_k']:.4f}")
print(f"NDCG@{K}:             {mean_metrics['ndcg_at_k']:.4f}")