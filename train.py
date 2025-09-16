import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from factorization_machine import FactorizationMachineModel
from instacart_dataset import InstacartDataset
from torch.utils.data import DataLoader

# Training config
K_FACTORS = 32
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 1e-3


print("Loading dataframes...")

# The CSV files are inside a folder named "archive" within the working directory
orders = pd.read_csv('./archive/orders.csv')
products = pd.read_csv('./archive/products.csv')
order_products_prior = pd.read_csv('./archive/order_products__prior.csv')
order_products_train = pd.read_csv('./archive/order_products__train.csv')

print("Dataframes loaded successfully!\n")

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


train_data = final_train_df[final_train_df['user_id'].isin(train_user_ids)]
test_data = final_train_df[final_train_df['user_id'].isin(test_user_ids)]


train_dataset = InstacartDataset(train_data, user_map=global_user_map, product_map=global_product_map)

dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = FactorizationMachineModel(
    num_users = len(global_user_map),
    num_products = len(global_product_map),
    k = K_FACTORS
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Using device: {device}\n")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting pointwise LTR training ...\n")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in dataloader:
        users = batch['user'].to(device)
        products = batch['product'].to(device)
        labels = batch['label'].to(device)

        outputs = model(users, products)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss/len(dataloader)
    print(f"EPOCH {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
