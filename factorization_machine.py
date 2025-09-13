import torch 
import torch.nn as nn


class FactorizationMachineModel(nn.Module):
    def __init__(self, num_users, num_products, k):
        super(FactorizationMachineModel, self).__init__()
        self.user_embeds = nn.Embedding(num_users, k)
        self.products_embeds = nn.Embedding(num_products, k)
        self.user_bias = nn.Embedding(num_users, 1)
        self.products_bias = nn.Embedding(num_products, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, product):
        user_v = self.user_embeds(user)
        product_v = self.products_embeds(product)
        user_w = self.user_bias(user)
        product_w = self.products_bias(product)
        
        linear_term = self.bias + user_w + product_w
        interaction_term = (user_v * product_v).sum(1, keepdim=True)

        raw_prediction = linear_term + interaction_term
        return raw_prediction.squeeze()