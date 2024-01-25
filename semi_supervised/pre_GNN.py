import torch
import torch.nn.functional as F

class pre_gnn(torch.nn.Module):
    def __init__(self, encoder, num_features, out_features, pre_dataset):
        super(pre_gnn, self).__init__()
        in_features = pre_dataset.num_features
        num_classes = pre_dataset.num_classes
        self.encoder = encoder
        self.mlp1 = torch.nn.Linear(in_features, num_features)
        self.mlp2 = torch.nn.Linear(out_features, num_classes)
    
    def forward(self, data):
        data.x = self.mlp1(data.x)
        y = self.encoder.forward_cl(data)
        y = self.mlp2(y)
        y = F.log_softmax(y, dim=-1)
        return y
    
    def get_embedding(self, data):
        x = self.encoder.forward_cl(data)
        return x