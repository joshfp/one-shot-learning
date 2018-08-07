import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """
    Implementation of the model described in the Protypical Networks for Few-shot Learning paper.
    
    Attributes:
        emb_fn (torch.nn.Module): Function to used to extract the embeddings.
    """
    
    def __init__(self):
        super().__init__()
        self.emb_fn = EmbeddingFunction()
        
    def forward(self, support_X, support_y, query_X):
        """
        Args:
            support_X (FloatTensor): Support samples features.
            support_y (LongTensor): Support samples labels.
            query_X (FloatTensor): Query samples features.
        """
        classes_n = len(torch.unique(support_y))
        
        # step 1: compute embeddings
        support_embs = self.emb_fn(support_X).cpu()
        query_embs = self.emb_fn(query_X)
        
        # step 2: compute prototypes (support samples)
        prototypes = torch.Tensor(classes_n, support_embs.size(1))
        for c in range(classes_n):
            class_idxs = (support_y == c).nonzero().view(-1)
            prototypes[c] = torch.mean(support_embs[class_idxs], dim=0)
        
        # step 3: compute  distances & probs (query samples)
        dist = euclidean_dist(query_embs, prototypes.to(query_embs.device))
        return F.log_softmax(-dist, dim=1)   

    
class EmbeddingFunction(nn.Module):    
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            conv_block(1, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64))
        
    def forward(self, x):
        x = self.conv_blocks(x)
        return x.view(x.size(0), -1)

    
def conv_block(in_n, filters_n):
    return nn.Sequential(
        nn.Conv2d(in_n, filters_n, 3, padding=1),
        nn.BatchNorm2d(filters_n),
        nn.ReLU(),
        nn.MaxPool2d(2))

    
def euclidean_dist(t1, t2):
    """
    Compute the euclidian distance between two tensors.

    Args:
        t1 (tensor): Tensor 1 (N x D).
        t2 (tensor): Tensor 2 (M x D).
    
    Return:
        tensor: Euclidian distance between tensors (M x N).
    """
    assert t1.size(1) == t2.size(1), 'Dim 1 of both tensors must be the same.'
    n = t1.size(0)
    m = t2.size(0)
    d = t1.size(1)
    t1 = t1.unsqueeze(1).expand(-1, m, -1)
    t2 = t2.unsqueeze(0).expand(n, -1, -1)
    return ((t1 - t2) ** 2).sum(2)
    

