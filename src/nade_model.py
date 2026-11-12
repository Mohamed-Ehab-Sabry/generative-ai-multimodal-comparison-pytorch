import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NADE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=500):
        super(NADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Explicit parameter matrices for autoregressive formulation
        # V: Input to hidden weights
        # W: Hidden to output weights
        self.V = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.W = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b = nn.Parameter(torch.empty(input_dim))
        self.c = nn.Parameter(torch.empty(hidden_dim))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        bound_b = 1 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.b, -bound_b, bound_b)
        bound_c = 1 / math.sqrt(self.input_dim)
        nn.init.uniform_(self.c, -bound_c, bound_c)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state a_0 = c for the entire batch
        a = self.c.unsqueeze(0).expand(batch_size, -1).clone()
        p_hat = []
        
        # Autoregressive generation sequentially over the 784 dimensions
        for d in range(self.input_dim):
            h_d = torch.sigmoid(a)
            
            # p(x_d = 1 | x_{<d})
            p_d = torch.sigmoid(F.linear(h_d, self.W[d:d+1, :], self.b[d:d+1]))
            p_hat.append(p_d)
            
            if d < self.input_dim - 1:
                # Update hidden state using the actual input x_d
                # a_d = a_{d-1} + V_d * x_d
                a = a + F.linear(x[:, d:d+1], self.V[d:d+1, :])

        return torch.cat(p_hat, dim=1)

    def exact_nll(self, x):
        """
        Computes the exact Negative Log-Likelihood (NLL).
        """
        p_hat = self.forward(x)
        
        # Exact NLL is calculated by computing the BCE for each dimension,
        # summing over the 784 dimensions per sample, then averaging across the batch.
        nll = F.binary_cross_entropy(p_hat, x, reduction='none').sum(dim=1).mean()
        return nll