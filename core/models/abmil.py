#----> pytorch imports
import torch 
from torch import nn
import torch.nn.functional as F


class ABMILEmbedder(nn.Module):

    def __init__(
        self,
        pre_attention_params: dict = None,
        attention_params: dict = None,
        aggregation: str = 'regular',
    ) -> None:
        """
        """
        super(ABMILEmbedder, self).__init__()

        # 1- build pre-attention params 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre_attention_params = pre_attention_params
        if pre_attention_params is not None:
            self._build_pre_attention_params(params=pre_attention_params)

        # 2- build attention params
        self.attention_params = attention_params
        if attention_params is not None:
            self._build_attention_params(
                attn_model=attention_params['model'],
                params=attention_params['params']
            )

        # 3- set aggregation type 
        self.agg_type = aggregation  # Option are: mean, regular, additive, mean_additive

    def _build_pre_attention_params(self, params):
        """
        Build pre-attention params 
        """
        leight_pre_attn = False
        if 'leight_pre_attn' in params:
            if params['leight_pre_attn']:
                leight_pre_attn = True
        
        if leight_pre_attn:
            self.pre_attn = self._fc1 = nn.Sequential(nn.Linear(params['input_dim'], params['hidden_dim']), nn.GELU())
        else:
            self.pre_attn = nn.Sequential(
                nn.Linear(params['input_dim'], params['hidden_dim']),
                nn.LayerNorm(params['hidden_dim']),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(params['hidden_dim'], params['hidden_dim']),
                nn.LayerNorm(params['hidden_dim']),
                nn.GELU(),
                nn.Dropout(0.1),
            )

    def _build_attention_params(self, attn_model='ABMIL', params=None):
        """
        Build attention params 
        """
        if attn_model == 'ABMIL':
            self.attn = BatchedABMIL(**params)
        else:
            raise NotImplementedError('Attention model not implemented -- Options are ABMIL, PatchGCN and TransMIL.')
        

    def forward(
        self,
        bags: torch.Tensor,
        return_attention: bool = False, 
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            bags (torch.Tensor): batched representation of the tokens 
            return_attention (bool): if attention weights should be returned (raw attention)
        Returns:
            torch.tensor: Model output.
        """

        # pre-attention
        if self.pre_attention_params is not None:
            embeddings = self.pre_attn(bags)
        else:
            embeddings = bags

        # compute attention weights  
        if self.attention_params is not None:
            if return_attention:
                attention, raw_attention = self.attn(embeddings, return_raw_attention=True)
            else:
                attention = self.attn(embeddings)  # return post softmax attention

        if self.agg_type == 'regular':
            embeddings = embeddings * attention
            if self.attention_params["params"]["activation"] == "sigmoid":
                slide_embeddings = torch.mean(embeddings, dim=1)
            else:
                slide_embeddings = torch.sum(embeddings, dim=1)

        else:
            raise NotImplementedError('Agg type not supported. Options are "additive" or "regular".')

        
        return slide_embeddings


class BatchedABMIL(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=256, dropout=False, n_classes=1, activation='softmax'):
        """
        Attention Network with Sigmoid Gating (3 fc layers). Supports batching 
        args:
            input_dim (int): input feature dimension
            hidden_dim (int): hidden layer dimension
            dropout (bool): whether to use dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(BatchedABMIL, self).__init__()

        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention_a = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()]

        self.attention_b = [nn.Linear(input_dim, hidden_dim),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, return_raw_attention=False):
        """
        Forward pass 
        x List[(torch.Tensor)]: List of [patches x d] w/ len(x) = bs
        """

        # gated attention 
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes

        if self.activation == 'softmax':
            activated_A = F.softmax(A, dim=1)
        elif self.activation == 'leaky_relu':  # enable "counting" 
            activated_A = F.leaky_relu(A)
        elif self.activation == 'relu':
            activated_A = F.relu(A)
        elif self.activation == 'sigmoid':  # enable "counting"
            activated_A = torch.sigmoid(A)
        else:
            raise NotImplementedError('Activation not implemented.')

        if return_raw_attention:
            return activated_A, A

        return activated_A
