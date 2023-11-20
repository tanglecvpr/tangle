
#----> pytorch imports
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.blocks=nn.Sequential(self.build_block(in_dim=self.input_dim, out_dim=int(self.input_dim)),
                                            self.build_block(in_dim=int(self.input_dim), out_dim=int(self.input_dim)),
                                            nn.Linear(in_features=int(self.input_dim), out_features=self.output_dim),
                                )
        

    def build_block(self, in_dim, out_dim):
        return nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class CLSS(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CLSS, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(nn.Linear(self.input_dim, 512), 
                               nn.ReLU(), 
                               nn.Linear(512, output_dim))
    def forward(self, x):
        x = self.layers(x)
        return x
    

class ProjHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjHead, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=int(self.input_dim)),
                nn.LayerNorm(int(self.input_dim)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=int(self.input_dim) ,out_features=self.output_dim),
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
    




