#----> internal imports
from core.models.vision_transformer import VisionTransformer
from core.models.abmil import ABMILEmbedder
from core.models.helpers import MLP, ProjHead

#----> pytorch imports
import torch
from torch import nn


class MMSSL(nn.Module):
    def __init__(self, config, n_tokens_wsi, n_tokens_rna, patch_embedding_dim=768):
        super(MMSSL, self).__init__()
        self.config = config
        self.n_tokens_wsi = n_tokens_wsi
        self.n_tokens_rna = n_tokens_rna
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.patch_embedding_dim = patch_embedding_dim

        ########## WSI embedder #############
        if self.config["wsi_encoder"] == "vit-base":
            self.wsi_embedder = VisionTransformer(pos_embed_type=self.config["pos_encoding"],
                                                add_cls=False, 
                                                num_patches=self.n_tokens_wsi,
                                                qkv_bias=True, 
                                                return_all_tokens=True, 
                                                )
            
        elif self.config["wsi_encoder"] == "abmil":
            pre_params = {'input_dim': self.patch_embedding_dim,
                         'hidden_dim': 768
                            }
            attention_params = {'model': 'ABMIL',
                            'params': {
                            'input_dim': 768,
                            'hidden_dim': 512,
                            'dropout': True, 
                            'activation': self.config["activation"],
                            'n_classes': 1 
                                }
                                }
            self.wsi_embedder = ABMILEmbedder(pre_attention_params=pre_params,
                                              attention_params=attention_params,
                                              )


        ########## RNA embedder #############
        if self.config["rna_encoder"] == "linear":
            self.rna_embedder = nn.Linear(in_features=n_tokens_rna, out_features=self.config["embedding_dim"])
        else:
            self.rna_embedder = MLP(input_dim=n_tokens_rna, output_dim=self.config["embedding_dim"])
        
        
        ########## RNA Reconstruction module #############
        if self.config["intra_modality_rna"]:
            if self.config["rna_encoder"] == "linear":
                self.rna_reconstruction = nn.Linear(in_features=self.config["embedding_dim"], out_features=n_tokens_rna)
            else:
                self.rna_reconstruction = MLP(input_dim=self.config["embedding_dim"], output_dim=n_tokens_rna)
        else:
            self.rna_reconstruction = None
    
        ########## Projection Head #############
        if self.config["embedding_dim"] != 768:
            self.proj = ProjHead(input_dim=768, output_dim=self.config["embedding_dim"])
        else:
            self.proj = None
        

    def forward(self, wsi_emb, rna_emb, token_position_wsi=None):
        
        if self.config["wsi_encoder"] == "vit-base":
            wsi_emb = self.wsi_embedder(wsi_emb, token_pos=token_position_wsi)
            wsi_emb = wsi_emb.mean(dim=1)
        else:
            wsi_emb = self.wsi_embedder(wsi_emb)
        
        if self.proj:
            wsi_emb = self.proj(wsi_emb)

        if rna_emb is None:
            rna_emb = None
        else:
            rna_emb = self.rna_embedder(rna_emb)
        
        if self.config["intra_modality_rna"]:
            rna_reconstruction = self.rna_reconstruction(rna_emb)
        else:
            rna_reconstruction = None

        return wsi_emb, rna_emb, rna_reconstruction
    
    
    def get_features(self, wsi_emb, token_position_wsi=None):
        
        if self.config["wsi_encoder"] == "vit-base":
            wsi_emb = self.wsi_embedder(wsi_emb, token_pos=token_position_wsi)
            wsi_emb = wsi_emb.mean(dim=1)
        else:
            wsi_emb = self.wsi_embedder(wsi_emb)
            
        if self.proj:
            wsi_emb = self.proj(wsi_emb)

        return wsi_emb
        
