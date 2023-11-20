#----> pytorch imports
import torch 
from torch.utils.data import Dataset

#----> general imports 
from sklearn.preprocessing import minmax_scale, StandardScaler
import os
import pandas as pd
import h5py
import random
import numpy as np


# Gloabl modifications on testset labels and grouping
EXCLUDED_SLIDES_CURATED_TESTSET_DICT = {
                                            'Proliferation, bile duct': [
                                                '63777.h5'
                                                '63748.h5',
                                                '46844.h5',
                                                '46833.h5',
                                                '46835.h5',
                                                '46839.h5',
                                                '46423.h5',
                                            ],
                                            'Hypertrophy': [
                                                '10688.h5',
                                                '27624.h5',
                                                '63546.h5',
                                                '63616.h5',
                                                '63676.h5',
                                                '63687.h5',
                                                '63777.h5',
                                            ],
                                            'Increased mitosis': [
                                                '54900.h5',
                                                '53532.h5',
                                                '53808.h5',
                                                '55638.h5',
                                                '55633.h5',
                                                '55100.h5',
                                                '54443.h5',
                                                '54549.h5',
                                                '54568.h5',
                                                '54574.h5',
                                                '54677.h5',
                                                '54762.h5',
                                                '54805.h5',
                                                '54894.h5',
                                                '70809.h5'
                                            ]
}
MODIFICATIONS_CURATED_TESTSET = {
                                            'Fatty change': {  # keep!
                                                '19714.h5': 1,
                                                '46360.h5': 0,
                                                '46387.h5': 0,
                                                '53169.h5': 0,
                                                '53257.h5': 0,
                                                '53262.h5': 0,
                                                '58909.h5': 1
                                            },
                                            'Proliferation, bile duct': {  # Keep!
                                                '46841.h5': 0,
                                                '46835.h5': 0
                                            },
                                            'Hematopoiesis, extramedullary': { # Keep!
                                                '31305.h5': 0
                                            },
                                            'Change, eosinophilic': { # Keep!
                                                '19714.h5': 1,
                                                '30270.h5': 1,
                                                '30272.h5': 1,
                                                '30274.h5': 1,
                                                '30278.h5': 1,
                                                '49064.h5': 1,
                                                '49066.h5': 1,
                                                '55728.h5': 1,
                                                '57174.h5': 1,
                                                '57331.h5': 1,
                                                '63546.h5': 1,
                                                '63616.h5': 1,
                                                '63941.h5': 1,
                                                '63952.h5': 1,
                                                '63955.h5': 1,
                                                '63955.h5': 1,
                                                '70760.h5': 1,
                                                '19764.h5': 1,
                                                '19773.h5': 1,
                                                '19777.h5': 1,
                                                '19781.h5': 1,
                                                '19785.h5': 1,
                                                '46387.h5': 1,  # switch to 1
                                                '46423.h5': 1,  # switch to 1
                                                '49064.h5': 0, # ground glass
                                                '49066.h5': 0, # ground glass
                                                '57174.h5': 0, # 1 (more hypertrophic)
                                                '57331.h5': 0, # 1
                                                '70760.h5': 0,
                                            },
                                            'Increased mitosis': {
                                                '52496.h5': 1,
                                                '54300.h5': 1,
                                                '54187.h5': 1,
                                                '57174.h5': 1,
                                                '54187.h5': 1,
                                                '59123.h5': 1
                                            },
                                            'Hypertrophy': {
                                                '46387.h5': 1, # switch to 1
                                                '51184.h5': 1, # switch to 1
                                                '51330.h5': 1, # switch to 1
                                                '52101.h5': 1, # switch to 1
                                                '52476.h5': 1, # switch to 1
                                                '54931.h5': 1, # switch to 1
                                                '55078.h5': 1,
                                                '55179.h5': 1,
                                                '55541.h5': 1,
                                                '58295.h5': 1,
                                                '58436.h5': 1,
                                                '58541.h5': 1,
                                                '14211.h5': 0,  # set to 0
                                                '30270.h5': 0,  # set to 0
                                                '30272.h5': 0,  # set to 0
                                                '30274.h5': 0,  # set to 0
                                                '30278.h5': 0,  # set to 0
                                            },
                                            'Cellular infiltration': {
                                                '11746.h5': 0,
                                                '11961.h5': 0,
                                                '46387.h5': 0,
                                                '46756.h5': 0,
                                                '46829.h5': 1,
                                            }
                                        }
MODIFICATIONS_TESTSET = {
                                                    'Proliferation, bile duct': {
                                                        '63752.h5': 0,
                                                    },
                                                    'Hypertrophy': {
                                                        '49032.h5': 1,
                                                        '49072.h5': 1,
                                                        '58708.h5': 1,
                                                        '58711.h5': 1,
                                                        '58714.h5': 1,
                                                        '58726.h5': 1,
                                                        '58729.h5': 1,
                                                        '58732.h5': 1,
                                                        '58738.h5': 1,
                                                        '6270.h5': 1,
                                                        '30499.h5': 1,
                                                        '30503.h5': 1,
                                                        '46696.h5': 1,
                                                        '46705.h5': 1,
                                                        '46711.h5': 1,
                                                        '46714.h5': 1,
                                                        '46717.h5': 1,
                                                        '46723.h5': 1,
                                                        '46730.h5': 1,
                                                        '46733.h5': 1,
                                                        '48806.h5': 1,
                                                        '49032.h5': 1,
                                                        '49044.h5': 1,
                                                        '49058.h5': 1,
                                                        '49068.h5': 1,
                                                        '49072.h5': 1,
                                                        '6245.h5': 1,
                                                        '6258.h5': 1,
                                                        '6267.h5': 1,
                                                        '6270.h5': 1,
                                                        '6274.h5': 1,
                                                        '6388.h5': 1,
                                                        '6390.h5': 1,
                                                        '6391.h5': 1,
                                                        '6392.h5': 1,
                                                        '63926.h5': 1,
                                                        '63930.h5': 1,
                                                        '10690.h5': 0,
                                                        '54351.h5': 0,
                                                        '63763.h5': 0,
                                                        '63768.h5': 0,
                                                    },
                                                    'Necrosis': {
                                                        '58993.h5': 1, # ok
                                                        '10648.h5': 1, # ok + add infiltration
                                                        '10684.h5': 1, # ok
                                                        '13335.h5': 1, # ok
                                                        '13383.h5': 1, # ok
                                                        '14489.h5': 1, # ok
                                                        '2619.h5': 1, # ok
                                                        '2638.h5': 1, # ok
                                                        '30510.h5': 1, # ok
                                                        '30516.h5': 1, # ok
                                                        '31017.h5': 1, # ok
                                                        '31108.h5': 1, # ok
                                                        '31109.h5': 1, # ok
                                                        '31133.h5': 1, # ok
                                                        '31841.h5': 1, # ok
                                                        '32980.h5': 1, # ok
                                                        '33027.h5': 1, # ok
                                                        '46304.h5': 1, # ok
                                                        '48834.h5': 1, # ok
                                                        '48983.h5': 1, # ok
                                                        '52004.h5': 1, # ok
                                                        '52887.h5': 1, # ok
                                                        '58844.h5': 1, # ok
                                                        '58856.h5': 1, # ok
                                                        '58867.h5': 1, # ok
                                                        '58873.h5': 1, # ok
                                                        '58884.h5': 1, # ok
                                                        '58897.h5': 1, # ok
                                                        '60683.h5': 1, # ok
                                                        '6164.h5': 1, # ok
                                                        '6510.h5': 1, # ok
                                                        '10618.h5': 1, # ok
                                                        '11494.h5': 1, # ok
                                                        '11796.h5': 1, # ok
                                                        '13337.h5': 1, # ok
                                                        '13406.h5': 1, # ok
                                                        '14508.h5': 1, # ok
                                                        '19357.h5': 1, # ok
                                                        '19375.h5': 1, # ok
                                                        '19632.h5': 1, # ok
                                                        '2408.h5': 1, # ok
                                                        '2413.h5': 1, # ok
                                                        '2455.h5': 1, # ok
                                                        '27550.h5': 1, #  ok
                                                        '30242.h5': 1, # ok
                                                        '30263.h5': 1, # ok
                                                        '30426.h5': 1, # ok
                                                        '30428.h5': 1, # ok
                                                        '31004.h5': 1, # ok
                                                        '31075.h5': 1, #
                                                        '32925.h5': 1, #
                                                        '32978.h5': 1, #
                                                        '48414.h5': 1, #
                                                        '48529.h5': 1, #
                                                        '51455.h5': 1, #
                                                        '53220.h5': 1, #
                                                        '54828.h5': 1, #
                                                        '60342.h5': 1, #
                                                        '60369.h5': 1, #
                                                        '60577.h5': 1, #
                                                        '20483.h5': 0, #
                                                        '53775.h5': 0,
                                                        '58179.h5': 0,
                                                        '58216.h5': 0,
                                                        '59212.h5': 0,
                                                        '59230.h5': 0,
                                                        '63433.h5': 0,
                                                        '63438.h5': 0,
                                                        '63442.h5': 0,
                                                        '63477.h5': 0,
                                                        '63481.h5': 0,
                                                        '63485.h5': 0,
                                                        '63489.h5': 0,
                                                        '63497.h5': 0,
                                                        '63501.h5': 0,
                                                        '63505.h5': 0,
                                                        '63509.h5': 0,
                                                        '63513.h5': 0,
                                                        '63517.h5': 0,
                                                        '63522.h5': 0,
                                                        '63526.h5': 0,
                                                        '63553.h5': 0,
                                                        '63562.h5': 0,
                                                        '63566.h5': 0,
                                                        '63571.h5': 0,
                                                        '63575.h5': 0,
                                                        '63587.h5': 0,
                                                        '63598.h5': 0,
                                                    },
                                                    'Fatty change': {
                                                        '13441.h5': 1,
                                                        '14548.h5': 1,
                                                        '19714.h5': 1,
                                                        '19720.h5': 1,
                                                        '19736.h5': 1,
                                                        '19739.h5': 1,
                                                        '19742.h5': 1,
                                                        '19744.h5': 1,
                                                        '19747.h5': 1,
                                                        '19749.h5': 1,
                                                        '19751.h5': 1,
                                                        '19753.h5': 1,
                                                        '19755.h5': 1,
                                                        '19761.h5': 1,
                                                        '2428.h5': 1,
                                                        '2453.h5': 1,
                                                        '63903.h5': 1,
                                                        '63911.h5': 1,
                                                        '63937.h5': 1,
                                                        # @TODO: inspect these elements!
                                                        '46358.h5': 0, # from here
                                                        '46360.h5': 0,
                                                        '46362.h5': 0,
                                                        '46363.h5': 0,
                                                        '46365.h5': 0,
                                                        '46367.h5': 0,
                                                        '46368.h5': 0,
                                                        '46369.h5': 0,
                                                        '46371.h5': 0,
                                                        '46373.h5': 0,
                                                        '46376.h5': 0,
                                                        '46380.h5': 0,
                                                        '46387.h5': 0,
                                                        '46389.h5': 0,
                                                        '46391.h5': 0,
                                                        '46395.h5': 0,
                                                        '46403.h5': 0, # to here: same morphology, double check if fatty change or just glycogen
                                                        '53153.h5': 0,
                                                        '53158.h5': 0,
                                                        '53169.h5': 0,
                                                        '53174.h5': 0,
                                                        '53234.h5': 0,
                                                        '53251.h5': 0,
                                                        '53257.h5': 0,
                                                        '53262.h5': 0,
                                                        '53273.h5': 0,
                                                        '53278.h5': 0,
                                                    },
                                                    'Cellular infiltration': {
                                                        '46358.h5': 1,
                                                        '11404.h5': 1,
                                                        '11796.h5': 1,
                                                        '20066.h5': 1,
                                                        '31859.h5': 1,
                                                        '46062.h5': 1,
                                                        '46091.h5': 1,
                                                        '46094.h5': 1,
                                                        '46098.h5': 1,
                                                        '46292.h5': 1,
                                                        '46294.h5': 1,
                                                        '46300.h5': 1,
                                                        '46307.h5': 1,
                                                        '46308.h5': 1,
                                                        '46344.h5': 1,
                                                        '46382.h5': 1,
                                                        '11499.h5': 1,
                                                        '11724.h5': 1,
                                                        '20406.h5': 1,
                                                        '20439.h5': 1,
                                                        '27642.h5': 1,
                                                        '31822.h5': 1,
                                                        '31841.h5': 1,
                                                        '31909.h5': 1,
                                                        '46111.h5': 1,
                                                        '46380.h5': 1,
                                                        '46717.h5': 1,
                                                        '46720.h5': 1,
                                                        '46740.h5': 1,
                                                        '46758.h5': 1,
                                                        '46820.h5': 1,
                                                        '46823.h5': 1,
                                                        '46829.h5': 1,
                                                        '48811.h5': 1,
                                                        '6174.h5': 1,
                                                        '31314.h5': 0,
                                                        '31320.h5': 0,
                                                        '31335.h5': 0,
                                                        '31340.h5': 0,
                                                        '31365.h5': 0,
                                                        '31371.h5': 0,
                                                        '31385.h5': 0,
                                                        '31391.h5': 0,
                                                        '31397.h5': 0,
                                                        '31426.h5': 0,
                                                        '31433.h5': 0,
                                                        '31446.h5': 0,
                                                        '31509.h5': 0,
                                                        '31515.h5': 0,
                                                        '31521.h5': 0,
                                                        '31529.h5': 0,
                                                        '31537.h5': 0,
                                                        '31556.h5': 0,
                                                        '31562.h5': 0,
                                                        '31583.h5': 0,
                                                        '31626.h5': 0,
                                                        '31654.h5': 0,
                                                        '31673.h5': 0,
                                                        '31679.h5': 0,
                                                        '31686.h5': 0,
                                                        '31713.h5': 0,
                                                        '31731.h5': 0,
                                                        '63473.h5': 0,
                                                        '63481.h5': 0,
                                                        '63485.h5': 0,
                                                        '63494.h5': 0,
                                                        '63497.h5': 0,
                                                        '63501.h5': 0,
                                                        '63526.h5': 0,
                                                        '63550.h5': 0,
                                                        '63553.h5': 0,
                                                        '63566.h5': 0,
                                                        '63571.h5': 0,
                                                        '63598.h5': 0,
                                                        '63605.h5': 0,
                                                        '63748.h5': 0,
                                                    },
                                                    'Hematopoiesis, extramedullary': {
                                                        '31305.h5': 0,
                                                        '31806.h5': 0,
                                                        '2347.h5': 1,
                                                        '2396.h5': 1,
                                                    }
                                                }                                           


######################################################################################################### 
#                                                                                                       #
#                                              DATASET ORGANIZATION                                     #
#                                                                                                       #
######################################################################################################### 

class DatasetFactory:
    """
    dataset factory handles data (expression + slides) and data splitting 
    """

    def __init__(self,
        patch_feature_dir, 
        csv_path,
        rnaseq_path, 
        split_file_path, 
        print_info=True,
        n_tokens=-1,
        sampling_strategy=None,
        sampling_augmentation=False,
        prune_compunds_ssl=False,
        prune_compunds_downstream=False,
        normalization_mode=None,
        prune_genes_1k=False
        ):

        self.patch_feature_dir = patch_feature_dir
        self.dataset_csv = pd.read_csv(csv_path)
        self.rnaseq_path = rnaseq_path 
        self.split_path = split_file_path
        self.print_info = print_info
        self.n_tokens = n_tokens
        self.sampling_strategy=sampling_strategy
        self.sampling_augmentation=sampling_augmentation
        self.prune_compounds_ssl=prune_compunds_ssl
        self.prune_compounds_downstream=prune_compunds_downstream
        self.normalization_mode = normalization_mode
        self.prune_genes_1k = prune_genes_1k
        if self.rnaseq_path is not None:
            self._normalize_rnaseq()
        self._summarize()
        self.compounds_to_keep = ['lomustine', 'methyltestosterone', 'griseofulvin', 'monocrotaline', 'vitamin A', 'ibuprofen', 'fenofibrate', 'chlorpropamide', 'naproxen', 'clofibrate', 'naphthyl isothiocyanate', 'carbon tetrachloride', 'thioacetamide', 'ethionine', 'coumarin', 'WY-14643', 'gemfibrozil', 'bromobenzene', 'amiodarone', 'ethambutol', 'colchicine', 'dantrolene', 'ethionamide', 'methapyrilene', 'disulfiram', 'aspirin', 'promethazine', 'chlormadinone', 'phenylanthranilic acid', 'benzbromarone', 'nitrosodiethylamine', 'cycloheximide', 'tunicamycin', 'phalloidin', 'galactosamine', 'phorone', 'hexachlorobenzene', 'dexamethasone', 'methylene dianiline', 'aflatoxin B1', 'acetamide', 'bortezomib', 'gefitinib', 'perhexiline', 'methimazole', 'chlormezanone', 'ethinylestradiol', 'imipramine', 'iproniazid', 'amitriptyline', 'hydroxyzine', 'diltiazem', 'nicotinic acid', 'phenobarbital', 'phenylbutazone', 'acetaminophen', 'carbamazepine', 'phenytoin', 'ketoconazole', 'papaverine', 'clomipramine', 'trimethadione', 'terbinafine', 'bendazac', 'benziodarone', 'nimesulide', 'phenacetin', 'danazol', 'acetamidofluorene', 'bromoethylamine', 'ticlopidine', 'diethyl maleate', 'diazepam', '3-methylcholanthrene']
        self.genes_1k = ['Aatf', 'Abcb1a', 'Abcc3', 'Abce1', 'Abhd16a', 'Abhd4', 'Abitram', 'Ablim3', 'Abt1', 'Acaa2', 'Acad11', 'Acbd6', 'Aco1', 'Aco2', 'Acot1', 'Acot2', 'Acot3', 'Acot4', 'Acot8', 'Acot9', 'Acsl4', 'Acsm2', 'Acyp1', 'Acyp2', 'Adgrg2', 'Adpgk', 'Adprs', 'Adrm1', 'Adsl', 'Aen', 'Afg3l2', 'Agfg1', 'Ahsa1', 'Aimp1', 'Aimp2', 'Akr1a1', 'Akr1b8', 'Akr7a3', 'Alas1', 'Aldh1a1', 'Aldh1a7', 'Aldoa', 'Alg12', 'Alkbh3', 'Amd1', 'Anapc16', 'Anp32b', 'Anxa7', 'Apip', 'Apoo', 'App', 'Aprt', 'Aqp7', 'Arfrp1', 'Arl4a', 'Arpp19', 'Asl', 'Asns', 'Atad3a', 'Atf4', 'Atg16l1', 'Atp1b1', 'Atp5f1d', 'Atp5if1', 'Atp6v0e1', 'Atp6v1b2', 'Atp6v1d', 'Atp6v1e1', 'Atp6v1f', 'Atp6v1g1', 'Atr', 'Bag5', 'Baiap2l1', 'Banf1', 'Bbs2', 'Bcs1l', 'Bfar', 'Birc3', 'Bloc1s2', 'Bloc1s4', 'Bnip3', 'Bnip3l', 'Bola2', 'Bola3', 'Bop1', 'Bpnt1', 'Brix1', 'Btaf1', 'Btg2', 'Btg3', 'Bud23', 'Bysl', 'C1qbp', 'C8h11orf52', 'Cacul1', 'Cacybp', 'Caprin1', 'Car2', 'Casp4', 'Cavin3', 'Cc2d1b', 'Ccdc120', 'Ccdc86', 'Ccm2', 'Ccng1', 'Cct2', 'Cct3', 'Cct4', 'Cct5', 'Cct6a', 'Cct7', 'Cct8', 'Cd276', 'Cd36', 'Cdc42se1', 'Cdca7', 'Cdkn1a', 'Cdkn2aip', 'Cdr2', 'Cdv3', 'Cebpb', 'Cebpz', 'Cemip', 'Cfap20', 'Cgref1', 'Chchd10', 'Chchd6', 'Chd1l', 'Chmp4c', 'Chordc1', 'Chpf', 'Chrna2', 'Chuk', 'Ciao2b', 'Ciapin1', 'Cidea', 'Cipc', 'Clec10a', 'Clpp', 'Cmbl', 'Colq', 'Comtd1', 'Cops5', 'Coq10b', 'Cox14', 'Cox18', 'Cox5a', 'Cpt1b', 'Cpt2', 'Crat', 'Creb3', 'Creb3l3', 'Creg1', 'Creld1', 'Cryl1', 'Csrp2', 'Cstf3', 'Ctf1', 'Ctps1', 'Ctr9', 'Ctsd', 'Ctsl', 'Cyb5a', 'Cyc1', 'Cyp1a1', 'Cyp20a1', 'Cyp2c6v1', 'Cyp2j4', 'Cyp3a23-3a1', 'Cyp4a1', 'Cyp4a3', 'Cyria', 'Cystm1', 'Dact2', 'Dancr', 'Dcaf13', 'Ddit3', 'Ddit4', 'Ddx1', 'Ddx18', 'Ddx21', 'Ddx28', 'Ddx39a', 'Ddx39b', 'Ddx49', 'Ddx51', 'Ddx56', 'Decr1', 'Dedd2', 'Degs1', 'Dgat1', 'Dhdds', 'Dhrs7b', 'Dhx30', 'Diablo', 'Dimt1', 'Dnaja2', 'Dnajb1', 'Dnajb9', 'Dnajc2', 'Dnlz', 'Dnm1l', 'Dph3', 'Dph5', 'Drg1', 'Dtd1', 'Dus1l', 'Dus2', 'Dusp22', 'Dynlt1', 'E2f5', 'Ears2', 'Ech1', 'Eci2', 'Edc3', 'Eef1a1', 'Eef1b2', 'Eef1d', 'Eef1e1', 'Eef1g', 'Ehhadh', 'Eif1', 'Eif1a', 'Eif1ad', 'Eif1ax', 'Eif2a', 'Eif2b1', 'Eif2b2', 'Eif2b3', 'Eif2s1', 'Eif2s2', 'Eif3b', 'Eif3c', 'Eif3d', 'Eif3f', 'Eif3g', 'Eif3i', 'Eif3j', 'Eif3m', 'Eif4a3', 'Eif4b', 'Eif4ebp1', 'Eif4h', 'Eif5a', 'Eif6', 'Elac2', 'Ell2', 'Elp6', 'Emc6', 'Enc1', 'Eola2', 'Epb41l5', 'Epcam', 'Ephx1', 'Erh', 'Esrra', 'Etf1', 'Exosc1', 'Exosc7', 'Exosc9', 'Faap20', 'Fam118a', 'Fam136a', 'Fam174c', 'Fam216a', 'Fam220a', 'Fam98a', 'Fastkd2', 'Fat1', 'Fbl', 'Fbxo27', 'Fbxo30', 'Fgf21', 'Fip1l1', 'Fkbp11', 'Fkbp5', 'Flad1', 'Fmo5', 'Fnta', 'Ftsj3', 'Fv1', 'G6pd', 'Gabarapl1', 'Gadd45a', 'Gadd45b', 'Gale', 'Gar1', 'Gars1', 'Gart', 'Gas5', 'Gclc', 'Gda', 'Gde1', 'Gdf15', 'Ggt1', 'Ghitm', 'Gjb2', 'Glrx2', 'Glrx3', 'Glyatl2', 'Gnai1', 'Gnl3', 'Gosr1', 'Got1', 'Gpatch4', 'Gpd1l', 'Gpnmb', 'Gpx2', 'Grin2c', 'Grpel1', 'Grwd1', 'Gsr', 'Gss', 'Gsta1', 'Gsta3', 'Gsta4', 'Gstm1', 'Gstm3', 'Gstm4', 'Gstp1', 'Gstt3', 'Gtf2b', 'Gtf2h1', 'Gtf3a', 'Gtf3c3', 'Gtpbp4', 'Gucy2c', 'Guk1', 'Hadhb', 'Hars1', 'Hdc', 'Hebp2', 'Herpud1', 'Hes1', 'Hgh1', 'Hgs', 'Hibch', 'Hikeshi', 'Hmgcl', 'Hmgcr', 'Hmox1', 'Hnrnpa1', 'Hnrnpf', 'Hprt1', 'Hscb', 'Hsdl2', 'Hsf1', 'Hsp90ab1', 'Hspa1a', 'Hspb1', 'Hspb8', 'Hspd1', 'Hspe1', 'Hsph1', 'Htatip2', 'Hyal2', 'Ica1', 'Id1', 'Ifrd1', 'Ifrd2', 'Igfbp1', 'Igfbp2', 'Imp3', 'Imp4', 'Impdh2', 'Inhbb', 'Ints2', 'Ipo4', 'Ipo7', 'Irak1', 'Irak3', 'Irs2', 'Isyna1', 'Jagn1', 'Jmjd6', 'Josd1', 'Josd2', 'Jun', 'Junb', 'Kat7', 'Kcmf1', 'Kdm8', 'Kif21a', 'Klf10', 'Klf11', 'Klf6', 'Krr1', 'Krt18', 'Krt8', 'Kti12', 'Kxd1', 'LOC100909675', 'LOC100911177', 'LOC100911266', 'LOC100911946', 'LOC100912002', 'LOC100912041', 'Lamtor3', 'Lbp', 'Lcn2', 'Lgals3', 'Limk2', 'Llph', 'Lpl', 'Lrrc47', 'Lrrfip1', 'Lsm3', 'Lsm8', 'Ltv1', 'Luc7l', 'Lurap1l', 'Lyrm2', 'Lysmd2', 'MGC116121', 'Maff', 'Mafg', 'Mafk', 'Magee1', 'Magoh', 'Mak16', 'Malsu1', 'Manba', 'Map3k12', 'Mbd2', 'Mcl1', 'Mcm7', 'Mcrip2', 'Mcts2', 'Mdm2', 'Me1', 'Med6', 'Med7', 'Mettl1', 'Mettl13', 'Mettl16', 'Mgme1', 'Mgmt', 'Mgst2', 'Minpp1', 'Miox', 'Mlec', 'Mnd1', 'Mob4', 'Morc4', 'Mospd2', 'Mpc2', 'Mphosph10', 'Mphosph6', 'Mrm3', 'Mrpl22', 'Mrpl38', 'Mrpl43', 'Mrpl47', 'Mrpl49', 'Mrpl50', 'Mrpl53', 'Mrpl55', 'Mrpl58', 'Mrps18a', 'Mrps18b', 'Mrps2', 'Mrps30', 'Mrps31', 'Mrps34', 'Mrps35', 'Mrps36', 'Mrps6', 'Mrps7', 'Mrto4', 'Mt2A', 'Mtap', 'Mterf3', 'Mtg1', 'Mtpap', 'Mybbp1a', 'Myc', 'Myl12a', 'Myo5b', 'N4bp2l2', 'Naa20', 'Naca', 'Nagk', 'Nap1l1', 'Nars1', 'Nat10', 'Nbn', 'Ncaph2', 'Ncdn', 'Ncl', 'Ndc1', 'Ndufaf3', 'Ndufaf4', 'Nedd8', 'Nek6', 'Neu2', 'Neurl3', 'Nfe2l1', 'Nfkbib', 'Nfu1', 'Ngrn', 'Nhej1', 'Nif3l1', 'Nifk', 'Nip7', 'Nipal2', 'Nkrf', 'Nle1', 'Nmd3', 'Nme1', 'Nme2', 'Nmt1', 'Noa1', 'Nob1', 'Noc2l', 'Nol12', 'Nol3', 'Nol9', 'Nolc1', 'Nop14', 'Nop16', 'Nop2', 'Nop56', 'Nop9', 'Nopchap1', 'Npdc1', 'Npepl1', 'Nploc4', 'Nqo1', 'Nr1d1', 'Nr2c2ap', 'Nsa2', 'Nsfl1c', 'Nsmce1', 'Nsun2', 'Nucks1', 'Nudt7', 'Nup153', 'Nup205', 'Nup35', 'Nup43', 'Nup85', 'Nup93', 'Nupr1', 'Nxt1', 'Obp1f', 'Odc1', 'Odf2l', 'Ogdh', 'Ola1', 'Orc3', 'Orc4', 'Orm1', 'Oser1', 'Osgin2', 'Otud1', 'Otud4', 'Oxnad1', 'Pabpc4', 'Pak1ip1', 'Pam16', 'Parl', 'Parp1', 'Pbdc1', 'Pcbp1', 'Pcgf6', 'Pcna', 'Pcsk6', 'Pdcd10', 'Pdcl3', 'Pdhx', 'Pdk4', 'Pdp1', 'Pdrg1', 'Pef1', 'Pelo', 'Pes1', 'Pex11a', 'Pex11g', 'Pfdn4', 'Pgam1', 'Pgam5', 'Pgs1', 'Phf5a', 'Phgdh', 'Phgr1', 'Pigl', 'Pim3', 'Pinx1', 'Pip4p2', 'Pir', 'Pisd', 'Pitrm1', 'Pla2g12a', 'Plin5', 'Plpp2', 'Pmm1', 'Pno1', 'Pold2', 'Polr1a', 'Polr1b', 'Polr1c', 'Polr1d', 'Polr1e', 'Polr1f', 'Polr2c', 'Polr2e', 'Polr2m', 'Polr3a', 'Polr3d', 'Polr3e', 'Polr3g', 'Pomgnt1', 'Pomp', 'Por', 'Ppan', 'Ppcs', 'Ppie', 'Ppil3', 'Ppl', 'Ppm1g', 'Ppme1', 'Ppp1r15a', 'Ppp2cb', 'Ppp2r1b', 'Ppp2r2d', 'Pprc1', 'Pptc7', 'Prdx6', 'Preb', 'Prelid3b', 'Prmt3', 'Prmt5', 'Prmt7', 'Prps1', 'Psat1', 'Psma1', 'Psma2', 'Psma3', 'Psma4', 'Psma5', 'Psma6', 'Psma7', 'Psmb1', 'Psmb2', 'Psmb3', 'Psmb4', 'Psmb7', 'Psmc1', 'Psmc2', 'Psmc3', 'Psmc5', 'Psmd1', 'Psmd12', 'Psmd13', 'Psmd14', 'Psmd2', 'Psmd3', 'Psmd4', 'Psmd5', 'Psmd6', 'Psmd7', 'Psmd8', 'Psme2', 'Psmg2', 'Psmg3', 'Psmg4', 'Ptbp1', 'Pter', 'Ptges2', 'Ptges3', 'Ptpmt1', 'Ptpn21', 'Ptrh2', 'Pum3', 'Pus1', 'Pus3', 'Pvr', 'Pwp2', 'Pxmp4', 'Qpct', 'Qtrt1', 'RGD1310553', 'RGD1359127', 'RGD1559459', 'RGD1561149', 'Rab21', 'Rab24', 'Rab30', 'Rab5if', 'Rabggtb', 'Rad17', 'Ran', 'Rangrf', 'Rapgef4', 'Rapgef5', 'Rars1', 'Rbbp8', 'Rbm3', 'Rce1', 'Reep6', 'Rell1', 'Retreg1', 'Rexo2', 'Rgs5', 'Rhbdd2', 'Rhob', 'Rhoc', 'Riok2', 'Riok3', 'Riox1', 'Rnf25', 'Rnf4', 'Rnh1', 'Rnps1', 'Rpa2', 'Rpf2', 'Rpl11', 'Rpl12', 'Rpl13a', 'Rpl18', 'Rpl18a', 'Rpl19', 'Rpl22', 'Rpl23', 'Rpl23a', 'Rpl24', 'Rpl26', 'Rpl27', 'Rpl27a', 'Rpl34', 'Rpl35', 'Rpl4', 'Rpl6', 'Rpl7l1', 'Rplp0', 'Rpp38', 'Rpp40', 'Rps11', 'Rps15', 'Rps15a', 'Rps16', 'Rps23', 'Rps27l', 'Rps3', 'Rps3a', 'Rps5', 'Rps8', 'Rrp1', 'Rrp12', 'Rrp15', 'Rrp8', 'Rrp9', 'Rrs1', 'Rtn4', 'Rusc1', 'Ruvbl2', 'S100a10', 'Samm50', 'Sar1b', 'Sars1', 'Scarb2', 'Sdad1', 'Sdc4', 'Sdhaf4', 'Selenos', 'Sestd1', 'Setd4', 'Setsip', 'Sf3b4', 'Sf3b6', 'Sgcb', 'Sgms1', 'Shq1', 'Sik2', 'Simc1', 'Skic8', 'Skp1', 'Slain2', 'Slc16a6', 'Slc20a1', 'Slc22a5', 'Slc25a20', 'Slc25a28', 'Slc25a29', 'Slc25a3', 'Slc25a33', 'Slc25a42', 'Slc35e3', 'Slc38a2', 'Slc3a2', 'Slc4a4', 'Slc5a6', 'Slc66a3', 'Slc7a6os', 'Smim26', 'Smim7', 'Smn1', 'Snrpa', 'Snrpa1', 'Snrpb', 'Snrpd3', 'Snrpf', 'Snx10', 'Spata5', 'Spcs1', 'Spryd7', 'Spsb4', 'Sptan1', 'Srd5a3', 'Srfbp1', 'Sri', 'Srp19', 'Srp68', 'Srsf4', 'Srxn1', 'Ssr2', 'Stat3', 'Stbd1', 'Stip1', 'Stk17b', 'Stk24', 'Stk40', 'Suclg1', 'Sugt1', 'Sult1b1', 'Sult2a6', 'Sun2', 'Supt6h', 'Supv3l1', 'Susd6', 'Sys1', 'Taf4b', 'Taf9', 'Tars3', 'Tatdn1', 'Tatdn2', 'Tax1bp1', 'Tbc1d15', 'Tbc1d2', 'Tbl3', 'Tceal9', 'Tcp1', 'Tefm', 'Tex10', 'Tex49', 'Tfap4', 'Tfb1m', 'Tfb2m', 'Tfpt', 'Tfrc', 'Tgif1', 'Tgm1', 'Them4', 'Thg1l', 'Thnsl1', 'Thop1', 'Thumpd3', 'Thyn1', 'Timm10b', 'Timm13', 'Timm17a', 'Timm21', 'Timm8a1', 'Timm8b', 'Timm9', 'Timp1', 'Tinf2', 'Tm2d3', 'Tma16', 'Tmem109', 'Tmem11', 'Tmem120a', 'Tmem147', 'Tmem14c', 'Tmem167a', 'Tmem184c', 'Tmem199', 'Tmem30b', 'Tmem41a', 'Tmem62', 'Tnfaip1', 'Tnfrsf12a', 'Tnk2', 'Tomm20', 'Tomm22', 'Tomm40', 'Tomm5', 'Tomm6', 'Tomm70', 'Tp53inp1', 'Tpd52', 'Traf4', 'Trappc4', 'Trib3', 'Trim24', 'Trim27', 'Trim37', 'Trit1', 'Trmt1', 'Trmt10a', 'Trmt10c', 'Trmt112', 'Trmt6', 'Trmt61a', 'Trp53rkb', 'Trub1', 'Tsc22d2', 'Tsen2', 'Tsfm', 'Tsku', 'Tspo', 'Tsr1', 'Ttc1', 'Ttpal', 'Tubb4b', 'Tusc3', 'Tut1', 'Txn1', 'Txnrd1', 'Tyms', 'U2af1', 'Uba2', 'Uba3', 'Uba52', 'Ubac1', 'Ube2f', 'Ube2g2', 'Ube2j1', 'Ube2v2', 'Ublcp1', 'Ubxn2a', 'Uchl3', 'Ufd1', 'Ugdh', 'Uggt1', 'Ugt2a1', 'Ugt2b17', 'Ugt2b7', 'Umps', 'Unkl', 'Uqcc4', 'Uqcrc2', 'Urb1', 'Usp10', 'Usp14', 'Usp15', 'Usp36', 'Uspl1', 'Utp14a', 'Utp15', 'Utp20', 'Utp25', 'Utp3', 'Utp4', 'Vdac2', 'Vdac3', 'Vnn1', 'Vps37a', 'Vwa8', 'Wasl', 'Wdr12', 'Wdr3', 'Wdr36', 'Wdr43', 'Wdr45', 'Wdr46', 'Wdr74', 'Wdr75', 'Wfdc21', 'Wwc3', 'Xpnpep3', 'Xpot', 'Yars1', 'Yars2', 'Ybx3', 'Yrdc', 'Ythdf2', 'Ywhag', 'Ywhah', 'Ywhaq', 'Zc3h15', 'Zc3h18', 'Zcchc10', 'Zdhhc13', 'Zdhhc2', 'Zfand2a', 'Zfand5', 'Zfp330', 'Zfp46', 'Zfp622', 'Zfp639', 'Zfp655', 'Zfp703', 'Znhit2', 'Znhit3', 'Znrd2', 'Zswim7', 'Zwint', 'mrpl11', 'mrpl24']
        
        
    def _normalize_rnaseq(self):
        rnaseq = pd.read_csv(self.rnaseq_path)
        data = rnaseq.values[:, 1:]   
        if self.normalization_mode == 'minmax':
            transformed = minmax_scale(data, feature_range=(-1, 1), axis=1)
        elif self.normalization_mode == 'std':
            scaler = StandardScaler()
            transformed = scaler.fit_transform(data)
        elif self.normalization_mode == None:
            transformed = data
        rnaseq.iloc[:, 1:] = transformed
        self.rnaseq = rnaseq

    def _summarize(self):
        if self.print_info:
            print('\nInit dataset factory...')

    def return_splits(self):
        all_splits = pd.read_csv(self.split_path) 
        train_split = self._get_split_from_df(all_splits=all_splits, split_key='train')
        val_split = self._get_split_from_df(all_splits=all_splits, split_key='val')
        test_split = self._get_split_from_df(all_splits=all_splits, split_key='test')
        curated_test_split = self._get_split_from_df(all_splits=all_splits, split_key='curated_test')

        return train_split, val_split, test_split, curated_test_split
    
    def return_test_split(self):
        all_splits = pd.read_csv(self.split_path) 
        test_split = self._get_split_from_df(all_splits=all_splits, split_key='test')
        return test_split

    def return_train_split(self):
        all_splits = pd.read_csv(self.split_path) 
        train_split = self._get_split_from_df(all_splits=all_splits, split_key='train')
        return train_split
    
    def return_val_split(self):
        all_splits = pd.read_csv(self.split_path) 
        val_split = self._get_split_from_df(all_splits=all_splits, split_key='val')
        return val_split
    
    def return_cur_test_split(self):
        all_splits = pd.read_csv(self.split_path)
        curated_test_split = self._get_split_from_df(all_splits=all_splits, split_key='curated_test')
        return curated_test_split

    def return_all(self):
        dataset = MMDataset(
                data_dir=self.patch_feature_dir,
                dataset_csv=self.dataset_csv,
                n_tokens=-1,
                sampling_strategy=self.sampling_strategy,
                sampling_augmentation=self.sampling_augmentation  
            )
        return dataset

    def _get_split_from_df(self, all_splits: dict={}, split_key: str='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        split = list(split.values)
        
        # get all the labels whose case ID is in split 
        dataset_csv = self.dataset_csv[self.dataset_csv['IMAGE_NAME'].isin(split)]
        
        # prunce desired compounds if specified 
        if self.prune_compounds_ssl and split_key == "train":
            dataset_csv = dataset_csv[dataset_csv["COMPOUND_NAME"].isin(self.compounds_to_keep)]
        if self.prune_compounds_downstream and split_key in ["val", "test"]:
            dataset_csv = dataset_csv[dataset_csv["COMPOUND_NAME"].isin(self.compounds_to_keep)]
        barcodes = dataset_csv['BARCODE'].values.tolist()
        barcodes = list(map(str, barcodes))
        
        # get rna sequence data
        if split_key == 'train' and self.rnaseq_path is not None:
            rnaseq = self.rnaseq[['Gene'] + barcodes]
            if self.prune_genes_1k:
                rnaseq = rnaseq[rnaseq['Gene'].isin(self.genes_1k)]
                if rnaseq.shape[0] != 1000:
                    raise ValueError('Subselection of 1k Genes was not possible.')
        else:
            rnaseq = None 

        if len(split) > 0:
            split_dataset = MMDataset(
                data_dir=self.patch_feature_dir,
                dataset_csv=dataset_csv,
                rnaseq=rnaseq,
                n_tokens=self.n_tokens if split_key == "train" else -1,
                sampling_strategy=self.sampling_strategy,
                sampling_augmentation=self.sampling_augmentation 
            )
        else:
            split_dataset = None
        
        return split_dataset


######################################################################################################### 
#                                                                                                       #
#                              DATASET FOR MULTI-MODAL PRE-TRAINING                                     #
#                                                                                                       #
######################################################################################################### 

class MMDataset(Dataset):
    
    """
    Dataset for multi-modal SSL pre-training
    """

    def __init__(self,
        data_dir, # dir with pre-extracted patch embeddings 
        dataset_csv,  # all meta information about the samples (lesions, grades, )
        rnaseq=None, # rnaseq values as df
        n_tokens=-1,   # number of patches to keep when subsampling. If -1, no subsampling.
        sampling_strategy = None,
        sampling_augmentation = False,
        ): 
        super(Dataset, self).__init__()
        #---> self
        self.data_dir = data_dir
        self.dataset_csv = dataset_csv.copy()
        self.rnaseq = rnaseq
        self.n_tokens = n_tokens
        self.ids = dataset_csv["IMAGE_NAME"].tolist()
        self.sampling_strategy = sampling_strategy
        self.sampling_augmentation = sampling_augmentation
        #----> extract class names  
        if self.rnaseq is not None:
            self.gene_names = rnaseq['Gene'].values
            
        self.excluded_slides_curated_testset_dict = EXCLUDED_SLIDES_CURATED_TESTSET_DICT
        self.excluded_slides_curated_testset_list = [item for sublist in self.excluded_slides_curated_testset_dict.values() for item in sublist]
        self.modifications_curated_testset = MODIFICATIONS_CURATED_TESTSET
        self.modifications_testset = MODIFICATIONS_TESTSET
        
        do_testset_label_modifications = True
        if do_testset_label_modifications:
            self.fuse_liver_classes = True
            self._modify_test_dataset()
            self.dataset_csv = self.dataset_csv.reset_index()
            
        
    def __len__(self):
        return len(self.dataset_csv)
    
    def __getitem__(self, idx):

        # 1. load patch embeddings 
        slide_info = self.dataset_csv.iloc[idx]
        patch_emb, patch_positions, cluster_idx = self._load_wsi_embs_from_path(slide_id=slide_info['IMAGE_NAME'])
        avg_patch_emb = patch_emb.mean(dim=0)
        
        # Token sampling
        if self.n_tokens > -1:
            # Random sampling
            if self.sampling_strategy == "random": 
                patch_indices = [torch.randint(0, patch_emb.size(0), (self.n_tokens,)).tolist() if  patch_emb.shape[0] < self.n_tokens else torch.randperm(patch_emb.size(0))[:self.n_tokens].tolist()]   
                if self.sampling_augmentation:
                    patch_indices.append(torch.randint(0, patch_emb.size(0), (self.n_tokens,)).tolist() if patch_emb.shape[0] < self.n_tokens else torch.randperm(patch_emb.size(0))[:self.n_tokens].tolist())           
            # cluster sampling
            elif self.sampling_strategy == "kmeans_cluster":
                patch_indices = self._sample_cluster_indices(cluster_idx=cluster_idx)
                if self.sampling_augmentation:
                    patch_indices.append(self._sample_cluster_indices(cluster_idx=cluster_idx)) 
            else:
                raise ValueError("Invalid sampling strategy.")
            if len(patch_indices) > 1:
                patch_emb_ = patch_emb[patch_indices[0]]
                patch_positions_ = patch_positions[patch_indices[0]]
                patch_emb_aug = patch_emb[patch_indices[1]]
                patch_positions_aug = patch_positions[patch_indices[1]]
            else:
                patch_emb_ = patch_emb[patch_indices]
                patch_positions_ = patch_positions[patch_indices]
                patch_emb_aug = ['nan']
                patch_positions_aug = ['nan']
        else:
            patch_emb_ = patch_emb
            patch_positions_ = patch_positions
            patch_emb_aug = ['nan']
            patch_positions_aug = ['nan']
        
        # 2. load RNASeq values
        if self.rnaseq is not None:
            rnaseq = torch.from_numpy(self.rnaseq[str(slide_info['BARCODE'])].values).to(torch.float32) 
            if len(rnaseq.shape) > 1:
                rnaseq = rnaseq[:, 0].squeeze()
            return patch_emb_, rnaseq, patch_positions_, slide_info['IMAGE_NAME'], patch_emb_aug, patch_positions_aug, avg_patch_emb
        else: 
            return patch_emb_, ['nan'], patch_positions_, slide_info['IMAGE_NAME'], patch_emb_aug, patch_positions_aug, avg_patch_emb

    def _sample_cluster_indices(self, cluster_idx):
        n = 2 if self.sampling_augmentation else 1
        patch_indices = []
        for i in range(n):
            pi = []
            if cluster_idx is not None:
                    # select random indices for each cluster
                    n = int(self.n_tokens/len(set(cluster_idx.tolist())))
                    for cluster in set(cluster_idx.tolist()):
                        indices = [i for i, x in enumerate(cluster_idx) if x == cluster]
                        selected_indices = random.sample(indices, min(n, len(indices)))
                        pi += selected_indices
                    # Fill up patch_indices with further indices until number of tokens is satisfied
                    all_indices = set(range(len(cluster_idx)))
                    all_indices -= set(pi)  # Remove already selected indices
                    pi += random.sample(all_indices, self.n_tokens - len(pi))
            else:
                raise ValueError("No cluster indices found in file.")
            patch_indices.append(pi)
        return patch_indices
    
        
    def _load_wsi_embs_from_path(self, slide_id):
        if not '.h5' in slide_id:
            slide_id += '.h5'
        feats_path = os.path.join(self.data_dir, slide_id)
        with h5py.File(feats_path, 'r') as hdf5_file:
            patch_features = torch.from_numpy(hdf5_file['features'][:]).squeeze().to(torch.float32)
            patch_positions = torch.from_numpy(hdf5_file['coords'][:].squeeze()).to(torch.float32)
            # normalize patch positions
            patch_positions -= patch_positions.min(0)[0]
            patch_positions /= patch_positions.max(0)[0]
            cluster_idx = torch.from_numpy(np.array(hdf5_file['cluster_idx'])).to(torch.float32) if "cluster_idx" in hdf5_file.keys() else None
        return patch_features, patch_positions, cluster_idx
    
    def _modify_test_dataset(self):
        
        if self.fuse_liver_classes:
            # Fuse Hypertrophy = 1 if Eosinophilic Change, Ground Glass appearance
            self.dataset_csv["Hypertrophy"] = self.dataset_csv["Hypertrophy"] + self.dataset_csv["Change, eosinophilic"] + self.dataset_csv["Ground glass appearance"] + self.dataset_csv["Degeneration, hydropic"]
            self.dataset_csv.loc[self.dataset_csv['Hypertrophy'] > 1, 'Hypertrophy'] = 1
            # Fuse Necrosis and Single Cell Necrosis
            self.dataset_csv["Necrosis"] = self.dataset_csv["Necrosis"] + self.dataset_csv["Single cell necrosis"] 
            self.dataset_csv.loc[self.dataset_csv['Necrosis'] > 1, 'Necrosis'] = 1
            # Fuse Profileration bile duct and oval cell
            self.dataset_csv["Proliferation"] = self.dataset_csv["Proliferation, oval cell"] + self.dataset_csv["Proliferation, bile duct"] 
            self.dataset_csv.loc[self.dataset_csv['Proliferation'] > 1, 'Proliferation'] = 1
        
        # remove specified slides from curated testset split
        self.ids = list(set(self.ids) - set(self.excluded_slides_curated_testset_list))
        
        # modify testset labels
        for lesion, wsi_labels in self.modifications_testset.items():
            for id, label in wsi_labels.items():
                self.dataset_csv.loc[self.dataset_csv['IMAGE_NAME'] == id, lesion] = label
            
        # modify curated testset labels
        for lesion, wsi_labels in self.modifications_curated_testset.items():
            for id, label in wsi_labels.items():
                self.dataset_csv.loc[self.dataset_csv['IMAGE_NAME'] == id, lesion] = label
    
        
    def subsample_k_shot_indices(self, k, lesions):
           
        slide_info = self.dataset_csv[self.dataset_csv["IMAGE_NAME"].isin(self.ids)]
        normal_samples_ids_run = slide_info[slide_info["Abnormality_sum"] == 0]["IMAGE_NAME"].to_list()

        k_shot_ids = []
        
        if k is None or k==-1 or k=='all':
            abnormal_samples_ids = slide_info[slide_info[lesions].eq(1).any(axis=1)]["IMAGE_NAME"].to_list()
            normal_samples_ids = random.sample(normal_samples_ids_run, len(abnormal_samples_ids))
            k_shot_ids += abnormal_samples_ids + normal_samples_ids
        else:
            for lesion in lesions:
                abnormal_samples_ids = slide_info[slide_info[lesion] == 1]["IMAGE_NAME"].to_list() 
                k_real = min(k, len(abnormal_samples_ids))
                if k_real < k:
                    abnormal_samples_ids = random.sample(abnormal_samples_ids, k_real)
                    normal_samples_ids = random.sample(normal_samples_ids_run, k_real)
                else:
                    abnormal_samples_ids = random.sample(abnormal_samples_ids, k)
                    normal_samples_ids = random.sample(normal_samples_ids_run, k)
                k_shot_ids += abnormal_samples_ids
                k_shot_ids += normal_samples_ids
        k_shot_indices = self.dataset_csv[self.dataset_csv['IMAGE_NAME'].isin(k_shot_ids)].index.tolist()

        
        return k_shot_indices


######################################################################################################### 
#                                                                                                       #
#                                      DATASET FOR SLIDE EMBEDDINGS                                     #
#                                                                                                       #
######################################################################################################### 

class SlideEmbeddingDataset(Dataset):
    
    """
    Dataset for slide embeddings and downstream evaluation tasks
    """

    def __init__(self, feature_folder_path, dataset_csv, lesions, equal_normal_abnormal, do_testset_label_modifications, ids=None, exclude_gradings=None):
        '''
        '''
        super(SlideEmbeddingDataset, self).__init__()
        self.feature_folder_path = feature_folder_path
        self.dataset_csv = dataset_csv
        self.ids = ids if ids else self.scan_folder(path=feature_folder_path)
        self.lesions = lesions
        self.excluded_samples = []
        self.equal_normal_abnormal = equal_normal_abnormal
        self.exclude_gradings = exclude_gradings
        
        self.excluded_slides_curated_testset_dict = EXCLUDED_SLIDES_CURATED_TESTSET_DICT
        self.excluded_slides_curated_testset_list = [item for sublist in self.excluded_slides_curated_testset_dict.values() for item in sublist]
        self.modifications_curated_testset = MODIFICATIONS_CURATED_TESTSET
        self.modifications_testset = MODIFICATIONS_TESTSET

        if do_testset_label_modifications:
            self.fuse_liver_classes = True
            self._modify_datasets()
        
        self.slide_feature, self.class_binary, self.class_multi, self.ids = self.get_features()
    
    def __len__(self):
        return len(self.slide_feature)
    
    def __getitem__(self, idx):
    
        slide_feature = self.slide_feature[idx]
        class_binary = self.class_binary[idx]
        class_multi = self.class_multi[idx]
        img_name = self.ids[idx]
        class_names = self.lesions
     
        return slide_feature, class_binary, class_multi, class_names, img_name 
    
    def print_dataset_summary(self):
        normal_samples = 0
        abnormal_samples = 0
        class_counts = None
        for tensor in self.class_multi:
            if torch.all(tensor == 0):
                normal_samples += 1
            elif torch.any(tensor == 1):
                abnormal_samples += 1
                if class_counts is None:
                    class_counts = torch.zeros_like(tensor)
                class_counts += tensor
        summary = f"Number of normal samples: {normal_samples}\n"
        summary += f"Number of abnormal samples: {abnormal_samples}\n"
        summary += "------------------\n"
        for i, lesion in enumerate(self.lesions):
            summary += f"Samples w. {lesion}: {int(class_counts[i])}\n"
        return summary

    def get_features(self):
        slide_features = []
        classes_binary = []
        classes_multi = []
        abnormal_idx = []
        ids = []
        num_abnormal = 0
        
        for idx, id in enumerate(self.ids):
            slide_info = self.dataset_csv[self.dataset_csv["IMAGE_NAME"] == id]
            # sometimes Normal slides have abnormalities - handle this...
            if (slide_info["Abnormality_sum"] == 0).bool(): 
                class_binary = torch.tensor([0.]).to(torch.float32) 
            else:
                class_binary = torch.tensor([1.]).to(torch.float32)
                num_abnormal += 1
                abnormal_idx.append(idx)
            class_multi = torch.tensor(slide_info[self.lesions].to_numpy()).to(torch.float32).squeeze()
            classes_binary.append(class_binary)
            classes_multi.append(class_multi)
            slide_features.append(self.load_wsi_embeddings_from_path(id))
            ids.append(id)
        
        normal_idx = [i for i, t in enumerate(classes_binary) if torch.any(t == 0)]
        if self.equal_normal_abnormal:
            normal_idx = random.sample(normal_idx, len(abnormal_idx))
        selected_idx = normal_idx + abnormal_idx
        
        return [slide_features[i] for i in selected_idx], [classes_binary[i] for i in selected_idx], [classes_multi[i] for i in selected_idx], [ids[i] for i in selected_idx]
          
          
    def load_wsi_embeddings_from_path(self, slide_id):
        feats_path = os.path.join(self.feature_folder_path, slide_id)
        with h5py.File(feats_path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            features = torch.from_numpy(features).to(torch.float32).squeeze()
            slide_feature = features.mean(dim=0) if len(features.shape) > 1 else features
        return slide_feature
    
    def scan_folder(self, path):
        file_names = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_names.append(file)
        return file_names
    
    
    def _modify_datasets(self):
        
        if self.fuse_liver_classes:
            # Fuse Hypertrophy = 1 if Eosinophilic Change, Ground Glass appearance
            self.dataset_csv["Hypertrophy"] = self.dataset_csv["Hypertrophy"] + self.dataset_csv["Change, eosinophilic"] + self.dataset_csv["Ground glass appearance"] + self.dataset_csv["Degeneration, hydropic"]
            self.dataset_csv.loc[self.dataset_csv['Hypertrophy'] > 1, 'Hypertrophy'] = 1
            # Fuse Necrosis and Single Cell Necrosis
            self.dataset_csv["Necrosis"] = self.dataset_csv["Necrosis"] + self.dataset_csv["Single cell necrosis"] 
            self.dataset_csv.loc[self.dataset_csv['Necrosis'] > 1, 'Necrosis'] = 1
            # Fuse Profileration bile duct and oval cell
            self.dataset_csv["Proliferation"] = self.dataset_csv["Proliferation, oval cell"] + self.dataset_csv["Proliferation, bile duct"] 
            self.dataset_csv.loc[self.dataset_csv['Proliferation'] > 1, 'Proliferation'] = 1
        
        # modify testset labels
        for lesion, wsi_labels in self.modifications_testset.items():
            for id, label in wsi_labels.items():
                self.dataset_csv.loc[self.dataset_csv['IMAGE_NAME'] == id, lesion] = label
     
        # modify curated testset labels
        for lesion, wsi_labels in self.modifications_curated_testset.items():
            for id, label in wsi_labels.items():
                self.dataset_csv.loc[self.dataset_csv['IMAGE_NAME'] == id, lesion] = label
        
        
    def get_few_shot_binary_datasets(self, lesions, k, test_flag=False):
        
        dataset_dict = {}
        slide_info = self.dataset_csv[self.dataset_csv["IMAGE_NAME"].isin(self.ids)]
        total_normal_samples_ids = slide_info[slide_info["Abnormality_sum"] == 0]["IMAGE_NAME"].to_list()
        total_ids = slide_info["IMAGE_NAME"].to_list()
        
        if k is not None and k != -1 and k != 'all':
            normal_samples_ids_run = random.sample(total_normal_samples_ids, k)    
        
        for lesion in lesions:
            features = []
            binary_classes = []
            lesion_dict = {}
            abnormal_samples_ids = slide_info[slide_info[lesion] == 1]["IMAGE_NAME"].to_list()    
            
            # k-shot train scenario
            if k is not None and k != -1 and k != 'all':
                k_real = min(k, len(abnormal_samples_ids))
                if k_real < k:
                    abnormal_samples_ids = random.sample(abnormal_samples_ids, k_real)
                    normal_samples_ids = random.sample(normal_samples_ids_run, k_real)
                else:
                    abnormal_samples_ids = random.sample(abnormal_samples_ids, k)
                    normal_samples_ids = normal_samples_ids_run
            # train scenario for all samples and test scenario
            else:
                if test_flag:
                    normal_samples_ids = list(set(total_ids)-set(abnormal_samples_ids))
                else: 
                    normal_samples_ids = random.sample(total_normal_samples_ids, len(abnormal_samples_ids))

            # get features and classes
            for normal_id in normal_samples_ids:
                features.append(np.array(self.load_wsi_embeddings_from_path(slide_id=normal_id)))
                binary_classes.append(np.array(0.))
            for abnormal_id in abnormal_samples_ids:  
                features.append(np.array(self.load_wsi_embeddings_from_path(slide_id=abnormal_id)))     
                binary_classes.append(np.array(1.))      
            lesion_dict["features"] = np.vstack(features)
            lesion_dict["binary_classes"] = np.vstack(binary_classes)
            
            if k is not None and k != -1 and k != 'all':
                if k_real < k:
                    print(f"WARNING: Sampled only {lesion_dict['features'].shape[0]} instead of {k*2} samples for lesion {lesion}")
                else:
                    if lesion_dict['features'].shape[0] != k*2:
                        raise ValueError(f"Not enough features sampled for lesion {lesion}")
                
            dataset_dict[lesion] = lesion_dict
        
        return dataset_dict
    