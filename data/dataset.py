import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset
from torch.utils.data import Dataset
import pandas as pd



class TextDataSet:
    def __init__(self, config, map_dict):
        self.path = config['data_path']
        self.load_data(config, map_dict)

    def load_data(self, config, map_dict):
        self.item2iid = {}
        with open(f'{self.path}/{config.dataset}.item2index', 'r') as f:

            for line in f.readlines():
                item, iid = line.rstrip('\n').split('\t')
                self.item2iid[item] = map_dict[iid]
                # self.item2iid[item] = int(iid)


        self.iid2text = {}
        with open(f'{self.path}/{config.dataset}.text', 'r') as f:

            next(f)
            for line in f.readlines():
                item, text = line.rstrip('\n').split('\t', 1)
                self.iid2text[self.item2iid[item]] = text

        self.texts = []
        for i in range(len(self.iid2text)):
            self.texts.append(self.iid2text[i + 1])


class CustomizeSequentialRecDataset(Dataset):

    def __init__(self, config, fname):

        self.max_seq_length = config['max_seq_length']
        data = pd.read_csv(fname, sep='\t')
        self.df2csr(data)

    def df2csr(self, data, split=True):
        interns = data["item_id_list:token_seq"]
        interns = interns.values.tolist()
        self.interns = self.padding(interns, split=split)

        self.label = data['item_id:token'].values.tolist()
        self.label = [int(label) for label in self.label]
        self.label = np.array(self.label, dtype=int)
        return

    def padding(self, interns, split=True):

        self.seq_length = []
        for i, x in enumerate(interns):
            if split:
                x = list(map(int, x.split()))
            else:
                x = x.tolist()
            x = x[-self.max_seq_length:]
            x = [e + 1 for e in x]
            num_pads = self.max_seq_length - len(x)
            interns[i] = x + [0] * num_pads

            self.seq_length.append(len(x))

        self.seq_length = np.array(self.seq_length)
        return np.array(interns, dtype=int)

    def __len__(self):
        return self.interns.shape[0]

    def __getitem__(self, idx):

        batch_interns = self.interns[idx]
        batch_label = self.label[idx] + 1
        batch_length = self.seq_length[idx]

        return batch_interns, batch_label.squeeze(), batch_length.squeeze()

class TestSequentialRecDataset(CustomizeSequentialRecDataset):

    def __getitem__(self, idx):

        batch_interns = self.interns[idx]
        batch_label   = self.label[idx]+1
        batch_length  = self.seq_length[idx]

        return torch.tensor(batch_interns, dtype=torch.long), torch.tensor(batch_label.squeeze(), dtype=torch.long), torch.tensor(batch_length.squeeze(), dtype=torch.long)

def collate_fn(batch):
    batch_interns, batch_label, batch_length = [item[0] for item in batch], [item[1] for item in batch], [item[2] for item in batch]
    batch_interns, batch_label, batch_length = np.array(batch_interns), np.array(batch_label), np.array(batch_length)
    all_items = np.concatenate((batch_interns.flatten(), batch_label.flatten()), axis=0)
    unique_items = sorted(np.unique(all_items))

    idx_mapping = dict(zip(unique_items, np.arange(len(unique_items))))
    batch_interns = np.vectorize(idx_mapping.get)(batch_interns)
    batch_label   = np.vectorize(idx_mapping.get)(batch_label)

    return torch.tensor(batch_interns), torch.tensor(batch_label), torch.tensor(batch_length), torch.tensor(unique_items)


class TestSequentialRecDataset(CustomizeSequentialRecDataset):

    def __getitem__(self, idx):
        batch_interns = self.interns[idx]
        batch_label = self.label[idx] + 1
        batch_length = self.seq_length[idx]

        return torch.tensor(batch_interns, dtype=torch.long), torch.tensor(batch_label.squeeze(),
                                                                           dtype=torch.long), torch.tensor(
            batch_length.squeeze(), dtype=torch.long)

class FederatedDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.ft_layer = config['ft_layer']
        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        self.device = config['device']
        plm_embedding_weight, self.layer_embedding = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, 12, self.plm_size)
        loaded_feat = loaded_feat[:, -self.ft_layer:, :].reshape(-1, self.ft_layer * self.plm_size)

        mapped_feat = np.zeros((self.item_num, self.plm_size * self.ft_layer))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(token)]
        layer_embedding = torch.as_tensor(mapped_feat.reshape(-1, self.ft_layer, self.plm_size), dtype=torch.float32)
        return mapped_feat, layer_embedding.to(self.device)

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.ft_layer * self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding

class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, 12, self.plm_size)[:, -1, :]
        # loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding


class PretrainUniSRecDataset(UniSRecDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_suffix_aug = config['plm_suffix_aug']
        plm_embedding_weight_aug = self.load_plm_embedding(plm_suffix_aug=self.plm_suffix_aug)
        self.plm_embedding_aug = self.weight2emb(plm_embedding_weight_aug)

    def load_plm_embedding(self, plm_suffix_aug=None):
        with open(osp.join(self.config['data_path'], f'{self.dataset_name}.pt_datasets'), 'r') as file:
            dataset_names = file.read().strip().split(',')
        self.logger.info(f'Pre-training datasets: {dataset_names}')

        d2feat = []
        for dataset_name in dataset_names:
            if plm_suffix_aug is None:
                feat_path = osp.join(self.config['data_path'], f'{dataset_name}.{self.plm_suffix}')
            else:
                feat_path = osp.join(self.config['data_path'], f'{dataset_name}.{plm_suffix_aug}')
            loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
            d2feat.append(loaded_feat)

        iid2domain = np.zeros((self.item_num, 1))
        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            did, iid = token.split('-')
            loaded_feat = d2feat[int(did)]
            mapped_feat[i] = loaded_feat[int(iid)]
            iid2domain[i] = int(did)
        self.iid2domain = torch.LongTensor(iid2domain)

        return mapped_feat
