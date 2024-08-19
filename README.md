# FFMSR

>Ziang Lu, Lei Guo, Xu Yu, Zhiyong Cheng, Xiaohui Han, Lei Zhu. Federated Semantic Learning for Privacy-preserving
Cross-domain Recommendation

![](asset/PFCR.jpg)

## Requirements

```
recbole==1.0.1
python==3.8.13
cudatoolkit==11.3.1
pytorch==1.11.0
```

## Dataset

You should download the raw Amazon dataset from [Amazon](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) and then put them in `dataset/raw`. You can download the processed dataset we used in the paper from [百度网盘](https://pan.baidu.com/s/1_32S8znijLWVTGZIdEe8XA?pwd=fbw8).



## Quick Start

### Data Preparation

Preparing data:

```bash
python process_amazon_ML.py --output_path your_dataset_path
```


### Federated pre-train

```bash
python fedtrain.py
```
Before train, you need to modify the relevant configuration in the configuration files `props/FFMSR.yaml` and `props/pretrain.yaml`. 

Here are some important parameters in `props/pretrain.yaml` you may need to modify:

1.`data_path`: The path of the dataset you want to use for pre-training.

2.`cluster_centroids`: The number of cluster centroids for clustering in the server.

3.`cluster_iters`: Maximum number of rounds of clustering, beyond which clustering will be paused if not yet completed.

```bash
python single_train.py --d=OA --p=your_pretrained_model.pth
```

You also need to modify the `index_pretrain_dataset` in `props/finetune.yaml` to the abbreviation of the first letter of the current single dataset. The `pq_data` is consistent with the `index_pretrain_dataset` in `props/pretrain.yaml`.
### Fine-tuning after federated pre-training

Prompt finetune pre-trained recommender of "Pantry":

```bash
python finetune.py --d=Pantry --p=your_pretrained_model.pth
```
You can adjust the corresponding parameters for fine-tuning in  `props/finetune.yaml`.
