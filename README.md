## Dependencies
```
rdkit
python 3.8.8
pytorch 1.12.0
torch_geometric 2.3.1
torch_scatter 2.1.1
```
You also could choose to [download](https://drive.google.com/file/d/1MqYoBrIyhg1jUgXigO3LVKO2P9K3VZow/view) our ACME.tar.gz(2.8GB) which is created by conda pack. After you download, run
```
$ cd your_conda/envs
$ mkdir acme
$ cd acme
$ tar -xzf ACME.tar.gz
$ rm -rf ACME.tar.gz
$ conda activate acme
```
## Unsupervised Learning
### Run
```
$ cd unsupervised
$ mkdir pth
$ bash us_run.sh 0
# where 0 is the GPU device id to use
```
## Transfer Learning
### Prepare the Dataset
```
$ cd transfer
$ wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
$ unzip chem_dataset.zip
$ rm -rf dataset/*/processed
```
### Run the Fine-tuning
```
$ bash tran_run.sh finetune 0
```
## Semi-supervised Learning
### Run the Fine-tuning
```
$ cd semi_supervised
$ bash semi_run.sh finetune 0
```
## Citation
