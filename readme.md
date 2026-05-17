# IIE: Intra-sample and Intra-modal Enhancement for Multimodal Sentiment Analysis with Missing Modalities
The code for Intra-sample and Intra-modal Enhancement for Multimodal Sentiment Analysis with Missing Modalities, which is accepted by IEEE Transactions on Multimedia ([TMM](https://ieeexplore.ieee.org/document/11303977)). The code is being organized and will be released.

### Prerequisites:
```
* Python 3.8.10
* CUDA 11.5
* pytorch 1.12.1+cu113
* sentence-transformers 3.1.1
* transformers 4.30.2
```
**Note that the torch version can be changed to your cuda version, but please keep the transformers==4.30.2 as some functions will change in later versions**

### Pretrained model:
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ./BERT-EN/.

### Datasets:
**Please move the following datasets into directory ```./datasets/```**

The aligned CMU-MOSI and CMU-MOSEI datasets can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1FW7A-gfbzomKjj_-7ZR8Og?pwd=8xii) with extraction code 8xii, rename the pkl as ```aligned_{dataset}.pkl```. 

### Run IIE
For fixed missing scenarios on MOSI and MOSEI datasets, please run the following code by change ```--dataset```:
```
bash run.sh
```

### Citation:
Please cite our paper if you find our work useful for your research:
```
@article{zhuang2026iie,
  author={Zhuang, Yan and Zhang, Yanru and Deng, Jiawen and Ren, Fuji},
  journal={IEEE Transactions on Multimedia}, 
  title={Intra-Sample and Intra-Modal Enhancement for Multimodal Sentiment Analysis With Missing Modalities}, 
  year={2026},
  volume={28},
  number={},
  pages={1847-1859},
  doi={10.1109/TMM.2025.3645559}}
```

### Acknowledgement
Thanks to [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck), [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer), [MCL](https://github.com/TmacMai/Multimodal-Correlation-Learning), [HKT](https://github.com/matalvepu/HKT), [LFMIM](https://github.com/sunjunaimer/LFMIM) and [GCNet](https://github.com/zeroQiaoba/GCNet) for their great help to our codes and research. 
