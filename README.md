## SCSN
A cross-scene generalization method for remote sensing scene classification

This repo is based on [this repo](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG). More details can be found in it

Code will be updated soon

## Requirement
* python 3
* pytorch 1.10 or above

## Datasets
Four scene classification datasets: `AID, CLRS, MLRSN, and RSSCN7`

The specific data in our experiments is recorded in the ./data

If you want to use your own dataset, please organize your data in the following structure

```
RootDir
└───Domain1Name
│   └───Class1Name
│       │   file1.jpg
│       │   file2.jpg
│       │   ...
│   ...
└───Domain2Name
|   ...    
```
And then, modifty `util/util.py` to contain the dataset

## Usage

1. Modify the file in the scripts

2. The main script file is `train.py`, and you can follow `run_helper.txt` to train or infer the model

## Customization

It is easy to design your own method following the steps:

1. Add your method (a Python file) to `alg/algs`, and add the reference to it in the `alg/alg.py`

2. Modify `utils/util.py` to make it adapt your own parameters

3. Modify `scripts/run.sh` and execuate it

## Results
|  Method  |    A   |    C   |    M   |    R   |   avg  |
|:--------:|:------:|:------:|:------:|:------:|:------:|
|    ERM   | 93.16  | 83.88  | 75.54  | 72.61  | 81.30  |
|   DANN   | 93.80  | 85.36  | 73.07  | 73.79  | 81.51  |
|    MMD   | 95.04  | 84.45  | 75.19  | 72.50  | 81.80  |
|   CORAL  | 94.92  | 86.01  | 74.78  | 73.52  | 82.31  |
|    SNR   | 93.68  | 82.33  | 74.62  | 73.64  | 81.07  |
| GroupDRO | 94.04  | 85.50  | 74.56  | 71.50  | 81.40  |
|    ARM   | 94.20  | 83.81  | 75.29  | 72.75  | 81.51  |
|  SagNet  | 93.32  | 82.60  | 77.52  | 72.96  | 81.60  |
|   mixup  | 93.72  | 83.83  | 75.14  | 74.36  | 81.76  |
|   VREx   | 93.96  | 83.12  | 77.35  | 73.14  | 81.89  |
|  ANDmask | 94.68  | 83.33  | 75.11  | 74.46  | 81.90  |
|  IB_ERM  | 93.84  | 85.43  | 74.45  | 73.96  | 81.92  |
|    RSC   | 94.80  | 82.93  | 76.40  | 74.54  | 82.17  |
|  IB_IRM  | 94.60  | 85.69  | 74.36  | 74.11  | 82.19  |
|    IRM   | 94.80  | 87.50  | 74.22  | 73.57  | 82.52  |
|   SCSN   | 95.80  | 86.17  | 77.66  | 75.29  | 83.73  |
|   SCSN_bt   | 95.68  | 86.26  | 77.40  | 75.82  | 83.79  |

> 
## Reference

```
Wait for publication.
```

or in bibtex style:

```
Wait for publication.
```
