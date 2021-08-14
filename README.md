# Graph Few-shot Class Incremental Learning (GFCIL)
## Requirements
`python==3.7.10`

`pytorch==1.8.1`

`cuda=11.1`

`tqdm==4.59.0`
## Useage
### Go to the directory
`cd incremental`
### Pretrain
`python pretrain.py --use_cuda --dataset Amazon_clothing --seed 1234` 
### Meta-train and Evaluation
`python meta-train.py --use_cuda --dataset Amazon_clothing --episodes 1000 --incremental --checkpoint 100 --way 5 --shot 5`
