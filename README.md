# Graph Few-shot Class Incremental Learning (GFCIL)
![plot](./frame_model.png)
## Requirements
`python==3.7.10`
`pytorch==1.8.1`
`cuda=11.1`
`tqdm==4.59.0`
## Useage
### Pretrain
`python pretrain.py --use_cuda --dataset Amazon_clothing --seed 1234` 
### Meta-train and Evaluation
`python meta-train.py --use_cuda --dataset Amazon_clothing --episodes 1000 --incremental --attention --checkpoint 100 --way 5 --shot 5`
