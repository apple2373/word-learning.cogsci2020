# word-learning.cogsci2020
- See https://arxiv.org/abs/2006.02802
```
@inproceedings{tsutsui2020wordlearning,
	Author = {Satoshi Tsutsui and Arjun Chandrasekaran and Md Reza and David Crandall and Chen Yu},
	Booktitle = {Annual Conference of the Cognitive Science Society (CogSci)},
	Title = {A Computational Model of Early Word Learning from the Infant's Point of View},
	Year = {2020}
  }
```

## Environment
See [`env.yml`](./env.yml) for the exact environment. As a shortcut, you can use the following python binary as is. 
```
/data/stsutsui/public/word-learning.cogsci2020/miniconda/bin/python
```

## Image Data
I copied the necessary image files into the following. 
- `./data/naming_3s_imgs`
- `./data/test_images`


The naming rule is same as experiment 13 where `cam07` is child view and `cam08` is parent view, and `frames` are raw frames, and `acuity` is after applied acuity filter. These are not included in the github but is in the salk. 


## Training Data
The point of this work is to make a training set based on a criteria, and then train CNNs. I represent each set as a txt file. These files are inside [`./data/dataset_txt`](./data/dataset_txt). The name is sort of informative enough to figure out the corresponding set mentioned in the paper. It's a simple csv file to list path to images and corresponding labels. 

## CNN training
To train cnns, you need to go to the `./train_cnn` directory. [`main.py`](./train_cnn/main.py) is the training script. This script should be readable. You can see the help to get the meanings of args.
```
cd train_cnn
/data/stsutsui/public/word-learning.cogsci2020/miniconda/bin/python main.py --help
```

An example command to train is:
```
cd train_cnn
/data/stsutsui/public/word-learning.cogsci2020/miniconda/bin/python main.py --saveroot ../experiments/cogsci2020/ --train  ../data/dataset_txt/naiming_3s_whole_img_acuity.txt --seed 1 --gpu 1
```
This trains with the subset of `naiming_3s_whole_img_acuity.txt`. 
  
All the training I did for this paper is [`all_trainings.sh`](./train_cnns/all_trainings.sh)  . The training results are in `./experiments/cogsci2020-reported/`, which is included in the salk, but not in this github repository.

## Results
The results plot used in the paper is in [`./results`](./results). The code to make these plots are [`./ipython/results-cnn-train-reported-cogsci2020.ipynb`](./ipython/results-cnn-train-reported-cogsci2020.ipynb)
