# LCR-Rot-hop-ont++

Source code for Injecting Knowledge from a Domain Sentiment Ontology in a Neural Approach for Aspect-Based Sentiment
Classification.

## Installation

### Data

First, create a `data/raw` directory and download
the [SemEval 2015](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools), [SemEval 2016](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)
datasets, and the [ontology](https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData). Then
rename the SemEval datasets to end up with the following files:

- `data/raw`
    - `ABSA15_Restaurants_Test.xml`
    - `ABSA15_Restaurants_Train.xml`
    - `ABSA16_Restaurants_Test.xml`
    - `ABSA16_Restaurants_Train.xml`
    - `ontology.owl-Extended.owl`

### Setup environment

Create a conda environment with Python version 3.10, the required packages and their versions are listed
in `requirements.txt`, note that you may need to install some packages using `conda install` instead of `pip install`
depending on your platform.

## Usage

To view the available cli args for a program, run `python [FILE] --help`. These CLI args can for example be used to pick
the year of the dataset.

- `main_preprocess.py`: remove opinions that contain implicit targets and generate embeddings, these embeddings are used
  by the other programs. To generate all embeddings for a given year, run `python main_preprocess.py --all`
- `main_hyperparam.py`: run hyperparameter optimization
- `main_train.py`: train the model for a given set of hyperparameters
- `main_validate.py`: validate a trained model. To do an ablation experiment, run `python main_validate.py --ablation`,
  this requires all embeddings to be created for a given year.

## Acknowledgements

Code and ideas are used from:
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus.git
- https://github.com/StijnCoremans/LCR-Rot-hop-ont.git
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
- Tru ̧scˇa, M. M., Wassenberg, D., Frasincar, F., and Dekker, R. (2020). A hybrid approach
  for aspect-based sentiment analysis using deep contextual word embeddings and hierarchical
  attention. In 20th Conference on Web Engineering (ICWE 2020), volume 12128 of LNCS, pages 365–380. Springer.
- Yao, Y., Huang, S., Dong, L., Wei, F., Chen, H., and Zhang, N. (2022). Kformer: Knowledge
  Injection in Transformer Feed-Forward Layers. In 11th International Conference on Natural
  Language Processing and Chinese Computing (NLPCC 2022), volume 13551, pages 131–143. Springer.

