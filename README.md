# LCR-Rot-hop-Kfont++

Source code for injecting knowledge from a domain-specific ontology into LCR-Rot-hop++ inspired by work on Kformer for Aspect-Based Sentiment Classification. 

## Setup
- Create environment
   - Create a conda environment in Python 3.10
   - Install the required packages by running pip install -r requirements.txt in the terminal
 
- Download data
  - Download the following files
    - Train and test data from [SemEval 2015](http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools)
    - Train and test data from [SemEval 2016](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)
    - Restaurant [ontology](https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData)
  - Add the files to 'data/raw' and rename them to the following:
    - `ABSA15_Restaurants_Test.xml`
    - `ABSA15_Restaurants_Train.xml`
    - `ABSA16_Restaurants_Test.xml`
    - `ABSA16_Restaurants_Train.xml`
    - `ontology.owl'


## Usage

To view the available cli args for a program, run `python [FILE] --help`. These CLI args can for example be used to pick
the year of the dataset.

- `main_preprocess.py`: remove opinions that contain implicit targets and generate embeddings, these embeddings are used
  by the other programs. To generate all embeddings for a given year, run `python main_preprocess.py --all`
- `main_hyperparam.py`: run hyperparameter optimization
- `main_train.py`: train the model for a given set of hyperparameters
- `main_validate.py`: validate a trained model.
  python main_validate.py --model "Model path"

## Acknowledgements

Code and ideas are used from:
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus.git
- https://github.com/StijnCoremans/LCR-Rot-hop-ont.git
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
- https://github.com/KRR-Oxford/OWL2Vec-Star.git
- Trusca, M. M., Wassenberg, D., Frasincar, F., and Dekker, R. (2020). A hybrid approach
  for aspect-based sentiment analysis using deep contextual word embeddings and hierarchical
  attention. In 20th Conference on Web Engineering (ICWE 2020), volume 12128 of LNCS, pages 365–380. Springer.
- Yao, Y., Huang, S., Dong, L., Wei, F., Chen, H., and Zhang, N. (2022). Kformer: Knowledge
  Injection in Transformer Feed-Forward Layers. In 11th International Conference on Natural
  Language Processing and Chinese Computing (NLPCC 2022), volume 13551, pages 131–143. Springer.

