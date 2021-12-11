
## Knowledge Graph Embedding Bias Analysis with Poincare and Euclidean embedding methods


### Language Selection

For the DBpedia knowledge graphs we are looking into, we decided to pick English (en), Swedish (sv), and Indonesian (id).

We picked English because it is the most relevant language of choice. We picked Swedish because Scadanavian countries 
are known for exhibiting less gender inequality, and Swedish has the largest DBpedia collection out of all other
Scandanavian countries (Swedish (221.1 Mb), Danish (20.1 Mb), Norwegian (32.6 Mb), Icelandic (2.6 Mb), and Faroese 
(496 Kb)). Similarly, we picked Indonesian because it is one of countries with relatively more orthodox 
traditions and has the most data (Indonesian (60.3 Mb), Urdu (21 Mb), Hindi (11.3 Mb), and Arabic (5.5 Mb))

### Data Preprocessing

To run data preprocessing code, see in main functions in `data_explorations.py`, specifically, `import_dbpedia_data` 
for dbpedia datasets.


### To train a model

To train a model, see code in `run_train.py`

Available datasets are:
    
    FB15k-237
    WN18RR
    DBpedia-english
    DBpedia-Swedish
    DBpedia-Indonesian
    (and other languages using `/download_datasets.sh` file

Our training statistics is automatically logged in WanDB. To install:
```bash
pip install wandb
wandb login
```

### To Compute Bias
First you need three files for the dataset:
- humans.ent
- gender.ent
- professions.ent

To generate those, run `data_exploration.extract_dbpedia_entities`

Then, see code in `run_predict.py`


### Disclaimer
The work we implemented is followed after the original [MuRP repo](https://github.com/ibalazevic/multirelational-poincare)