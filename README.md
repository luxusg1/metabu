# Metabu
Metabu is meta-features learned  via Optimal transport procedure, aligning the manually designed meta-features (hand-crafted) with the space of distributions on the hyper-parameter configurations (target representation).

# Intallation 
There are two way to install metabu :
## Manually installation
First, you can do it manually by installing all required package: 

`pip install -r requirements.txt`

Then install metabu using :

`python setup.py install`

## Installation with singularity 
install singularity using this command :  

`apt-install singularity` 

follow this command to setup environement needed for using metabu :
`singularity build --sandbox metabu.sif create_metabu_vm.def`

## Run training of Metabu meta-features

To train metabu metafeatures, run the following `sigularity` command:

`singulartiy exec python main.py --metafetures_file='examples/data/adaboost.csv'`

NB : if you are not using `singularity` you can remove the `singularity exec` 


All Options available on `main.py` can be show using help option :

`python main.py --help `

## Cite :
` @inproceedings{ HeriICLR,

    title     = {Learning Meta-features for AutoML},
    author    = {Herilalaina Rakotoarison, Louisot Milijaona and Andry Rasoanaivo and Michèle Sebag and Marc Shöenawer},
    booktitle = {ICLR 2022},
    year      = {2022}

}`
