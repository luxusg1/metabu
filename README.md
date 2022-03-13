# Metabu
This is the Official code for ICLR 2022 paper **"Learning meta-features for Automl"**.
If you have any questions about code or the paper, we are happy to help you!

We use the implementation of the ICML 2020 work **"Learning Autoencoders with Relational Regularization"** [https://arxiv.org/pdf/2002.02913.pdf] 
to compute the fused gromov loss.

# Intallation
First, you must install all required package using: 

`pip install -r requirements.txt`

Then install metabu using :

`python setup.py install`

## Run training of Metabu meta-features

To train metabu metafeatures, run the following command:

```
python main.py 
            --metafetures_file = <the metafeatures file> # csv file
            --target_representation_file = <the target representation file> # csv file
            --store = <the file to store the Metabu meta-features> 
            --top_k = <the number of top target_representation_file used in the traning>
            --ranking_column_name = <the name of column to rank the target representation in the target representation file>
```

All Options available on `main.py` can be show using help option :

`python main.py --help `


NB : 
- Note that both the `metafetures_file` and `target_representation_file` must have a column named `task_id` to refer all data corresponding to this datasets.
- If a `task_id` is missing in one the file we don't use it to train the metabu metafeatures and use only the remaining `task_id`  available in both file.

## Cite :
If you use Metabu in your scientific project or publication, we would appreciate citations.


``` 
@inproceedings{rakotoarison2022learning,
    title       = {Learning meta-features for Auto{ML}},
    author      = {Herilalaina Rakotoarison and Louisot Milijaona and Andry RASOANAIVO and Michele Sebag and Marc Schoenauer},
    booktitle   = {International Conference on Learning Representations},
    year        = {2022},
    url         = {https://openreview.net/forum?id=DTkEfj0Ygb8}
}
```
