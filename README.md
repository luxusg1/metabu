# Metabu - Learning meta-features (Experiments)


> **This repository is still under active developement.**


## Installation
Install with pip:

```bash
pip install -r requirements.txt
python setup.py install
```

Use Singularity:
* build *local* container with definition file `env/metabu.def`.
* fetch *remote* container from (coming soon).

## Usage

Simple to use:

```python
from metabu import Metabu

basic_representations = pd.read_csv(...)
target_representations = pd.read_csv(...)
metabu = Metabu()
metabu.train(basic_reprs=basic_representations,
             target_reprs=target_representations,
             column_id="task_id")
metabu.predict(basic_reprs=basic_representations)
metabu.get_importances()
```


Try: `cd examples; python metabu_adaboost.py`

Feel free to create an issue if you have questions.


## Experiments

> Script to reproduce experiments will be available under the **experiments** branch.



## Cite Metabu

``` 
@inproceedings{rakotoarison2022learning,
    title       = {Learning meta-features for Auto{ML}},
    author      = {Herilalaina Rakotoarison and Louisot Milijaona and Andry Rasoanaivo and Michele Sebag and Marc Schoenauer},
    booktitle   = {International Conference on Learning Representations},
    year        = {2022},
    url         = {https://openreview.net/forum?id=DTkEfj0Ygb8}
}
```
