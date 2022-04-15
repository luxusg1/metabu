.. toctree::
   :maxdepth: 2

Metabu: Learning Meta-Features
-------------------------------

This work tackles the AutoML problem :cite:p:`hutter_automated_2019`, aimed to automatically select an ML algorithm and its hyper-parameter configuration most appropriate to the dataset at hand. The proposed approach, MetaBu, learns new meta-features via an Optimal Transport procedure :cite:p:`titouan_optimal_2019`, aligning the manually designed meta-features :cite:p:`rivolli_meta-features_2022` with the space of distributions on the hyper-parameter configurations. MetaBu meta-features, learned once and for all, induce a topology on the set of datasets that is exploited to define a distribution of promising hyper-parameter configurations amenable to AutoML. Experiments on the OpenML CC-18 benchmark :cite:p:`bischl_openml_2017` demonstrate that using MetaBu meta-features boosts the performance of state of the art AutoML systems, AutoSklearn :cite:p:`feurer_efficient_2015` and Probabilistic Matrix Factorization :cite:p:`fusi_probabilistic_2018`. Furthermore, the inspection of MetaBu meta-features gives some hints into when an ML algorithm does well. Finally, the topology based on MetaBu meta-features enables to estimate the intrinsic dimensionality of the OpenML benchmark w.r.t. a given ML algorithm or pipeline.

.. image:: ../illustration_metabu.png
   :align: center
   :width: 600



Installation
-------------
Install with pip:

.. code-block:: bash

   pip install -r requirements.txt
   python setup.py install


or with Singularity:

* build *local* container with definition file `env/metabu.def`.
* fetch *remote* container from (coming soon).

Example
--------

.. code-block:: python

   from metabu import Metabu

   basic_representations = pd.read_csv(...)
   target_representations = pd.read_csv(...)
   metabu = Metabu()
   metabu.train(basic_reprs=basic_representations,
                target_reprs=target_representations,
                column_id="task_id")
   metabu.predict(basic_reprs=basic_representations)
   metabu.get_importances()

Experiments
------------

> Script to reproduce experiments will be available under the **experiments** branch.


API
---
.. autoclass:: metabu.Metabu
   :members:


Cite Metabu
-----------

.. code-block:: bash

   @inproceedings{rakotoarison2022learning,
    title       = {Learning meta-features for Auto{ML}},
    author      = {Herilalaina Rakotoarison and Louisot Milijaona and Andry Rasoanaivo and Michele Sebag and Marc Schoenauer},
    booktitle   = {International Conference on Learning Representations},
    year        = {2022},
    url         = {https://openreview.net/forum?id=DTkEfj0Ygb8}
   }


References
-----------

.. bibliography::

