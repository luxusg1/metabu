.. toctree::
   :maxdepth: 2


Installation
-------------
Install with pip:

.. code-block:: bash

   pip install -r requirements.txt
   python setup.py install


Use Singularity:

* build *local* container with definition file `env/metabu.def`.
* fetch *remote* container from (coming soon).

Usage
-------------

Simple to use:

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
----------

.. tip::

    All next cited parameters are also attributes of Metabu class.


The `metabu.metabu.Metabu` class
-----------------------------------
.. autoclass:: metabu.metabu.Metabu
   :members:


Cite Metabu
-----------

.. code-block:: bash

   @inproceedings{rakotoarison2022learning,
    title       = {Learning meta-features for Auto{ML}},
    author      = {Herilalaina Rakotoarison and Louisot Milijaona and Andry RASOANAIVO and Michele Sebag and Marc Schoenauer},
    booktitle   = {International Conference on Learning Representations},
    year        = {2022},
    url         = {https://openreview.net/forum?id=DTkEfj0Ygb8}
   }



