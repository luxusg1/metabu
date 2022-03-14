from .smac import SMAC_Optimizer


class BestHPNeighborSampling(SMAC_Optimizer):
    def __init__(self, hp_distribution, classifier, flow_id, seed, nb_iterations, n_jobs):
        super(BestHPNeighborSampling, self).__init__(hp_distribution=hp_distribution,
                                                     classifier=classifier,
                                                     flow_id=flow_id,
                                                     seed=seed,
                                                     nb_iterations=nb_iterations,
                                                     n_jobs=n_jobs,
                                                     nb_init=nb_iterations)
