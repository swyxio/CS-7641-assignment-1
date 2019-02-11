import warnings

import numpy as np
import sklearn

import experiments
import learners


class KNNExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/KNN.py

        params = None
        complexity_param = None

        best_params = None
        if self._details.ds_name == "spam":
            params = {
                "KNN__metric": ["minkowski", "chebyshev", "euclidean"],
                "KNN__n_neighbors": np.arange(1, 21, 3),
                # "KNN__n_neighbors": [3, 5, 7, 9],
                "KNN__weights": ["uniform", "distance"],
            }
            complexity_param = {
                "name": "KNN__n_neighbors",
                "display_name": "Neighbor count",
                "values": np.arange(1, 21, 3),
                # "values": [3, 5, 7, 9],
                # "values": [1, 2, 3, 4],
            }
        elif self._details.ds_name == "poisonous_mushrooms":
            params = {
                "KNN__metric": ["minkowski", "chebyshev", "euclidean"],
                "KNN__n_neighbors": np.arange(1, 21, 3),
                "KNN__weights": ["uniform", "distance"],
            }
            complexity_param = {
                "name": "KNN__n_neighbors",
                "display_name": "Neighbor count",
                "values": np.arange(1, 21, 3),
            }
        # # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # # the various graphs
        # # #
        # if self._details.ds_name == "spam":
        #     best_params = {
        #         "metric": "chebyshev",
        #         "n_neighbors": 7,
        #         "weights": "uniform",
        #     }
        # elif self._details.ds_name == "poisonous_mushrooms":
        #     best_params = {
        #         "metric": "chebyshev",
        #         "n_neighbors": 7,
        #         "weights": "uniform",
        #     }

        learner = learners.KNNLearner(n_jobs=self._details.threads)
        if best_params is not None:
            learner.set_params(**best_params)

        experiments.perform_experiment(
            self._details.ds,
            self._details.ds_name,
            self._details.ds_readable_name,
            learner,
            "KNN",
            "KNN",
            params,
            complexity_param=complexity_param,
            seed=self._details.seed,
            best_params=best_params,
            threads=self._details.threads,
            verbose=self._verbose,
        )

