import copy
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from collections import Counter

from sklearn import preprocessing, utils
import sklearn.model_selection as ms
from scipy.sparse import isspmatrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import os
import seaborn as sns

from abc import ABC, abstractmethod

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = "./output"

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists("{}/images".format(OUTPUT_DIRECTORY)):
    os.makedirs("{}/images".format(OUTPUT_DIRECTORY))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_pairplot(title, df, class_column_name=None):
    plt = sns.pairplot(df, hue=class_column_name)
    return plt


# Adapted from https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count / n) * np.log((count / n)) for clas, count in classes])
    return H / np.log(k) > 0.75


class DataLoader(ABC):
    def __init__(self, path, verbose, seed):
        self._path = path
        self._verbose = verbose
        self._seed = seed

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def load_and_process(self, data=None, preprocess=True):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and should be called before any processing is done.

        :return: Nothing
        """
        if data is not None:
            self._data = data
            self.features = None
            self.classes = None
            self.testing_x = None
            self.testing_y = None
            self.training_x = None
            self.training_y = None
        else:
            self._load_data()
        self.log(
            "Processing {} Path: {}, Dimensions: {}",
            self.data_name(),
            self._path,
            self._data.shape,
        )
        if self._verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            self.log("Data Sample:\n{}", self._data)
            pd.options.display.max_rows = old_max_rows

        if preprocess:
            self.log("Will pre-process data")
            self._preprocess_data()

        self.get_features()
        self.get_classes()
        self.log("Feature dimensions: {}", self.features.shape)
        self.log("Classes dimensions: {}", self.classes.shape)
        self.log("Class values: {}", np.unique(self.classes))
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]
        self.log("Class distribution: {}", class_dist)
        self.log(
            "Class distribution (%): {}", (class_dist / self.classes.shape[0]) * 100
        )
        self.log("Sparse? {}", isspmatrix(self.features))

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)

        self.log("Binary? {}", self.binary)
        self.log("Balanced? {}", self.balanced)

    def scale_standard(self):
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def build_train_test_split(self, test_size=0.3):
        if (
            not self.training_x
            and not self.training_y
            and not self.testing_x
            and not self.testing_y
        ):
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features,
                self.classes,
                test_size=test_size,
                random_state=self._seed,
                stratify=self.classes,
            )

    def get_features(self, force=False):
        if self.features is None or force:
            self.log("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.log("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    def dump_test_train_val(self, test_size=0.2, random_state=123):
        ds_train_x, ds_test_x, ds_train_y, ds_test_y = ms.train_test_split(
            self.features,
            self.classes,
            test_size=test_size,
            random_state=random_state,
            stratify=self.classes,
        )
        pipe = Pipeline([("Scale", preprocessing.StandardScaler())])
        train_x = pipe.fit_transform(ds_train_x, ds_train_y)
        train_y = np.atleast_2d(ds_train_y).T
        test_x = pipe.transform(ds_test_x)
        test_y = np.atleast_2d(ds_test_y).T

        train_x, validate_x, train_y, validate_y = ms.train_test_split(
            train_x,
            train_y,
            test_size=test_size,
            random_state=random_state,
            stratify=train_y,
        )
        test_y = pd.DataFrame(np.where(test_y == 0, -1, 1))
        train_y = pd.DataFrame(np.where(train_y == 0, -1, 1))
        validate_y = pd.DataFrame(np.where(validate_y == 0, -1, 1))

        tst = pd.concat([pd.DataFrame(test_x), test_y], axis=1)
        trg = pd.concat([pd.DataFrame(train_x), train_y], axis=1)
        val = pd.concat([pd.DataFrame(validate_x), validate_y], axis=1)

        tst.to_csv(
            "data/{}_test.csv".format(self.data_name()), index=False, header=False
        )
        trg.to_csv(
            "data/{}_train.csv".format(self.data_name()), index=False, header=False
        )
        val.to_csv(
            "data/{}_validate.csv".format(self.data_name()), index=False, header=False
        )

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def data_name(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def class_column_name(self):
        pass

    @abstractmethod
    def pre_training_adjustment(self, train_features, train_classes):
        """
        Perform any adjustments to training data before training begins.
        :param train_features: The training features to adjust
        :param train_classes: The training classes to adjust
        :return: The processed data
        """
        return train_features, train_classes

    def reload_from_hdf(self, hdf_path, hdf_ds_name, preprocess=True):
        self.log("Reloading from HDF {}".format(hdf_path))
        loader = copy.deepcopy(self)

        df = pd.read_hdf(hdf_path, hdf_ds_name)
        loader.load_and_process(data=df, preprocess=preprocess)
        loader.build_train_test_split()

        return loader

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))


class Spam(DataLoader):
    def __init__(self, path="data/DATA_spam/spambase.data", verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        labels = [
            "word_freq_make",
            "word_freq_address",
            "word_freq_all",
            "word_freq_3d",
            "word_freq_our",
            "word_freq_over",
            "word_freq_remove",
            "word_freq_internet",
            "word_freq_order",
            "word_freq_mail",
            "word_freq_receive",
            "word_freq_will",
            "word_freq_people",
            "word_freq_report",
            "word_freq_addresses",
            "word_freq_free",
            "word_freq_business",
            "word_freq_email",
            "word_freq_you",
            "word_freq_credit",
            "word_freq_your",
            "word_freq_font",
            "word_freq_000",
            "word_freq_money",
            "word_freq_hp",
            "word_freq_hpl",
            "word_freq_george",
            "word_freq_650",
            "word_freq_lab",
            "word_freq_labs",
            "word_freq_telnet",
            "word_freq_857",
            "word_freq_data",
            "word_freq_415",
            "word_freq_85",
            "word_freq_technology",
            "word_freq_1999",
            "word_freq_parts",
            "word_freq_pm",
            "word_freq_direct",
            "word_freq_cs",
            "word_freq_meeting",
            "word_freq_original",
            "word_freq_project",
            "word_freq_re",
            "word_freq_edu",
            "word_freq_table",
            "word_freq_conference",
            "char_freq_;",
            "char_freq_(",
            "char_freq_[",
            "char_freq_!",
            "char_freq_$",
            "char_freq_#",
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
            "isSpam",
        ]
        emails = pd.read_csv(self._path, sep=",", header=None, names=labels)
        self._data = emails

    def class_column_name(self):
        return "spam"

    def data_name(self):
        return "spam"

    def _preprocess_data(self):
        pass

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


# class Poker(DataLoader):
#     def __init__(
#         self,
#         path="data/DATA_poker/poker-hand-training-true.data",
#         verbose=False,
#         seed=1,
#     ):
#         super().__init__(path, verbose, seed)

#     def _load_data(self):
#         labels = []
#         for i in range(1, 6):
#             labels.append("S" + str(i))
#             labels.append("C" + str(i))
#         labels.append("hand")
#         hands = pd.read_csv(self._path, sep=",", header=None, names=labels)
#         handsY = (hands["hand"] > 0).astype(int)
#         handsX = hands.drop("hand", 1)
#         hands = pd.concat([handsX, handsY], axis=1)

#         self._data = hands

#     def class_column_name(self):
#         return "poker"

#     def data_name(self):
#         return "poker"

#     def _preprocess_data(self):
#         pass
#         # for col in self._data.columns.values:
#         #     if self._data[col].dtypes == "object":
#         #         col_count = self._data[col].value_counts()
#         #         print(col_count)
#         #         # sns.barplot(col_count.index, col_count.values, alpha=0.8)
#         #         # plt.title(col)
#         #         # plt.show()

#         # # Encode labels
#         # le = LabelEncoder()
#         # for col in self._data.columns.values:
#         #     if self._data[col].dtypes == "object":
#         #         data = self._data[col]
#         #         le.fit(data.values)
#         #         self._data[col] = le.transform(self._data[col])

#     def pre_training_adjustment(self, train_features, train_classes):
#         return train_features, train_classes


class PoisonousMushrooms(DataLoader):
    def __init__(
        self, path="data/DATA_mushroom/agaricus-lepiota.data", verbose=False, seed=1
    ):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        labels = pd.read_csv("data/DATA_mushroom/agaricus.labels", header=None)
        mushrooms = pd.read_csv(self._path, sep=",", header=None, names=labels[0])
        mushroomsY = mushrooms["edible-or-poisonous"]
        mushroomsX = mushrooms.drop("edible-or-poisonous", 1)
        mushrooms = pd.concat([mushroomsX, mushroomsY], axis=1)
        self._data = mushrooms
        # self._data = pd.read_csv(self._path, sep=",", header=None)

    def class_column_name(self):
        return "edible-or-poisonous"

    def data_name(self):
        return "Poisonous Mushrooms"

    def _preprocess_data(self):
        for col in self._data.columns.values:
            if self._data[col].dtypes == "object":
                col_count = self._data[col].value_counts()
                print(col_count)
                # sns.barplot(col_count.index, col_count.values, alpha=0.8)
                # plt.title(col)
                # plt.show()

        # self._data.drop(columns=["contact", "default", "month", "day_of_week"], axis=1, inplace=True)

        # self._data.replace(" ?", pd.np.nan, inplace=True)
        # self._data.dropna(axis=0, inplace=True)
        # Encode labels
        le = LabelEncoder()
        for col in self._data.columns.values:
            if self._data[col].dtypes == "object":
                data = self._data[col]
                le.fit(data.values)
                self._data[col] = le.transform(self._data[col])

    def pre_training_adjustment(self, train_features, train_classes):
        return train_features, train_classes


if __name__ == "__main__":
    cd_data = PoisonousMushrooms(verbose=True)
    cd_data.load_and_process()
    cd_data = Spam(verbose=True)
    cd_data.load_and_process()

"""
Legacy stuff for reference
"""


# class BankMarketingData(DataLoader):
#     def __init__(
#         self, path="data/dataset1/bank-additional-full.csv", verbose=False, seed=1
#     ):
#         super().__init__(path, verbose, seed)

#     def _load_data(self):
#         self._data = pd.read_csv(self._path, sep=";")

#     def class_column_name(self):
#         return "y"

#     def data_name(self):
#         return "Bank Marketing"

#     def _preprocess_data(self):
#         for col in self._data.columns.values:
#             if self._data[col].dtypes == "object":
#                 col_count = self._data[col].value_counts()
#                 print(col_count)
#                 # sns.barplot(col_count.index, col_count.values, alpha=0.8)
#                 # plt.title(col)
#                 # plt.show()

#         self._data.drop(
#             columns=["contact", "default", "month", "day_of_week"], axis=1, inplace=True
#         )

#         self._data.replace("unknown", pd.np.nan, inplace=True)
#         self._data.dropna(axis=0, inplace=True)
#         # Encode labels
#         le = LabelEncoder()
#         for col in self._data.columns.values:
#             if self._data[col].dtypes == "object":
#                 data = self._data[col]
#                 le.fit(data.values)
#                 self._data[col] = le.transform(self._data[col])

#     def pre_training_adjustment(self, train_features, train_classes):
#         return train_features, train_classes


# class AdultSalary(DataLoader):
#     def __init__(self, path="data/dataset2/adult.data.txt", verbose=False, seed=1):
#         super().__init__(path, verbose, seed)

#     def _load_data(self):
#         self._data = pd.read_csv(self._path, sep=",", header=None)

#     def class_column_name(self):
#         return "14"

#     def data_name(self):
#         return "Adult Salary"

#     def _preprocess_data(self):
#         for col in self._data.columns.values:
#             if self._data[col].dtypes == "object":
#                 col_count = self._data[col].value_counts()
#                 print(col_count)
#                 # sns.barplot(col_count.index, col_count.values, alpha=0.8)
#                 # plt.title(col)
#                 # plt.show()

#         # self._data.drop(columns=["contact", "default", "month", "day_of_week"], axis=1, inplace=True)

#         self._data.replace(" ?", pd.np.nan, inplace=True)
#         self._data.dropna(axis=0, inplace=True)
#         # Encode labels
#         le = LabelEncoder()
#         for col in self._data.columns.values:
#             if self._data[col].dtypes == "object":
#                 data = self._data[col]
#                 le.fit(data.values)
#                 self._data[col] = le.transform(self._data[col])

#     def pre_training_adjustment(self, train_features, train_classes):
#         return train_features, train_classes
