"""
Copyright 2018 CS Systèmes d'Information

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import logging
import time

import numpy as np
# All scalers used in case SPARK=True
import pyspark.ml.feature

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import tkinter
# All scalers used in case SPARK=False
import sklearn.decomposition

# IKATS utils
from ikats.core.data.ts import TimestampedMonoVal
from ikats.core.library.exception import IkatsException, IkatsConflictError

# Spark utils
from ikats.core.library.spark import SSessionManager, SparkUtils
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client.temporal_data_mgr import DTYPE

# TODO: finir doc
"""
Principal component analysis (PCA) Algorithm: Feature reduction, using Singular 
Value Decomposition (SVD).

Must be performed on scaled values (Z-norm) for coherent results.
Orthogolnal transformation
Axis changes

PC: principal components

"""

# Define a logger for this operator
LOGGER = logging.getLogger(__name__)

# SPARK CRITERION
NB_TS_CRITERIA = 100
NB_POINTS_BY_CHUNK = 50000

# (Intermediate) Names of created columns (during spark operation only)
_INPUT_COL = "features"
_OUTPUT_COL = "PCAFeatures"


class Pca(object):
    """
    Wrapper of sklearn / pyspark.ml PCA algo. Uniform the behaviour of sklearn / pyspark.ml.
    """

    def __init__(self, n_components=None, spark=False):
        """
        Init `Pca` object.

        :param n_components: The number of principal component to keep (positive int)
        :type n_components: int

        :param spark: Use spark (case True) or not. Default False.
        :type spark: Bool
        """
        self.spark = spark

        self.n_components = n_components

        # Init self.pca
        self._get_pca()

    def _get_pca(self):
        """
        Init PCA. Note that if self.spark=True, function init a SSessionManager itself.
        """
        # CASE Spark=True (pyspark)
        # ----------------------------------
        if self.spark:
            # Init SparkSession (necessary for init Spark Scaler)
            SSessionManager.get()

            # Init pyspark.ml.feature PCA object
            self.pca = pyspark.ml.feature.PCA()

            # Additional arguments to set
            # --------------------------------
            # Set n_components
            self.pca.setK(self.n_components)
            # Set input / output columns names (necessary for Spark functions)
            self.pca.setInputCol(_INPUT_COL)
            self.pca.setOutputCol(_OUTPUT_COL)

        # Indeed: PCA requires a Vector column as an input.
        # You have to assemble your data first.

        # CASE Spark=False (sklearn)
        # -----------------------------------
        else:
            # Init sklearn.decomposition pca object
            self.pca = sklearn.decomposition.PCA()

            # Additional arguments to set
            # --------------------------------
            # Set n_components
            self.pca.n_components = self.n_components
            # Perform an inplace scaling (for performance)
            self.pca.copy = False

    def perform_pca(self, x):
        """
        Perform pca. This function replace:
            - the sklearn's `fit_transform()`
            - the pyspark's `fit().transform()`

        :param x: The data to scale,
            * shape = (N, M) if np.array
            * df containing at least column [_INPUT_COL] containing`vector` type if spark DataFrame
        :type x: np.array or pyspark.sql.DataFrame

        :return: Object x scaled with `self.pca`
        :rtype: type(x)
        """
        # CASE : Use spark = True: use pyspark
        # --------------------------------------------------------
        if self.spark:
            return self.pca.fit(x).transform(x)

        # CASE : Use spark = False: use sklearn
        # --------------------------------------------------------
        else:

            # Particular cases (align behaviour on Spark results)
            # -----------------------------------------------------
            # TODO: conformer le comportement sklearn avec celui de spark
            pass

            return self.pca.fit_transform(x)


def spark_pca(ts_list,
              fid_pattern,
              n_components,
              nb_points_by_chunk=NB_POINTS_BY_CHUNK):
    """

    :param ts_list:
    :param n_components:
    :param fid_pattern:
    :param nb_points_by_chunk:
    :return:
    """
    pass
    # TODO: assembler les colonnes en une seule, en utilisant VectorAsssembler
    # Indeed: PCA requires a Vector column as an input.
    # You have to assemble your data first.

def pca(ts_list, fid_pattern, n_components):
    """

    :param ts_list:
    :param n_components:
    :return:
    """
    pass


def pca_ts_list(ts_list,
                n_components=None,
                fid_pattern="{fid}_CP{cp_id}",
                nb_points_by_chunk=NB_POINTS_BY_CHUNK,
                nb_ts_criteria=NB_TS_CRITERIA,
                spark=None,
                ):
    """
    Wrapper: compute a PCA on a provided ts_list

    :param ts_list: List of TS to scale
    :type ts_list: List of dict
    ..Example: [ {'tsuid': tsuid1, 'funcId' funcId1}, ...]

    :param n_components: Number of principal components to keep.
    :type n_components: int

    :param fid_pattern: pattern used to name the FID of the output TSUID.
           {fid} will be replaced by the FID of the original TSUID FID
           {cp_id} will be replaced by the id of the current principal component
    :type fid_pattern: str

    :param nb_points_by_chunk: size of chunks in number of points
    (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :param nb_ts_criteria: The minimal number of TS to consider for considering Spark is
    necessary
    :type nb_ts_criteria: int


    :param spark: Flag indicating if Spark usage is:
        * forced (case True),
        * forced to be not used (case False)
        * case None: Spark usage is checked (function of amount of data)
    :type spark: bool or NoneType

    :returns:two objects:
        * ts_list : A list of dict composed by original TSUID and the information about the new TS
        * table: An array containing variance explained, and cumulative variance explained per PC (3 col)
    :rtype: Tuple composed by:
        * list of dict
        * np.array

    ..Example: result1=[{"tsuid": new_tsuid,
                        "funcId": new_fid
                        "origin": tsuid
                        }, ...]

                result2=[]
    """
    # TODO: compléter la doc: exemple de résultat

    # 0/ Check inputs
    # ----------------------------------------------------------
    # ts_list (non empty list)
    if type(ts_list) is not list:
        raise TypeError("Arg. type `ts_list` is {}, expected `list`".format(type(ts_list)))
    elif not ts_list:
        raise ValueError("`ts_list` provided is empty !")

    try:
        # Extract TSUID from ts_list
        tsuid_list = [x['tsuid'] for x in ts_list]
    except Exception:
        raise ValueError("Impossible to get tsuid list...")

    # n_components (int or None)
    if type(n_components) is not int and n_components is not None:
        raise TypeError("Arg. type `n_components` is {}, expected `int`".format(type(n_components)))
    if n_components is not None and n_components < 1:
        raise ValueError("Arg. `n_components` must be an integer > 1 (or None), get {}".format(n_components))

    # fid_pattern (str)
    if type(fid_pattern) is not str:
        raise TypeError("Arg. type `fid_pattern` is {}, expected `str`".format(type(fid_pattern)))

    # Nb points by chunk (int > 0)
    if type(nb_points_by_chunk) is not int or nb_points_by_chunk < 0:
        raise TypeError("Arg. `nb_points_by_chunk` must be an integer > 1, get {}".format(nb_points_by_chunk))

    # nb_ts_criteria (int > 0)
    if type(nb_ts_criteria) is not int or nb_ts_criteria < 0:
        raise TypeError("Arg. `nb_ts_criteria` must be an integer > 1, get {}".format(nb_ts_criteria))

    # spark (bool or None)
    if type(spark) is not bool and spark is not None:
        raise TypeError("Arg. type `spark` is {}, expected `bool` or `NoneType`".format(type(spark)))

    # 1/ Check for spark usage and run
    # ----------------------------------------------------------
    if spark is True or (spark is None and SparkUtils.check_spark_usage(tsuid_list=tsuid_list,
                                                                        nb_ts_criteria=nb_ts_criteria,
                                                                        nb_points_by_chunk=nb_points_by_chunk)):
        # Arg `spark=True`: spark usage forced
        # Arg `spark=None`: Check using criteria (nb_points and number of ts)
        return spark_pca(ts_list=tsuid_list, fid_pattern=fid_pattern, n_components=n_components, nb_points_by_chunk=nb_points_by_chunk)
    else:
        return pca(ts_list=tsuid_list, fid_pattern=fid_pattern, n_components=n_components)
