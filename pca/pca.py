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
from pyspark.ml.feature import PCA as spark_pca

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# All scalers used in case SPARK=False
from sklearn.decomposition import PCA as sk_pca

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

Orthogolnal transformation
Axis changes

PC: principal components

"""

# Define a logger for this operator
LOGGER = logging.getLogger(__name__)

# Dict containing list of scaler used in operator.
# Structure of the result: dict
# {'no_spark': sklearn.preprocessing scaler,
# 'spark': pyspark.ml.feature scaler}
# TODO: changer, ça n'est pas adapté
# These objects are not init (for 'spark' scalers, need to init a Spark Context else, raise error).
PCA_DICT = {
    "PCA": {'no_spark': sk_pca,
            'spark': spark_pca},
    # 'no_spark': need to set `copy=False`,
}

# Example: To init sklearn (-> no spark) :
# PCA_DICT["PCA"]['no_spark']()

# SPARK CRITERION
NB_TS_CRITERIA = 100
NB_POINTS_BY_CHUNK = 50000


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

    :param ts_list: List of TS to scale
    :type ts_list: List of dict
    ..Example: [ {'tsuid': tsuid1, 'funcId' funcId1}, ...]

    :param n_components:
    :param fid_pattern:

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
