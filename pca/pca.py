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
from collections import defaultdict
import numpy as np
# All scalers used in case SPARK=True
import pyspark.ml.feature

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

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

# Other arguments
SEED = 0
# Set the seed: making results reproducible
np.random.seed(SEED)


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
            # Seed
            self.pca.random_state = SEED

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

            # Perform `fit_transform` if copy=False
            return self.pca.fit_transform(x)


def spark_pca(ts_list,
              fid_pattern,
              n_components,
              table_name,
              nb_points_by_chunk=NB_POINTS_BY_CHUNK):
    """
    Compute a PCA on a provided ts list ("spark" mode).

    :param ts_list: List of TS to scale
    :type ts_list: List of dict
    ..Example: [ {'tsuid': tsuid1, 'funcId' funcId1}, ...]

    :param fid_pattern: pattern used to name the FID of the output TSUID.
           {pc_id} will be replaced by the id of the current principal component
    :type fid_pattern: str

    :param n_components: Number of principal components to keep.
    :type n_components: NoneType or int

    :param table_name: Name of the created table (containing explained variance)
    :type table_name: str

    :param nb_points_by_chunk: size of chunks in number of points
    (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :return:
    """
    pass
    # TODO: assembler les colonnes en une seule, en utilisant VectorAsssembler
    # Indeed: PCA requires a Vector column as an input.
    # You have to assemble your data first.


def pca(ts_list, fid_pattern, n_components, table_name):
    """
    Compute a PCA on a provided ts list ("no spark" mode).

    :param ts_list: List of TS to scale
    :type ts_list: List of dict
    ..Example: [ {'tsuid': tsuid1, 'funcId' funcId1}, ...]

    :param fid_pattern: pattern used to name the FID of the output TSUID.
           {pc_id} will be replaced by the id of the current principal component
    :type fid_pattern: str

    :param n_components: Number of principal components to keep.
    :type n_components: NoneType or int

    :param table_name: Name of the created table (containing explained variance)
    :type table_name: str

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

    # 0/ Init Pca object
    # ------------------------------------------------
    # Init object
    current_pca = Pca(n_components=n_components, spark=False)

    # 1/ Build input
    # ------------------------------------------------
    # Need to build an array containing data only (the timestamps are NOT transformed by PCA)
    # -> data must be aligned

    start_loading_time = time.time()

    # Get timestamps from THE FIRST TS
    timestamps = IkatsApi.ts.read(ts_list[0])[0][:, 0]

    # Read data from each TS (data only, NO timestamps -> `[:, 1]`)
    # Transpose data for correct format: nrow=n_times, n_col=n_TS
    ts_data = np.array([IkatsApi.ts.read(ts)[0][:, 1] for ts in ts_list]).T
    # Shape = (n_times, n_ts)

    LOGGER.debug("Gathering time: %.3f seconds", time.time() - start_loading_time)

    # 2/ Perform PCA
    # ------------------------------------------------
    start_computing_time = time.time()

    # Perform PCA with NO spark mode (here, using the `fit_transform` sklearn operation)
    transformed_data = current_pca.perform_pca(x=ts_data)
    # Result: The transformed dataset: shape = (n_times, n_components)

    LOGGER.debug("Computing time: %.3f seconds", time.time() - start_computing_time)

    # 3/ Save transformed TS list
    # ------------------------------------------------
    def __save_ts(data, original_tsuid, PCId):
        """
        Build one resulting TS and save it.
        :param data: One column of transformed result (np.array)
        :param original_tsuid: The corresponding tsuid of the original TS (str)
        :param PCId: The Id of the current principal component (PC)

        :return: Dict containing:
            * resulted tsuid ("tsuid")
            * resulted funcId ("funcId")
            * original tsuid ("origin")
        """
        # Add timestamp to the data column, and store it into custom object
        result = np.array([timestamps, data]).T
        # Shape = (n_times, 2)

        # 1/ Generate new FID
        # -------------------------------
        # Add id of current PC (in 1..k), `fid_pattern` contains str '{pc_id}'
        new_fid = fid_pattern.format(PCId)
        # Example: "PORTFOLIO_pc1"
        try:
            IkatsApi.ts.create_ref(new_fid)
        # Exception: if fid already exist
        except IkatsConflictError:
            # TS already exist, append timestamp to be unique
            new_fid = '%s_%s' % (new_fid, int(time.time() * 1000))
            IkatsApi.ts.create_ref(new_fid)

        # 2/ Import time series result in database
        # -------------------------------
        try:
            res_import = IkatsApi.ts.create(fid=new_fid,
                                            data=result,
                                            generate_metadata=True,
                                            parent=original_tsuid,
                                            sparkified=False)
            new_tsuid = res_import['tsuid']

            # Inherit from parent
            IkatsApi.ts.inherit(new_tsuid, original_tsuid)

        except Exception:
            raise IkatsException("save scale failed")

        LOGGER.debug("TSUID: %s(%s), saved", new_fid, new_tsuid)

        return {"tsuid": new_tsuid, "funcId": new_fid, "origin": original_tsuid}

    # Save the result
    start_saving_time = time.time()

    result1 = [__save_ts(data=transformed_data[:, i],
                         original_tsuid=ts_list[i],
                         PCId=i+1  # pc from 1 to n_components + 1 (more readable for user)
                         ) for i in transformed_data.shape[1]]

    LOGGER.debug("Result import time: %.3f seconds", time.time() - start_saving_time)

    # 4/ Explained variance result (table)
    # ------------------------------------------------
    # Function should return table containing:
    # * List of PC ('PC1', ... 'PCk'), where k = n_components + 1
    # * Explained variance per PC
    explained_var = current_pca.pca.explained_variance_ratio_

    # Get the cumulative explained variance
    cum_explained_var = np.cumsum(explained_var)

    # Create row names
    pc_list = ["PC_" + str(i) for i in range(1, n_components + 1)]

    # Create resulting table in IKATS format
    result2 = _format_table(matrix=np.array([explained_var, cum_explained_var]).T,
                            colnames=['explained variance','cumulative explained variance'],
                            rownames=pc_list,
                            table_name=table_name)

    # Build result (table containing : PC id, explained_var, cum_explained_var)
    IkatsApi.table.create(result2)

    # 5/ Build final result
    # ------------------------------------------------
    return result1, result2


def pca_ts_list(ts_list,
                n_components=None,
                fid_pattern="PC{pc_id}",
                table_name="Variance explained PCA",
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
           {pc_id} will be replaced by the id of the current principal component
    :type fid_pattern: str

    :param table_name: Name of the created table (containing explained variance)
    :type table_name: str

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

    # n_components (positive int or None)
    if type(n_components) is not int and n_components is not None:
        raise TypeError("Arg. type `n_components` is {}, expected `int`".format(type(n_components)))
    if n_components is not None and n_components < 1:
        raise ValueError("Arg. `n_components` must be an integer > 1 (or None), get {}".format(n_components))

    # fid_pattern (str)
    if type(fid_pattern) is not str:
        raise TypeError("Arg. type `fid_pattern` is {}, expected `str`".format(type(fid_pattern)))
    if '{pc_id}' not in fid_pattern:
        # Add `_PC{pc_id}` into `fid_pattern` (at the end)
        fid_pattern += "_PC{pc_id}"
        LOGGER.debug("Add '_PC{pc_id}' to arg. `fid_pattern`")

    # Table name (str)
    if type(table_name) is not str:
        raise TypeError("Arg. `table_name` is {}, expected `str`".format(type(table_name)))

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
        return spark_pca(ts_list=tsuid_list, fid_pattern=fid_pattern, n_components=n_components,table_name=table_name, nb_points_by_chunk=nb_points_by_chunk)
    else:
        return pca(ts_list=tsuid_list, fid_pattern=fid_pattern, n_components=n_components, table_name=table_name)


def _format_table(matrix, colnames, rownames, table_name):
    """
    Fill an ikats table structure with table containing explained variance.

    ..Example of input: colnames: [CPId, explained variance, cumulative explained variance]

    :param matrix: array containing values to store
    :type matrix: numpy array

    :param colnames: The name of all column used (take care to the order)
    :type colnames: list of str

    :param rownames: The name of all rows used (take care to the order)
    :type rownames: list of str

    :param : table_name: Name of the table to save
    :type : table_name: str

    :return: dict formatted as awaited by functional type table
    :rtype: dict

    ..Example:
    {'content': {
        'cells': [['0.7','0.7'],
                  ['0.3', '1.0']]
        },
    'headers': {
        'col': {'data': ['Var','explained variance','cumulative explained variance']},
        'row': {'data': ['PC', 'PC1', 'PC2']}
        },
    'table_desc': {
        'desc': 'Explained variance from PCA operator',
        'name': 'truc',
        'title': 'Variance explained'
        }
    }


    Resulting viz:
    |PC \ Var|explained variance|cumulative explained variance|
    |--------|------------------|-----------------------------|
    |PC1     | 0.7              | 0.7                         |
    |PC2     | 0.3              | 1.0                         |
    """

    # Initializing table structure
    table = defaultdict(dict)

    # Filling title
    table['table_desc']['title'] = 'Variance explained'
    table['table_desc']['name'] = table_name
    table['table_desc']['desc'] = "Explained variance from PCA operator"

    # Filling headers columns
    table['headers']['col'] = dict()
    table['headers']['col']['data'] = ["Var"]
    # Fill column names (ex: 'explained variance','cumulative explained variance')
    table['headers']['col']['data'].extend(colnames)

    # Filling headers rows
    table['headers']['row'] = dict()
    table['headers']['row']['data'] = ["PC"]
    # Fill row names (ex: 'PC1', 'PC2')
    table['headers']['row']['data'].extend(rownames)

    # Filling cells content
    table['content']['cells'] = matrix.tolist()

    return table
