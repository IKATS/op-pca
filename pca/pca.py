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
import re

# Modules used in case SPARK=False
import sklearn.decomposition

# Modules used in case SPARK=True
import pyspark.ml.feature
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, DoubleType

# Spark utils
from ikats.core.library.spark import SSessionManager, SparkUtils
from ikats.core.resource.api import IkatsApi

# IKATS utils
from ikats.core.library.exception import IkatsException, IkatsConflictError, IkatsNotFoundError
from ikats.core.resource.client.temporal_data_mgr import DTYPE

"""
Principal component analysis (PCA) Algorithm: Feature reduction, using Singular Value Decomposition (SVD).

Technical details: Build pertinent orthogonal axis from a provided data-set: the Principal Components (PC).
It's a dimensional reduction operation.

Purpose: 
    * Clean a data-set (reduce noise)
    * Better representation of the data (each PC represents a particular phenomenon)

Remark:
    * Should be performed on scaled values (Z-norm) for coherent results.

Keywords:
    * PCA: Principal Component Analysis
    * PC: principal components: the new "Axis" created from initial data-set
    * variance explained: "pertinence" (in %) of each PC (the first PC have a big % variance explained).
"""

# GLOBAL VARS
# -------------------------------------------------
# Define a logger for this operator
LOGGER = logging.getLogger(__name__)

# SPARK CRITERION
NB_TS_CRITERIA = 100
NB_POINTS_BY_CHUNK = 50000

# (Intermediate) Names of created columns (during spark operation only)
_INPUT_COL = "features"
_OUTPUT_COL = "PCA_Features"


class Pca(object):
    """
    Wrapper of sklearn / pyspark.ml PCA algo. Uniform the behaviour of sklearn / pyspark.ml.
    """

    def __init__(self, n_components=None, spark=False, seed=None):
        """
        Init `Pca` object.

        :param n_components: The number of principal component to keep (positive int)
        :type n_components: int

        :param spark: Use spark (case True) or not. Default False.
        :type spark: Bool

        :param seed: The seed used for SKLEARN init only. If None, the seed is generated inside sklearn implementation,
        with `np.random`. Default None.
        :type seed: int or NoneType
        """
        # TODO: reformater la classe pour mettre dedant toutes les fonctions
        # Store arguments
        self.spark = spark

        self.n_components = n_components

        self.seed = seed

        if self.seed is not None:
            # Set the seed: making results reproducible
            np.random.seed(self.seed)

        # Init `variance_explained`, should contain the list of explained variance (in %) per PC
        self.variance_explained = None

        # Store model (from pyspark or sklearn)
        self.model = None

        # Init self.pca
        self.pca = self._init_pca()

    def _init_pca(self):
        """
        Init PCA. Note that if self.spark=True, function init a SSessionManager itself.
        """
        # CASE Spark=True (pyspark)
        # ----------------------------------
        if self.spark:
            # Init SparkSession (necessary for init Spark PCA)
            SSessionManager.get()

            # Init pyspark.ml.feature PCA object
            pca = pyspark.ml.feature.PCA()

            # Additional arguments to set
            # --------------------------------
            # Set n_components
            pca.setK(self.n_components)
            # Set input / output columns names (necessary for Spark functions)
            pca.setInputCol(_INPUT_COL)
            pca.setOutputCol(_OUTPUT_COL)

        # CASE Spark=False (sklearn)
        # -----------------------------------
        else:
            # Init sklearn.decomposition pca object
            pca = sklearn.decomposition.PCA()

            # Additional arguments to set
            # --------------------------------
            # Set n_components
            pca.n_components = self.n_components
            # Copy result for storing model
            pca.copy = True
            # Seed (default=None in sklearn.decomposition.PCA() )
            pca.random_state = self.seed

        return pca

    def perform_pca(self, x):
        """
        Perform pca. This function replaces:
            - the sklearn's `fit_transform()`
            - the pyspark's `fit().transform()`

        Make an uniform behaviour between sklearn and spark.

        :param x: The data to scale,
            * shape = (N, M) if np.array
            * df containing at least column [_INPUT_COL] containif sng`vector` type ipark DataFrame
        :type x: np.array or pyspark.sql.DataFrame

        :return: Object x scaled with `self.pca`
        :rtype: type(x)

        ..Note: This function set `self.variance_explained` containing % variance explained per PC.
        """
        # CASE : Use spark = True: use pyspark
        # --------------------------------------------------------
        if self.spark:
            # Store model (for extracting variance explained)
            self.model = self.pca.fit(x)

            # Store variance explained
            self.variance_explained = np.array(self.model.explainedVariance)
            # np.array of len (`n_component`)

            return self.model.transform(x)

        # CASE : Use spark = False: use sklearn
        # --------------------------------------------------------
        else:

            # Perform PCA
            # -----------------------------------------------------
            # Perform `fit` for storing the model (arg copy should be set to True)
            self.model = self.pca.fit(x)

            # Perform PCA
            result = self.pca.transform(x)

            # Perform variance explained
            self.variance_explained = self.pca.explained_variance_ratio_

            return result

    def table_variance_explained(self, table_name):
        """
        Create the table of variance explained.

        :param table_name: Name of the created table (containing explained variance)
        :type table_name: str

        :return: Table containing
            * List of PC ('PC1', ... 'PCk'), where k = n_components + 1 -> row names
            * Explained variance per PC -> column 1
            * Cumulative explained variance per PC -> column 2
        :rtype: dict
        """
        # 1/ Get raw data
        # -----------------------------------------------------
        # Get the explained variance per PC (in %)
        explained_var = self.variance_explained

        # Get the cumulative explained variance
        cum_explained_var = np.cumsum(explained_var)

        # Create row names ('PC1', 'PC2', ...)
        pc_list = ["PC_" + str(i) for i in range(1, self.n_components + 1)]

        # 2/ Create resulting table in IKATS format
        # -----------------------------------------------------
        # table containing : PC id, explained_var, cum_explained_var
        result2 = _format_table(matrix=np.array([explained_var, cum_explained_var]).T,
                                rownames=pc_list,
                                table_name=table_name)

        # Save the table (table name have been tested yet: is UNIQUE)
        IkatsApi.table.create(data=result2)


def _check_alignement(tsuid_list):
    """
    Check the alignment of the provided list of TS (`tsuid_list`): same `start_date`, `end_date`, and `period`.
    Operator `quality_stat` shall be launch on ts_list before !

    :param tsuid_list: List of tsuid of TS to check
    :type tsuid_list: List of str
    ..Example: ['tsuid1', 'tsuid2', ...]

    return: Tuple composed by:
        * start date (int)
        * end date (int)
        * period (`qual_ref_period`, or calculation of period) (int)

    :raises:
        * ValueError: TS are not aligned
        * ValueError: Some metadata are missing (start date, end date, nb points)

    ..Note: First TS is the reference. Indeed, ALL TS must be aligned (so aligned to the first TS)
    """
    # Read metadata
    meta_list = IkatsApi.md.read(tsuid_list)

    # Perform operation iteratively on each TS
    for tsuid in tsuid_list:

        # 1/ Retrieve meta data and check available meta-data
        # --------------------------------------------------------------------------
        md = meta_list[tsuid]

        # CASE 1: no md (sd, ed, nb_point) -> raise ValueError
        if 'ikats_start_date' not in md.keys() and 'ikats_end_date' not in md.keys():
            raise ValueError("No MetaData (start / end date) associated with tsuid {}... Is it an existing TS ?".format(tsuid))
        # CASE 2: metadata `period` not available -> raise ValueError
        elif 'qual_ref_period' not in md.keys():
            raise ValueError("No MetaData `qual_ref_period` with tsuid {}... Please launch `quality indicator`".format(tsuid))
        # CASE 3: OK (metadata `period` available...) -> continue
        else:
            period = int(float(md['qual_ref_period']))
            sd = int(md['ikats_start_date'])
            ed = int(md['ikats_end_date'])

        # 2/ Check if data are aligned (same sd, ed, period)
        # --------------------------------------------------------------------------
        # CASE 1: First TS -> get as reference
        if tsuid == tsuid_list[0]:
            ref_sd = sd
            ref_ed = ed
            ref_period = period

        # CASE 2: Other TS -> compared to the reference
        else:
            # Compare `sd`
            if sd != ref_sd:
                raise ValueError("TS {}, metadata `start_date` is {}:"
                                 " not aligned with other TS (expected {})".format(tsuid, sd, ref_sd))
            # Compare `ed`
            elif ed != ref_ed:
                raise ValueError("TS {}, metadata `end_date` is {}:"
                                 " not aligned with other TS (expected {})".format(tsuid, ed, ref_ed))
            # Compare `period`
            elif period != ref_period:
                raise ValueError("TS {}, metadata `ref_period` is {}:"
                                 " not aligned with other TS (expected {})".format(tsuid, ed, ref_period))

    return ref_sd, ref_ed, ref_period


def spark_pca(tsuid_list,
              fid_pattern,
              n_components,
              table_name,
              nb_points_by_chunk=NB_POINTS_BY_CHUNK):
    """
    Compute a PCA on a provided ts list ("spark" mode).

    :param tsuid_list: List of tsuid of TS to scale
    :type tsuid_list: List of str
    ..Example: ['tsuid1', 'tsuid2', ...]

    :param fid_pattern: pattern used to name the FID of the output TSUID.
           {pc_id} will be replaced by the id of the current principal component
    :type fid_pattern: str

    :param n_components: Number of principal components to keep.
    :type n_components: int

    :param table_name: Name of the created table (containing explained variance)
    :type table_name: str

    :param nb_points_by_chunk: size of chunks in number of points
    (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :returns:two objects:
        * tsuid_list : A list of dict composed by original TSUID and the information about the new TS
        * Name of the ikats table containing variance explained, and cumulative variance explained per PC (3 col)
    :rtype: Tuple composed by:
        * list of dict
        * str

    :raises:
        * ValueError: If one of the input TS have no metadata start date, or end date, or nb_point
    """
    # 0/ Init Pca object
    # ------------------------------------------------
    # Init object
    current_pca = Pca(n_components=n_components, spark=True)

    # 1/ Test alignment of the data
    # ------------------------------------------------
    ref_sd, ref_ed, ref_period = _check_alignement(tsuid_list)

    try:
        # 2/ Get data
        # ------------------------------------------------
        # DESCRIPTION: Import data into dataframe ([chunk_index, "Timestamp", `_INPUT_COL`])
        # INPUT  : tsuid list
        # OUTPUT : A DataFrame with result columns ["index", "Timestamp", `_INPUT_COL`]: int, int,  Vector
        _, df = SSessionManager().get_tslist_in_single_col(tsuid_list=tsuid_list,
                                                           sd=ref_sd, ed=ref_ed, period=ref_period,
                                                           nb_points_by_chunk=nb_points_by_chunk,
                                                           value_colname=_INPUT_COL)
        # ..Example : df =
        #  +-----+-------------+----------------+
        #  |index|Timestamp    |`_INPUT_COL` |
        #  +-----+-------------+-------------+
        #  |0    |1449755761000|[0.08,0.07]  |
        #  ...

        # DESCRIPTION: Select useful columns ("Timestamp", `_INPUT_COL`)
        # INPUT  : DataFrame ["index", "Timestamp", `_INPUT_COL`]: int, int, Vector
        # OUTPUT : Same DataFrame with columns ["Timestamp", `_INPUT_COL`]: int, Vector
        df = df.select(["Timestamp", _INPUT_COL])

        # 3/ Calculate PCA
        # -------------------------------
        # Input of PCA algo should be:
        # N times, M TS
        # +----------------------------------+
        # |InputFeatures                     |
        # +----------------------------------+
        # |[value_1_TS_1, ..., value_1_TS_M] |
        # | ...                              |
        # |[value_N_TS_1, ..., value_N_TS_M] |
        # +----------------------------------+

        # DESCRIPTION : Perform PCA
        # INPUT  : A DataFrame with columns ["Timestamp", _INPUT_COL]: int, Vector
        # OUTPUT : A DataFrame with result columns ["Timestamp", _INPUT_COL, _OUTPUT_COL]: int, Vector, Vector
        # Where _INPUT_COL is Vector of len(N_TS), and _OUTPUT_COL vector of len (`n_component`)
        pca_result = current_pca.perform_pca(df)
        # Example:
        # +-----------+-------------+--------------------+
        # |  Timestamp| `_INPUT_COL`|       `_OUTPUT_COL`|
        # +-----------+-------------+--------------------+
        # |14879030000|[1.0,1.0]    |[-1.4142135623730...|
        # ...
        # DataFrame[Timestamp: bigint, `_INPUT_COL`: vector, `_OUTPUT_COL`: vector]

        # 4/ Format result
        # ------------------------------------------------
        # A function that transform `vector` into `List` of double
        vector_to_list = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

        # DESCRIPTION : Transorm column containing result (type Vector) into multiple columns
        # INPUT  : A DataFrame with result columns ["Timestamp", _INPUT_COL, _OUTPUT_COL]: int, Vector, Vector
        # OUTPUT : Same DF with muliple columns (one per PC):  ["Timestamp", _INPUT_COL, _OUTPUT_COL, PC1, ..., PC{k}]
        pca_result = (pca_result
                      .withColumn("PC", vector_to_list(col(_OUTPUT_COL)))  # Vector to list
                      .select(["Timestamp"] + [col("PC")[i] for i in range(n_components)]))  # Split col containing list

        # Example
        # +-----------+------+-------+
        # |  Timestamp| PC[0]| PC[1] |
        # +-----------+------+-------+
        # |14879030000|-1.4  | -2    |
        # ...
        # DataFrame[Timestamp: bigint, PC[0]: double, PC[1]: double]

        # DESCRIPTION : Rename columns ("PC[1]" -> "1")
        # INPUT  : DF with muliple columns (one per PC):  ["Timestamp", "PC[0]", ..., "PC[`n_component`-1]"]
        # OUTPUT : Same DF with columns renamed:  ["Timestamp", "1", ..., "`n_component`"]
        # More readable for user (start from 1)
        new_colnames = ['Timestamp'] + ["{}".format(i) for i in range(1, n_components + 1)]
        pca_result = pca_result.toDF(*new_colnames)
        # Example
        # +-----------+------+-------+
        # |  Timestamp|    1 |    2  |
        # +-----------+------+-------+
        # |14879030000|-1.4  | -2    |
        # ...

        # 5/ Get the table of variance explained
        # ------------------------------------------------
        current_pca.table_variance_explained(table_name=table_name)

        # 6/ Store the pyspark's model
        # ------------------------------------------------
        model = current_pca.model

        # 7/ Save result
        # ------------------------------------------------
        fid_pattern = fid_pattern.replace("{pc_id}", "{}")  # remove `pc_id` from `fid_pattern`

        def __save_df(data_frame, fid_pattern=fid_pattern):
            """
            Save Each column of input `data_frame`, with associated timestamp (one of the col)
            :param data_frame: The data frame to save. Must contain col:
                * Timestamp
                * data 1
                * ... (other col to save)
            :type data_frame: pyspark.sql.dataframe.DataFrame

            :param fid_pattern: pattern used to name the FID of the output TSUID.
            :type fid_pattern: str

            :return result: The TS list created from input `data_frame` (save din Ikats).
            :rtype result: list of dict

            ..Example: result =
            [
                { "tsuid": "AB7CBD00000100000100000200000200000300000F", "funcId": "PC_1"}
                ...
            ]
            """
            # 7.1/ INIT
            # ------------------------------
            # Init result, list of dict
            result = []

            # List of columns to save
            list_col = data_frame.columns
            # The Timestamp is not DATA to save
            list_col.remove('Timestamp')
            # Ex: ["1", "2"]

            # 7.2/ Save each PC (column)
            # ------------------------------
            # TODO: Optimiser pour éviter une boucle
            for pc in list_col:
                # 7.3/ Generate new FID
                # -------------------------------
                # Add id of current PC (in 1..k), `fid_pattern` contains str '{}'
                current_fid = fid_pattern.format(pc)
                # Example: "PORTFOLIO_pc1"
                IkatsApi.ts.create_ref(current_fid)

                # Note that FID should be already checked !!

                # 7.4/ Save Data
                # -------------------------------
                # OPERATION: Import result by partition into database, and collect
                # INPUT: [Timestamp, 1] (col "1": correspond to first PC)
                # OUTPUT: the new tsuid of the scaled ts (not used)
                # TODO: peut être mapper par index, utiliser des opérations par ligne
                pca_result. \
                    select(['Timestamp', pc]). \
                    rdd. \
                    mapPartitions(lambda x: SparkUtils.save_data(fid=current_fid, data=list(x))) \
                    .collect()

                # 7.5/ Retrieve tsuid of the saved TS
                # -------------------------------
                new_tsuid = IkatsApi.fid.tsuid(current_fid)
                LOGGER.debug("TSUID: %s(%s), Result import time: %.3f seconds", current_fid, new_tsuid,
                             time.time() - start_saving_time)

                # 7.6/ store metadata ikats_start_date, ikats_end_date and qual_nb_points
                # ----------------------------------------------------------------------------
                if not IkatsApi.md.create(
                        tsuid=new_tsuid,
                        name='ikats_start_date',
                        value=ref_sd,
                        data_type=DTYPE.date,
                        force_update=True):
                    LOGGER.error("Metadata ikats_start_date couldn't be saved for TS %s", new_tsuid)

                if not IkatsApi.md.create(
                        tsuid=new_tsuid,
                        name='ikats_end_date',
                        value=ref_ed,
                        data_type=DTYPE.date,
                        force_update=True):
                    LOGGER.error("Metadata ikats_end_date couldn't be saved for TS %s", new_tsuid)

                if not IkatsApi.md.create(
                        tsuid=new_tsuid,
                        name='qual_ref_period',
                        value=ref_period,
                        data_type=DTYPE.number,
                        force_update=True):
                    LOGGER.error("Metadata qual_ref_period couldn't be saved for TS %s", new_tsuid)

                # Retrieve imported number of points from database
                qual_nb_points = IkatsApi.ts.nb_points(tsuid=new_tsuid)
                if not IkatsApi.md.create(
                        tsuid=new_tsuid,
                        name='qual_nb_points',
                        value=qual_nb_points,
                        data_type=DTYPE.number,
                        force_update=True):
                    LOGGER.error("Metadata qual_nb_points couldn't be saved for TS %s", new_tsuid)

                # 6/ Update results
                # -------------------------------
                result.append({
                    "tsuid": new_tsuid,
                    "funcId": current_fid
                })

            return result

        # Save the result
        start_saving_time = time.time()

        # Perform
        result_ts = __save_df(data_frame=pca_result, fid_pattern=fid_pattern)

        LOGGER.debug("Result import time: %.3f seconds", time.time() - start_saving_time)

        # For now, no model is outputed
        return result_ts, table_name  #, model

    except Exception:
        raise
    finally:
        SSessionManager.stop()


def pca(tsuid_list, fid_pattern, n_components, table_name):
    """
    Compute a PCA on a provided ts list ("no spark" mode).

    :param tsuid_list: List of tsuid of TS to scale
    :type tsuid_list: List of str
    ..Example: ['tsuid1', 'tsuid2', ...]

    :param fid_pattern: pattern used to name the FID of the output TSUID.
           {pc_id} will be replaced by the id of the current principal component
    :type fid_pattern: str

    :param n_components: Number of principal components to keep.
    :type n_components: int

    :param table_name: Name of the created table (containing explained variance)
    :type table_name: str

    :returns:two objects:
        * tsuid_list : A list of dict composed by original TSUID and the information about the new TS
        * Name of the ikats table containing variance explained, and cumulative variance explained per PC (3 col)
    :rtype: Tuple composed by:
        * list of dict
        * str
    """

    # 0/ Init Pca object
    # ------------------------------------------------
    # Init object
    current_pca = Pca(n_components=n_components, spark=False)

    # 1/ Test alignment of the data
    # ------------------------------------------------
    _, _, _ = _check_alignement(tsuid_list)

    # 2/ Build input
    # ------------------------------------------------
    # Need to build an array containing data only (the timestamps are NOT transformed by PCA)
    # -> data must be aligned

    start_loading_time = time.time()

    # Get timestamps from THE FIRST TS
    timestamps = IkatsApi.ts.read(tsuid_list[0])[0][:, 0]

    # Read data from each TS (data only, NO timestamps -> `[:, 1]`)
    # Transpose data for correct format: nrow=n_times, n_col=n_TS
    ts_data = np.array([IkatsApi.ts.read(ts)[0][:, 1] for ts in tsuid_list]).T
    # Shape = (n_times, n_ts)

    LOGGER.debug("Gathering time: %.3f seconds", time.time() - start_loading_time)

    # 3/ Perform PCA
    # ------------------------------------------------
    start_computing_time = time.time()

    # Perform PCA with NO spark mode (here, using the `fit_transform` sklearn operation)
    transformed_data = current_pca.perform_pca(x=ts_data)
    # Result: The transformed dataset: shape = (n_times, n_components)

    LOGGER.debug("Computing time: %.3f seconds", time.time() - start_computing_time)

    # 4/ Save transformed TS list
    # ------------------------------------------------
    fid_pattern = fid_pattern.replace("{pc_id}", "{}")  # remove `pc_id` from `fid_pattern`

    def __save_ts(data, PCId):
        """
        Build one resulting TS and save it.
        :param data: One column of transformed result (np.array)
        :param PCId: The Id of the current principal component (PC)

        :return: Dict containing:
            * resulted tsuid ("tsuid")
            * resulted funcId ("funcId")
            * original tsuid ("origin")
        """
        # Add timestamp to the data column, and store it into custom object
        result = np.array([timestamps, data]).T
        # Shape = (n_times, 2)

        # 4.1/ Generate new FID
        # -------------------------------
        # Add id of current PC (in 1..k), `fid_pattern` contains str '{}'
        new_fid = fid_pattern.format(PCId)
        # Example: "PORTFOLIO_pc1"
        try:
            IkatsApi.ts.create_ref(new_fid)
        # Exception: if fid already exist
        except IkatsConflictError:
            # TS already exist, append timestamp to be unique
            new_fid = '%s_%s' % (new_fid, int(time.time() * 1000))
            IkatsApi.ts.create_ref(new_fid)

        # 4.2/ Import time series result in database
        # -------------------------------
        try:
            res_import = IkatsApi.ts.create(fid=new_fid,
                                            data=result,
                                            generate_metadata=True,
                                            parent=None,
                                            sparkified=False)
            new_tsuid = res_import['tsuid']

        except Exception:
            raise IkatsException("save scale failed")

        LOGGER.debug("TSUID: %s(%s), saved", new_fid, new_tsuid)

        return {"tsuid": new_tsuid, "funcId": new_fid}

    # Save the result
    start_saving_time = time.time()

    result1 = [__save_ts(data=transformed_data[:, i],
                         PCId=i + 1  # pc from 1 to n_components + 1 (more readable for user)
                         ) for i in range(transformed_data.shape[1])]

    LOGGER.debug("Result import time: %.3f seconds", time.time() - start_saving_time)

    # 5/ Explained variance result (table)
    # ------------------------------------------------
    current_pca.table_variance_explained(table_name=table_name)

    # 6/ Store model from sklearn
    # ------------------------------------------------
    model = current_pca.model

    # 6/ Build final result
    # ------------------------------------------------
    # For now, no model is outputed
    return result1, table_name  #, model


def pca_ts_list(ts_list,
                n_components,
                table_name,
                fid_pattern="PC{pc_id}",
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

    :param table_name: Name of the created table (containing explained variance).
    :type table_name: str

    ..Note: At the beginning of the algo, test if `table_name` already exist.

    :param fid_pattern: pattern used to name the FID of the output TSUID.
           {pc_id} will be replaced by the id of the current principal component
    :type fid_pattern: str

    :param nb_points_by_chunk: size of chunks in number of points
    (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :param nb_ts_criteria: The minimal number of TS for considering Spark is necessary or not
    :type nb_ts_criteria: int

    :param spark: Flag indicating if Spark usage is:
        * forced (case True),
        * forced to be not used (case False)
        * case None: Spark usage is checked (depending of the amount of data to process)
    :type spark: bool or NoneType

    :returns:two objects:
        * tsuid_list : A list of dict composed by resulted TSUID and the information about the new TS
        * Name of the ikats table containing variance explained, and cumulative variance explained per PC (3 col)
    :rtype: Tuple composed by:
        * list of dict
        * str

    ..Example:
    result1=[{"tsuid": new_tsuid,
                        "funcId": new_fid
                        }, ...]


    result2 = "Variance explained PCA"
    """

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

    # n_components (positive int)
    if type(n_components) is not int:
        raise TypeError("Arg. type `n_components` is {}, expected `int`".format(type(n_components)))
    if n_components < 1:
        raise ValueError("Arg. `n_components` must be an integer > 1 (or None), get {}".format(n_components))
    if n_components >= len(tsuid_list):
        raise ValueError("Arg. `n_components` must be an integer <= n_ts, get {}".format(n_components))

    # CHECK IF requested `fid_pattern` will not produce aready created fid
    def __check_fid(pattern, n):
        """
        Check if `fid_pattern` correspond to an existing TS.

        :param pattern: pattern used to name the FID of the output TSUID.
           {pc_id} will be replaced by the id of the current principal component
        :type pattern: str

        :param n: The number of PC (`n_component`)
        :type n: int

        :raise ValueError: if an fid corresponding to `fid_pattern` already exists
        """
        pattern = pattern.replace("{pc_id}", "{}")
        # Example: [`PC_1`, `PC_2`]
        fid_pattern_list = [pattern.format(pc_id) for pc_id in range(1, n + 1)]

        # Check if Fid already exist... -> raise ValueError
        try:
            # Find all TS corresponding to pattern requested
            # (at least one of the pattern must match to not raise an error)
            result = IkatsApi.fid.find("funcIds", fid_pattern_list)
            # Example: [{'funcId': 'PC1', 'tsuid': 'D485E2000001000F7300000200004B000003000F74'}]

            # At this point, if neither error is raised => At least one TS correspond to `fid_pattern`
            raise NameError("Error in `fid_pattern`: correspond to already created TS: {}".format(
                [x['funcId'] for x in result]
            ))
            # Example: "Error in `fid_pattern`: correspond to already created TS: ["PC_1", "PC_2"]"

        # If NO TS FOUND -> ok
        except ValueError:
            pass

    # fid_pattern (str)
    if type(fid_pattern) is not str:
        raise TypeError("Arg. type `fid_pattern` is {}, expected `str`".format(type(fid_pattern)))

    if '{pc_id}' not in fid_pattern:
        # Add `_PC{pc_id}` into `fid_pattern` (at the end)
        fid_pattern += "_PC{pc_id}"
        LOGGER.debug("Add '_PC{pc_id}' to arg. `fid_pattern`")

    __check_fid(pattern=fid_pattern, n=n_components)

    # Table name (str)
    if type(table_name) is not str:
        raise TypeError("Arg. `table_name` is {}, expected `str`".format(type(table_name)))
    if table_name is None or re.match('^[a-zA-Z0-9-_]+$', table_name) is None:
        raise ValueError("Error in table name")
    # Table name: is it an already created table ?
    try:
        # Table already exist ...
        IkatsApi.table.read(name=table_name)
        # ... raise error
        raise ValueError("Table `{}` already exist ! Please, specify an other `table_name`".format(table_name))
    # Table do not already exist...
    except IkatsNotFoundError:
        pass  # ... ok

    # Nb points by chunk (int > 0)
    if type(nb_points_by_chunk) is not int or nb_points_by_chunk <= 0:
        raise TypeError("Arg. `nb_points_by_chunk` must be an integer > 1, get {}".format(nb_points_by_chunk))

    # nb_ts_criteria (int > 0)
    if type(nb_ts_criteria) is not int or nb_ts_criteria <= 0:
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
        return spark_pca(tsuid_list=tsuid_list, fid_pattern=fid_pattern, n_components=n_components,
                         table_name=table_name, nb_points_by_chunk=nb_points_by_chunk)
    else:
        return pca(tsuid_list=tsuid_list, fid_pattern=fid_pattern, n_components=n_components, table_name=table_name)


def _format_table(matrix, rownames, table_name,
                  colnames=['explained variance', 'cumulative explained variance']):
    """
    Fill an ikats table structure with table containing explained variance.

    ..Example of input: colnames: [CPId, explained variance, cumulative explained variance]

    :param matrix: array containing values to store
    :type matrix: numpy array

    :param rownames: The name of all rows used (take care to the order)
    :type rownames: list of str

    :param : table_name: Name of the table to save
    :type : table_name: str

    :param colnames: The name of all column used (take care to the order)
    :type colnames: list of str

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
    table['content']['cells'] = np.array(matrix).tolist()

    return table
