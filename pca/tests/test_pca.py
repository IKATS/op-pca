"""
Copyright 2018-2019 CS Systèmes d'Information

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
import unittest
from collections import defaultdict
import time
import numpy as np
# pyspark
import pyspark.ml.feature
# sklearn
import sklearn.decomposition
# Center / scale values
from sklearn.preprocessing import StandardScaler

# Core code
from ikats.algo.pca.pca import pca_ts_list, Pca, _INPUT_COL, _OUTPUT_COL, _format_table
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client import ServerError
from ikats.core.library.exception import IkatsConflictError

# Set LOGGER
LOGGER = logging.getLogger()
# Log format
LOGGER.setLevel(logging.INFO)#DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.INFO)#DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

# All use case tested during this script.
# Keys are argument `ts_id` of function `gen_ts`
USE_CASE = {
    1: "4 TS, 5 times, centered/scaled",
    2: "2 TS with no same start/end date",
    3: "2 TS with no same period",
    4: "30 random TS with 50 timestamps"  # SEED is fixed
}

# TOLERANCE for tests: we assume that this tol is acceptable
# (results between Spark and sklearn can be different at `tolerance`)
tolerance = 1e-5

# SEED
SEED = 0
# Set the seed: making results reproducible
np.random.seed(SEED)


def gen_ts(ts_id):
    """
    Generate a TS in database used for test bench where id is defined

    :param ts_id: Identifier of the TS to generate (see content below for the structure)
    :type ts_id: int

    :return: Tuple composed by:
        * the TSUID, funcId and all expected result (one per scaling)
        * the number of timestamp of the dataset
    :rtypes:
        * dict
        * int
    """

    # Build TS identifier
    fid = "UNIT_TEST_PCA_%s" % ts_id

    # 1/ Choose test case
    # ----------------------------------------------------------------
    if ts_id == 1:
        # CASE: Very simple dataset, already centered/scaled

        # Number of times
        n_times = 5
        # Get timestamps
        time1 = np.arange(14879030000, 14879030000 + (n_times * 1000), 1000)

        # Get values
        value = np.array([[1., 2., 3., 4., 4.],
                 [4, 5., 6., 7., 7.],
                 [7., 8., 9., 10., 10.],
                 [4., 3., 2., 1., 1.]])
        # shape = (n_ts, n_times)

        # Perform scaling (scale/center)
        scaler = StandardScaler(with_mean=True, with_std=True)
        value = scaler.fit_transform(value)

        # [[-1.41421356, -1.09108945, -0.73029674, -0.4472136 ],
        # [ 0.        ,  0.21821789,  0.36514837,  0.4472136 ],
        # [ 1.41421356,  1.52752523,  1.46059349,  1.34164079],
        # [ 0.        , -0.65465367, -1.09544512, -1.34164079]]

        # Build TS data
        # ---------------
        ts_content = []

        # Add time1 iteratively to each ts
        for ts in value:
            ts_content.append(np.array([time1, ts]).T)

        ts_content = np.array(ts_content)
        # ts_content.shape = (n_ts, n_times, 2) = (4, 5, 2)

    elif ts_id == 2:
        # CASE: 2 TS not aligned : not the same start date

        # Number of times
        n_times = 5

        # Get timestamps
        # ---------------
        # Gap between the 2 TS
        gap = 1000

        time1 = list(range(14879030000 + gap, 14879030000 + gap + (n_times * 1000), 1000))
        time2 = list(range(14879030000, 14879030000 + (n_times * 1000), 1000))

        # Get values
        value = [1., 2., 3., 4., 4.]
        # shape = (n_ts, n_times)

        # Build TS data
        # ---------------
        ts_content = np.array([np.array([time1, value]).T,
                               np.array([time2, value]).T])
        # ts_content.shape = (n_ts, n_times, 2) = (2, 5, 2)

    elif ts_id == 3:
        # CASE: 2 TS not aligned : not the period

        # Number of times
        n_times = 5

        # Get timestamps
        # ---------------
        # Gap between the 2 TS
        gap = 10

        time1 = list(range(14879030000, 14879030000 + (n_times * (1000 + gap)), 1000 + gap))
        time2 = list(range(14879030000, 14879030000 + (n_times * 1000), 1000))

        # Get values
        value = [1., 2., 3., 4., 4.]
        # shape = (n_ts, n_times)

        # Build TS data
        # ---------------
        ts_content = np.array([np.array([time1, value]).T,
                               np.array([time2, value]).T])
        # ts_content.shape = (n_ts, n_times, 2) = (2, 5, 2)

    elif ts_id == 4:
        # CASE : 30 random TS with 50 timestamps
        n_times = 50
        n_ts = 30

        # Timestamps
        time1 = np.arange(14879030000, 14879030000 + (n_times * 1000), 1000)

        # Random float values in [0, 5[
        values = np.random.ranf(size=(n_ts, n_times)) * 5
        # Random array, Shape = (n_ts, n_times) = (30, 50)

        # Build TS data
        # ---------------
        ts_content = []
        for ts in range(n_ts):
            ts_content.append(np.array([time1, values[ts, :]]).T)
            # ts_content.shape = (n_ts, n_times, 2) = (30, 50, 2)

        ts_content = np.array(ts_content)

    else:
        raise NotImplementedError

    # 2/ Build result
    # ----------------------------------------------------------------
    # Create the time series, build custom meta, and add into result
    result = []

    # For each available TS (shape[0])
    for ts in range(ts_content.shape[0]):

        # Generate fid
        current_fid = fid + '_TS_{}'.format(ts)
        # Example: "UNIT_TEST_PCA_1_TS_0" -> test case 0, ts n°0

        # Create TSUID
        current_ts_created = IkatsApi.ts.create(fid=current_fid,
                                                data=ts_content[ts])
        # `current_ts_created`: dict containing tsuid, fid created, and status

        # If error on creation:
        if not current_ts_created['status']:
            raise SystemError("Error while creating TS %s" % ts_id)

        # Generate metadata (`qual_nb_points`)
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="qual_ref_period", value=1000, force_update=True)

        # Finally, add to result
        result.append({"tsuid": current_ts_created["tsuid"],
                       "funcId": current_ts_created["funcId"]})

    return result, n_times


class TestPCA(unittest.TestCase):
    """
    Test the pca operator
    """

    @staticmethod
    def clean_up_db(ts_info):
        """
        Clean up the database by removing created TS
        :param ts_info: list of TS to remove
        """
        for ts_item in ts_info:
            # Delete created TS
            IkatsApi.ts.delete(tsuid=ts_item['tsuid'], no_exception=True)

    def test_class_pca(self):
        """
        Testing class `Pca`
        """
        # Test default implementation (no spark, n_components=None, random_state = None)
        # -----------------------------------------------------------------------------
        value = Pca().pca

        # -> Should be object sklearn.decomposition.PCA
        expected_type = sklearn.decomposition.PCA
        msg = "Error in init `Pca` object, get type {}, expected type {}"
        self.assertEqual(type(value), expected_type, msg=msg.format(type(value), expected_type))

        # -> Arg `copy` should be set to `True`
        msg = "Error in init `Pca`, arg `copy` is {}, should be set to `True` "
        self.assertTrue(value.copy, msg=msg.format(value.copy))

        # -> Arg `n_components` should be set to `None`
        msg = "Error in init `Pca`, arg `n_components` is {}, should be set to `None` "
        self.assertIsNone(value.n_components, msg=msg.format(value.n_components))

        # -> Arg `random_state` should be set to `None`
        msg = "Error in init `Pca`, arg `random_state` is {}, should be set to `None` "
        self.assertEqual(value.random_state, None, msg=msg.format(value.random_state))

        # Test implementation with spark
        # ----------------------------------------------
        value = Pca(spark=True).pca

        # -> Should be object sklearn.decomposition.PCA
        expected_type = pyspark.ml.feature.PCA
        msg = "Error in init `Pca` object, get type {}, expected type {}"
        self.assertEqual(type(value), expected_type, msg=msg.format(type(value), expected_type))

        # -> Arg `n_components` should be set to `None`
        msg = "Error in init `Pca`, arg `n_components` is {}, should be set to `None` "
        self.assertIsNone(value.getK(), msg=msg.format(value.getK()))

        # -> Arg `InputCol` should be set to `_INPUT_COL`
        msg = "Error in init `Pca` object (spark), arg `InputCol` is {}, expected `{}`"
        self.assertEqual(value.getInputCol(), _INPUT_COL,
                         msg=msg.format(value.getInputCol(), _INPUT_COL))

        # -> Arg `OutputCol` should be set to `_OUTPUT_COL`
        msg = "Error in init `Pca` object (spark), arg `OutputCol` is {}, expected `{}`"
        self.assertEqual(value.getOutputCol(), _OUTPUT_COL,
                         msg=msg.format(value.getOutputCol(), _OUTPUT_COL))

    @staticmethod
    def check_type_table(arg_to_test):
        """
        Test if argument `arg_to_test` is an IKATS table (dict with particular keys).

        :param arg_to_test: The element to test

        :return: True if `arg_to_test` is an IKATS table, False else.
        :rtype: bool

        Expected result:
        {'content': {
            'cells': [['0.7', '0.7'],
                      ['0.3', '1.0']]
            },
        'headers': {
            'col': {'data': ['Var', 'explained variance', 'cumulative explained variance']},
            'row': {'data': ['PC', 'PC1', 'PC2']}
            },
        'table_desc': {
            'desc': 'Explained variance from PCA operator',
            'name': 'UNIT_TEST_TABLE_RESULT',
            'title': 'Variance explained'
            }
        }

        ..Note: All Errors raised are `TypeError` (easier to test)
        """
        # ----------------------------------------------------
        # 1/ Test global type (defaultdict or dict)
        # ----------------------------------------------------
        msg = "Arg. `arg_to_test` have type {}, expected `defaultdict` or `dict`"
        if (type(arg_to_test) is not defaultdict) and (type(arg_to_test) is not dict):
            raise TypeError(msg.format(type(arg_to_test)))

        #
        # ----------------------------------------------------
        # 2/ Test "level one" keys (['content', 'header', 'table_desc'])
        # ----------------------------------------------------
        result = [k in ['content', 'headers', 'table_desc'] for k in list(arg_to_test.keys())]

        msg = "Arg. `arg_to_test` have incorrect keys, get {}, expected `['content', 'headers', 'table_desc']`"
        # Test if `result` contains at least one `False`
        if all(result) is False:
            raise TypeError(msg.format(list(arg_to_test.keys())))

        # ----------------------------------------------------
        #  3/ Test "level two" keys: These keys have to exist
        # ----------------------------------------------------
        # ['content']['cells'], ['headers']['row'], ['headers']['col'], ['table_desc']['desc'],
        # ['table_desc']['name'], ['table_desc']['title']

        # sub-key ['content']['cells']
        # ------------------------
        msg = "Arg. `arg_to_test` have incorrect sub-key, get {}, expected `['cells']`"

        # Test if `result` contains at least one `False`
        if list(arg_to_test['content'].keys()) != ['cells']:
            raise TypeError(msg.format(list(arg_to_test['content'].keys())))

        # sub-keys ['headers']['row'], ['headers']['col']
        # -------------------------------------
        msg = "Arg. `arg_to_test` have incorrect sub-keys, get {}, expected `['row', 'col']`"

        if set(arg_to_test['headers'].keys()) != set(('row', 'col')):
            raise TypeError(msg.format(list(arg_to_test['headers'].keys())))

        # sub-keys ['table_desc']['desc'], ['table_desc']['name'], ['table_desc']['title']
        # -------------------------------------------------------------------------------
        msg = "Arg. `arg_to_test` have incorrect sub-keys, get {}, expected `['desc', 'name', 'title']`"

        if set(arg_to_test['table_desc'].keys()) != set(['desc', 'name', 'title']):
            raise TypeError(msg.format(list(arg_to_test['desc'].keys())))

        # ----------------------------------------------------
        # 4/ Test "level three" keys: These keys have to exist
        # ----------------------------------------------------

        # sub-keys ['headers']['row']['data'], ['headers']['col']['data']
        # -------------------------------------
        msg = "Arg. `arg_to_test` have incorrect sub-keys, get {}, expected `{}`"

        if set(arg_to_test['headers']['col'].keys()) != set(['data']):
            raise TypeError(msg.format(set(arg_to_test['headers']['col'].keys()), set(['data'])))
        if set(arg_to_test['headers']['row'].keys()) != set(['data']):
            raise TypeError(msg.format(set(arg_to_test['headers']['row'].keys()), set(['data'])))
        # ----------------------------------------------------
        # 5/ Test content type: These type have to match
        # ----------------------------------------------------
        # ['content']['cells']: list,
        # ['headers']['row']['data'], ['headers']['col']['data']: list of str
        # ['table_desc']['desc'], ['table_desc']['name'], ['table_desc']['title'] : str

        # ['content']['cells']: np.array
        # -------------------------------
        msg = "Error in key: `['content']['cells']`, get type {}, expected `list`"

        if type(arg_to_test['content']['cells']) is not list:
            raise TypeError(msg.format(type(arg_to_test['content']['cells'])))

        #  ['headers']['row']['data'], ['headers']['col']['data']: list of str
        # --------------------------------------------------
        # Test type list
        if type(arg_to_test['headers']['row']['data']) is not list:
            raise TypeError("Error in key: `['headers']['row']['data']`, get type `{}`, expected `list`".format(
                type(arg_to_test['headers']['row']['data'])))
        if type(arg_to_test['headers']['col']['data']) is not list:
            raise TypeError("Error in key: `['headers']['col']['data']`, get type `{}`, expected `list`".format(
                type(arg_to_test['headers']['col']['data'])))

        # Test if the lists contains str only
        result = [type(k) is str for k in arg_to_test['headers']['row']['data']]
        if all(result) is False:
            raise TypeError("Error in key ['headers']['row']['data'], get {}, expected list full of str".format(
                [type(k) is str for k in arg_to_test['headers']['row']['data']]))
        result = [type(k) is str for k in arg_to_test['headers']['col']['data']]
        if all(result) is False:
            raise TypeError("Error in key ['headers']['col']['data'], get {}, expected list full of str".format(
                [type(k) is str for k in arg_to_test['headers']['col']['data']]))

        #  ['table_desc']['desc'], ['table_desc']['name'], ['table_desc']['title'] : str
        # -------------------------------------------------------------------------------
        result = [type(arg_to_test['table_desc'][key]) for key in ['desc', 'name', 'title']]

        msg = "Error in content of sub-keys ['table_desc']['desc'], ['table_desc']['name'], ['table_desc']['title']," \
              " get {}, expected `str`"
        # Test if `result` contains at least one `False`
        if result == 3*[type(str)]:
            raise TypeError(msg.format(result))

    def test_format_table(self):
        """
        Testing function `_format_table`
        """
        # CREATE DATA
        table_name="UNIT_TEST_TABLE_RESULT"
        matrix=np.array([['0.7', '0.7'], ['0.3', '1.0']])
        rownames= ["PC1", "PC2"]

        # GENERATE TABLE
        table = _format_table(matrix=matrix, rownames=rownames, table_name=table_name)
        # Arg. `colnames` let to default
        # (`['explained variance', 'cumulative explained variance']`)

        # Check
        self.check_type_table(table)

    def test_arguments_pca_ts_list(self):
        """
        Testing behaviour when wrong arguments on function `pca_ts_list`.
        """

        # Get the TSUID of the saved TS
        tsuid_list, _ = gen_ts(1)

        try:

            # TS list
            # ----------------------------
            # Wrong type ((not list)
            msg = "Testing arguments : Error in testing `ts_list` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=0.5, n_components=2, table_name="a")

            # empty TS list
            msg = "Testing arguments : Error in testing `ts_list` as empty list"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=[], n_components=2, table_name="a")

            # n_components
            # ----------------------------
            # wrong type (not int)
            msg = "Testing arguments : Error in testing `n_components` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components="no", table_name="a")

            # wrong type (None and not int)
            msg = "Testing arguments : Error in testing `n_components` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=None, table_name="a")

            # Not > 0
            msg = "Testing arguments : Error in testing `n_components` negative value"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=-100, table_name="a")

            # Not <= len(ts_list)
            msg = "Testing arguments : Error in testing `n_components` value (> `n_ts`)"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=len(tsuid_list)+10, table_name="a")

            # fid_pattern
            # ----------------------------
            # wrong type (not str)
            msg = "Testing arguments : Error in testing `fid_pattern` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, fid_pattern=2, table_name="a")

            # fid already exist (IkatsConflictError)*
            # Create a TS corresponding to the fid pattern
            try:
                created_tsuid = IkatsApi.ts.create_ref("PC1")
                # If already exist: get tsuid
            except Exception:
                created_tsuid = IkatsApi.fid.tsuid("PC1")
            msg = "Testing arguments : Error in testing `fid_pattern` type"
            with self.assertRaises(NameError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, fid_pattern="PC{pc_id}", table_name="a")
            # Delete created fid
            IkatsApi.ts.delete(created_tsuid)

            # table_name
            # ----------------------------
            # wrong type (not str)
            msg = "Testing arguments : Error in testing `table_name` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, fid_pattern="a", table_name=2)

            # wrong regexp
            msg = "Testing arguments : Error in testing `table_name` regexp"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, fid_pattern="a", table_name="a b ")

            # `table name` already exist (ValueError)
            table_name = "UNIT_TEST_TABLE_PCA"

            # Create a test table content
            table = _format_table(np.array([[1, 2]]), rownames=["row"], table_name=table_name, colnames=['1', '2'])

            # Create the table
            IkatsApi.table.create(data=table)
            # Get the name of A TABLE ALREADY CREATED (can bug if no available table)
            msg = "Testing arguments : Error in testing `table_name` name (already exist)"
            with self.assertRaises(ValueError, msg=msg):
                pca_ts_list(ts_list=tsuid_list, n_components=2, fid_pattern="a", table_name=table_name)

            # Delete test table
            IkatsApi.table.delete(name=table_name)

            # nb_points_by_chunk
            # ----------------------------
            # Wrong type (not int)
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, table_name="a", nb_points_by_chunk="a")

            # Not > 0
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` negative value"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, table_name="a", nb_points_by_chunk=-100)

            # nb_ts_criteria
            # ----------------------------
            # Wrong type (not int)
            msg = "Testing arguments : Error in testing `nb_ts_criteria` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, table_name="a", nb_ts_criteria="a")

            # Not > 0
            msg = "Testing arguments : Error in testing `nb_ts_criteria` negative value"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, table_name="a", nb_ts_criteria=-100)

            # spark
            # ----------------------------
            # Wrong type (not NoneType or bool)
            msg = "Testing arguments : Error in testing `spark` type"
            with self.assertRaises(TypeError, msg=msg):
                pca_ts_list(ts_list=tsuid_list, n_components=2, table_name="a", spark="True")

        finally:
            # Clean up database
            self.clean_up_db(tsuid_list)

    def test_non_aligned_ts(self):
        """
        Test functions when TS are not aligned (not same start/end date or period)
        """
        for case in [2, 3]:
            # Case 2: Not the same `start date`
            # Case 3: Not the same `period`
            # ----------------------------------------------------
            ts_list, _ = gen_ts(case)

            msg = "Error in testing case {}: {}; on {} mode."
            # Test on NO spark mode
            with self.assertRaises(ValueError, msg=msg.format(case, USE_CASE[case], "NO SPARK")):
                pca_ts_list(ts_list=ts_list, n_components=2, table_name="a", spark=False)
            # Test on spark mode
            with self.assertRaises(ValueError, msg=msg.format(case, USE_CASE[case], "SPARK")):
                pca_ts_list(ts_list=ts_list, n_components=2, table_name="a", spark=True)

    def test_pca_values(self):
        """
        Testing the result values of the pca algorithm.
        """
        # Get the TSUID of the saved TS
        try:
            IkatsApi.table.delete("Variance_explained_PCA")
        # Except: if the table do NOT already exist
        except ServerError:
            pass
        finally:
            tsuid_list, n_times = gen_ts(1)

        try:
            # Number of Principal component to build
            n_components = 2
            # perform pca (arg `Spark` forced to `False` for testing NO SPARK mode)
            result_tslist, result_table_name = pca_ts_list \
                (ts_list=tsuid_list,
                 n_components=n_components,
                 fid_pattern="PC{pc_id}_test",
                 table_name="Variance_explained_PCA",
                 spark=False)

            # 1/ Test output type
            # ---------------------------------------------
            # Check type of `result_tslist` (list)
            msg = "Error, `result_tslist` have type {}, expected `list`"
            self.assertTrue(type(result_tslist) is list, msg=msg.format(type(result_tslist)))

            # Check type of `result_table_name` (str)
            msg = "Error, `result_table_name` have type {}, expected `str`"
            self.assertTrue(type(result_table_name) is str, msg=msg.format(type(result_table_name)))

            # Check type of table from `result_table_name` (table)
            self.check_type_table(IkatsApi.table.read(result_table_name))

            # 2/ Test outputs values (TS transformed)
            # ---------------------------------------------
            # Get the resulted tsuid
            result_tsuid = [x['tsuid'] for x in result_tslist]

            # read TS
            #  List of TS [ [[time1, value1], [time2, value2],...] ]
            result_values = IkatsApi.ts.read(result_tsuid)

            # Test shape of result
            msg = "Error in result shape, get {}, expected {}"
            expected = (n_components, n_times, 2)

            self.assertTupleEqual(np.array(result_values).shape, expected,
                                  msg=msg.format(np.array(result_values), expected))

            # 3/ Clean TS list created / table created
            # ---------------------------------------------
            self.clean_up_db(result_tslist)
            IkatsApi.table.delete(result_table_name)
        finally:
            # Clean up database
            self.clean_up_db(tsuid_list)

    def test_spark(self):
        """
          Testing the result values of the pca algorithm.
          """
        # Get the TSUID of the saved TS
        try:
            IkatsApi.table.delete("Variance_explained_PCA")
        # Except: if the table do NOT already exist
        except ServerError:
            pass
        finally:
            tsuid_list, n_times = gen_ts(1)

        try:
            # Number of Principal component to build
            n_components = 2
            # perform pca (arg `Spark` forced to `True` for testing SPARK mode)
            result_tslist, result_table_name = pca_ts_list(ts_list=tsuid_list,
                                                           n_components=n_components,
                                                           fid_pattern="PC{pc_id}_test",
                                                           table_name="Variance_explained_PCA",
                                                           spark=True)

            # 1/ Test output type
            # ---------------------------------------------
            # Check type of `result_tslist` (list)
            msg = "Error, `result_tslist` have type {}, expected `list`"
            self.assertTrue(type(result_tslist) is list, msg=msg.format(type(result_tslist)))

            # Check type of `result_table_name` (str)
            msg = "Error, `result_table_name` have type {}, expected `str`"
            self.assertTrue(type(result_table_name) is str, msg=msg.format(type(result_table_name)))

            # Check type of table from `result_table_name` (table)
            self.check_type_table(IkatsApi.table.read(result_table_name))

            # 2/ Test outputs values (TS transformed)
            # ---------------------------------------------
            # Get the resulted tsuid
            result_tsuid = [x['tsuid'] for x in result_tslist]

            # read TS
            #  List of TS [ [[time1, value1], [time2, value2],...] ]
            result_values = IkatsApi.ts.read(result_tsuid)

            # Test shape of result
            msg = "Error in result shape, get {}, expected {}"
            expected = (n_components, n_times, 2)

            self.assertTupleEqual(np.array(result_values).shape, expected,
                                  msg=msg.format(np.array(result_values), expected))

            # 3/ Clean TS list created / table created
            # ---------------------------------------------
            self.clean_up_db(result_tslist)
            IkatsApi.table.delete(result_table_name)
        finally:
            # Clean up database
            self.clean_up_db(tsuid_list)

    # FOR NOW, SPARK AND SKLEARN PRODUCE DIFFERENT RESULTS: NO TEST
