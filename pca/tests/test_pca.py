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
import unittest
from collections import defaultdict

import numpy as np
# pyspark
import pyspark.ml.feature
# sklearn
import sklearn.decomposition
# Center / scale values
from sklearn.preprocessing import StandardScaler

# Core code
from ikats.algo.pca.pca import pca_ts_list, Pca, SEED, _INPUT_COL, _OUTPUT_COL, _format_table
from ikats.core.resource.api import IkatsApi
from ikats.core.resource.client import ServerError

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
    3: "2 TS with no same period"
}

# TOLERANCE for tests: we assume that this tol is acceptable
# (results between Spark and sklearn can be different at `tolerance`)
tolerance = 1e-5

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
        time = np.arange(14879030000, 14879030000 + (n_times * 1000), 1000)

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

        # Add time iteratively to each ts
        for ts in value:
            ts_content.append(np.array([time, ts]).T)

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

        # Create TS
        current_ts_created = IkatsApi.ts.create(fid=current_fid,
                                                data=ts_content[ts])
        # `current_ts_created`: dict containing tsuid, fid created, and status

        # If error on creation:
        if not current_ts_created['status']:
            raise SystemError("Error while creating TS %s" % ts_id)

        # Generate metadata (`qual_nb_points`, `metric`, `funcId`
        # NO PERIOD
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="qual_nb_points", value=ts_content.shape[1], force_update=True)
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="metric", value="metric_%s" % ts_id, force_update=True)
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="funcId", value=current_fid, force_update=True)

        # Finally, add to result
        result.append({"tsuid": current_ts_created["tsuid"],
                       "funcId": current_ts_created["funcId"]})

    return result, n_times


class TesScale(unittest.TestCase):
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
        # Test default implementation (no spark, n_components=None, random_state = SEED)
        # -----------------------------------------------------------------------------
        value = Pca().pca

        # -> Should be object sklearn.decomposition.PCA
        expected_type = sklearn.decomposition.PCA
        msg = "Error in init `Pca` object, get type {}, expected type {}"
        self.assertEqual(type(value), expected_type, msg=msg.format(type(value), expected_type))

        # -> Arg `copy` should be set to `False`
        msg = "Error in init `Pca`, arg `copy` is {}, should be set to `False` "
        self.assertFalse(value.copy, msg=msg.format(value.copy))

        # -> Arg `n_components` should be set to `None`
        msg = "Error in init `Pca`, arg `n_components` is {}, should be set to `None` "
        self.assertIsNone(value.n_components, msg=msg.format(value.n_components))

        # -> Arg `random_state` should be set to `SEED`
        msg = "Error in init `Pca`, arg `random_state` is {}, should be set to `{}` "
        self.assertEqual(value.random_state, SEED, msg=msg.format(value.random_state, SEED))

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
                pca_ts_list(ts_list=0.5, n_components=2)

            # empty TS list
            msg = "Testing arguments : Error in testing `ts_list` as empty list"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=[], n_components=2)

            # n_components
            # ----------------------------
            # wrong type (not int)
            msg = "Testing arguments : Error in testing `n_components` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components="no")

            # wrong type (None and not int)
            msg = "Testing arguments : Error in testing `n_components` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=None)

            # Not > 0
            msg = "Testing arguments : Error in testing `n_components` negative value"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=-100)

            # fid_pattern
            # ----------------------------
            # wrong type (not str)
            msg = "Testing arguments : Error in testing `fid_pattern` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components=2, fid_pattern=2)

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

            # nb_points_by_chunk
            # ----------------------------
            # Wrong type (not int)
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, nb_points_by_chunk="a")

            # Not > 0
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` negative value"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, nb_points_by_chunk=-100)

            # nb_ts_criteria
            # ----------------------------
            # Wrong type (not int)
            msg = "Testing arguments : Error in testing `nb_ts_criteria` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, nb_ts_criteria="a")

            # Not > 0
            msg = "Testing arguments : Error in testing `nb_ts_criteria` negative value"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, nb_ts_criteria=-100)

            # spark
            # ----------------------------
            # Wrong type (not NoneType or bool)
            msg = "Testing arguments : Error in testing `spark` type"
            with self.assertRaises(TypeError, msg=msg):
                pca_ts_list(ts_list=tsuid_list, spark="True")

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
                pca_ts_list(ts_list=ts_list, n_components=2, spark=False)
            # Test on spark mode
            with self.assertRaises(ValueError, msg=msg.format(case, USE_CASE[case], "SPARK")):
                pca_ts_list(ts_list=ts_list, n_components=2, spark=True)

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
                 fid_pattern="PC{pc_id}",
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
            # perform pca (arg `Spark` forced to `False` for testing NO SPARK mode)
            result_tslist, result_table_name = pca_ts_list(ts_list=tsuid_list,
                                                           n_components=n_components,
                                                           fid_pattern="PC{pc_id}",
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

    # @unittest.skip
    # FOR NOW, SPARK AND SKLEARN PRODUCE DIFFERENT RESULTS
    def test_diff_spark(self):
        """
        Testing difference of result between "Spark" and "No Spark"
        """
        # Arguments
        n_components = 2
        NB_POINTS_BY_CHUNK = 2
        # Table names
        table_spark = "Variance_explained_SPARK_PCA"
        table_nospark = "Variance_explained_PCA"

        # Get the TSUID of the saved TS
        try:
            IkatsApi.table.delete(table_spark)
            IkatsApi.table.delete(table_nospark)
        # Except: if the table do NOT already exist
        except ServerError:
            pass
        finally:
            # For each use case
            for case in [1]:
                # CASE 1: 4 TS, 5 times, scaled/centered

                ts_list, n_times = gen_ts(case)

                def __get_results(fid_pattern,
                                  table_name,
                                  spark,
                                  ts_list=ts_list,
                                  n_components=n_components,
                                  nb_points_by_chunk=NB_POINTS_BY_CHUNK):
                    """
                    Generate result from function `pca_ts_list`.
                    Same arg. than `pca_ts_list`.

                    :return: Tuple composed by:
                        * result_values: list of values [(time, values)...] (one list per TS) of the created PC
                        * result_table: The created table containing variance explained
                        * the list of ts created (for using `clean_up_db`)
                    """
                    # Perform pca, and get the resulting tsuid (force spark usage)
                    result_ts_list, result_table = pca_ts_list(ts_list=ts_list,
                                                               n_components=n_components,
                                                               fid_pattern=fid_pattern,
                                                               table_name=table_name,
                                                               nb_points_by_chunk=nb_points_by_chunk,
                                                               spark=spark)

                    # Get the list of tsuid (ex: ['tsuid1', 'tsuid2', ...])
                    result_tsuid = [x['tsuid'] for x in result_ts_list]

                    # Get the TS values (ex: [ [[time1, value1], [time2, value2],...] ])
                    result_values = np.array(IkatsApi.ts.read(result_tsuid))
                    # Shape: (n_ts, n_times, 2)

                    # Get the table of variance explained
                    result_table = IkatsApi.table.read(result_table)

                    return result_values, result_table, result_ts_list

                try:
                    # GET SPARK RESULT
                    # ------------------------
                    # Perform pca, and get the resulting tsuid (force spark usage)
                    result_values_spark, result_table_spark, result_tslist_spark = __get_results(fid_pattern="UNIT_TEST_SPARK_PC{pc_id}",
                                                                                                 table_name=table_spark,
                                                                                                 spark=True)

                    # GET NO SPARK RESULT
                    # ------------------------
                    result_values_nospark, result_table_nospark, result_tslist = __get_results(fid_pattern="UNIT_TEST_PC{pc_id}",
                                                                                               table_name=table_nospark,
                                                                                               spark=False)

                    # COMPARE TABLE OF VAR EXPLAINED
                    # -------------------------------
                    # Just test the content of the tables (each key is already checked)
                    result_table_nospark = np.array(result_table_nospark['content']['cells']).flatten()
                    result_table_spark = np.array(result_table_spark['content']['cells']).flatten()

                    msg = "Error in compare Spark/no spark: variance explained case {} ({}) \n" \
                          "Result Spark: {} \n" \
                          "Result no spark {}.\n" \
                          "Difference: {}".format(case,
                                                  USE_CASE[case],
                                                  result_table_spark,
                                                  result_table_spark,
                                                  np.subtract(result_table_nospark, result_table_spark))
                    self.assertTrue(np.allclose(result_table_nospark, result_table_spark,
                                                atol=tolerance), msg=msg)

                    # For each ts result
                    for ts in range(len(result_values_spark)):
                        # GET SPARK VALUES
                        # ------------------------
                        # Get column "Value"  ([:, 1])
                        result_values_ts_spark = np.array(result_values_spark[ts][:, 1])

                        # GET NO SPARK RESULT
                        # ------------------------
                        # Get column "Value"  ([:, 1])
                        result_values_ts_nospark = np.array(result_values_nospark[ts][:, 1])

                        # COMPARE VALUES OF PC
                        # ------------------------
                        msg = "Error in compare Spark/no spark: case {} ({}) \n" \
                              "Result Spark: {} \n" \
                              "Result no spark {}.\n" \
                              "Difference: {}".format(case,
                                                      USE_CASE[case],
                                                      result_values_ts_spark,
                                                      result_values_ts_nospark,
                                                      np.subtract(result_values_ts_spark, result_values_ts_nospark))

                        self.assertTrue(np.allclose(result_values_ts_spark, result_values_ts_nospark,
                                                    atol=tolerance), msg=msg)

                except Exception:
                    raise

                finally:
                    # Delete generated TS (from function `gen_ts`)
                    self.clean_up_db(ts_list)
                    # Delete TS created by `pca_ts_list` function
                    self.clean_up_db(result_tslist_spark)  # SPARK MODE
                    self.clean_up_db(result_tslist)  # NO SPARK MODE
                    # Delete tables containing variance explained
                    IkatsApi.table.delete(result_table_spark)  # SPARK MODE
                    IkatsApi.table.delete(result_table_nospark)  # NO SPARK MODE

