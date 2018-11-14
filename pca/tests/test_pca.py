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
import numpy as np
from collections import defaultdict

# Core code
from ikats.algo.pca.pca import pca_ts_list, Pca, SEED, _INPUT_COL, _OUTPUT_COL, _format_table
from ikats.core.resource.api import IkatsApi

# sklearn
import sklearn.decomposition

# pyspark
import pyspark.ml.feature

# Set LOGGER
LOGGER = logging.getLogger()
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

# All use case tested during this script.
# Keys are argument `ts_id` of function `gen_ts`
USE_CASE = {
    1: "2 TS"
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

    :return: the TSUID, funcId and all expected result (one per scaling)
    :rtype: dict
    """

    # Build TS identifier
    fid = "UNIT_TEST_PCA_%s" % ts_id

    # 1/ Choose test case
    # ----------------------------------------------------------------
    if ts_id == 1:
        # CASE: Use 2 TS nearly identical (same mean, min, max, sd)
        time = list(range(14879030000, 14879039000, 1000))
        value1 = [2.3, 3.3, 4.4, 9.9, 0.1, -1.2, -12.13, 20.6, 0.0]
        value2 = [0.0, 2.3, 3.3, 4.4, 9.9, 0.1, -1.2, -12.13, 20.6]

        ts_content = np.array([np.array([time, value1]).T,
                              np.array([time, value2]).T],
                              np.float64)
        # Shape = (n_ts, n_row, 2), where n_row = n_times

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
        # NO PERIODnt_
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="qual_nb_points", value=ts_content.shape[1], force_update=True)
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="metric", value="metric_%s" % ts_id, force_update=True)
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="funcId", value=current_fid, force_update=True)

        # Finally, add to result
        result.append({"tsuid": current_ts_created["tsuid"],
                       "funcId": current_ts_created["funcId"]})

    return result


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
        # 1/ Test global type (dict)
        # ----------------------------------------------------
        msg = "Arg. `arg_to_test` have type {}, expected `defaultdict`"
        if type(arg_to_test) is not defaultdict:
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
        tsuid_list = gen_ts(1)

        try:

            # TS list
            # ----------------------------
            # Wrong type ((not list)
            msg = "Testing arguments : Error in testing `ts_list` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=0.5)

            # empty TS list
            msg = "Testing arguments : Error in testing `ts_list` as empty list"
            with self.assertRaises(ValueError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=[])

            # n_components
            # ----------------------------
            # wrong type (not int)
            msg = "Testing arguments : Error in testing `n_components` type"
            with self.assertRaises(TypeError, msg=msg):
                # noinspection PyTypeChecker
                pca_ts_list(ts_list=tsuid_list, n_components="no")

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

    def test_pca_values(self):
        """
        Testing the result values of the pca algorithm.
        """
        pass

    def test_spark(self):
        """
        Testing the result values of the pca algorithm, when spark is forced true.
        """
        pass

    def test_diff_spark(self):
        """
        Testing difference of result between "Spark" and "No Spark"
        """
        pass
