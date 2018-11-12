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

# Core code
from ikats.algo.pca.pca import pca_ts_list
from ikats.core.resource.api import IkatsApi


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
        fid = fid + '_TS_{}'.format(ts)
        # Example: "UNIT_TEST_PCA_1_TS_0" -> test case 0, ts n°0

        # Create TS
        current_ts_created = IkatsApi.ts.create(fid=fid,
                                                data=ts_content[ts])
        # `current_ts_created`: dict containing tsuid, fid created, and status

        # If error on creation:
        if not current_ts_created['status']:
            raise SystemError("Error while creating TS %s" % ts_id)

        # Generate metadata (`qual_nb_points`, `metric`, `funcId`
        # NO PERIOD
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="qual_nb_points", value=ts_content.shape[1], force_update=True)
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="metric", value="metric_%s" % ts_id, force_update=True)
        IkatsApi.md.create(tsuid=current_ts_created['tsuid'], name="funcId", value=fid, force_update=True)

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

    def test_arguments_scale_ts_list(self):
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