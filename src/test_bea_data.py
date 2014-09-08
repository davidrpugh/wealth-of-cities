"""
Unit testing framework for bea_data_grab.py module.

author : David R. Pugh
date : 2014-09-06

"""
import glob
import os
import unittest

import pandas as pd

import bea_data_grab


class TestBEADownload(unittest.TestCase):

    base_url = 'http://www.bea.gov/api/data?'

    api_key = '6F508DBE-6979-4E85-A335-8DAE6187FC0D'

    def setUp(self):
        """Setup test fixtures."""
        # remove data files (if necessary)
        files = glob.glob('../data/bea/*.json')
        for tmp_file in files:
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)

    def tearDown(self):
        """Teardown test fixtures."""
        self.setUp()

    def test_download_raw_json_data(self):
        """Testing BEA data download."""
        key_code = 'POP_MI'
        bea_data_grab.download_raw_json_data(self.base_url,
                                             self.api_key,
                                             key_code)

        self.assertTrue(os.path.isfile('../data/bea/' + key_code + '.json'))

    def test_remove_duplicate_rows(self):
        """Test removal of duplicate rows from raw BEA data."""
        #raw_bea_data = bea_data_grab.download_bea_data(self.base_url,
        #                                               self.api_key,
        #                                               self.key_codes)

        #check_for_dups = ['CL_UNIT', 'Code', 'DataValue', 'GeoFips', 'NoteRef',
        #                  'TimePeriod', 'UNIT_MULT']
        #deduplicated_data = bea_data_grab.remove_duplicate_rows(raw_bea_data)
        #self.assertFalse(deduplicated_data.duplicated(check_for_dups).all())
        pass