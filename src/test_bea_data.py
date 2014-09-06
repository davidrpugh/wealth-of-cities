"""
Unit testing framework for bea_data_grab.py module.

author : David R. Pugh
date : 2014-09-06

"""
import os
import unittest

import pandas as pd

import bea_data_grab


class TestBEADownload(unittest.TestCase):

    base_url = 'http://www.bea.gov/api/data?'

    api_key = '6F508DBE-6979-4E85-A335-8DAE6187FC0D'

    # we are interested in grabbing the following data series:
    key_codes = ['POP_MI', 'GDP_MP', 'RGDP_MP', 'PCRGDP_MP', 'TPI_MI',
                 'PCPI_MI', 'DIR_MI', 'PCTR_MI', 'WS_MI', 'SUPP_MI',
                 'PROP_MI']

    def setUp(self):
        """Setup test fixtures."""
        # remove files (if they already exist)
        files = ['../data/raw_bea_metro_data.csv']
        for tmp_file in files:
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)

    def tearDown(self):
        """Teardown test fixtures."""
        self.setUp()

    def test_download_bea_data(self):
        """Test BEA data download."""
        raw_bea_data = bea_data_grab.download_bea_data(self.base_url,
                                                       self.api_key,
                                                       self.key_codes)

        # raw_bea_data should be a DataFrame
        self.assertIsInstance(raw_bea_data, pd.DataFrame)

        # raw_bea_data should have all the necessary columns
        actual_variables = set(raw_bea_data['Code'].unique())
        expected_variables = set(self.key_codes)
        self.assertTrue(actual_variables, expected_variables)

    def test_clean_bea_data(self):
        """Test cleaning of the bea data."""
        pass
