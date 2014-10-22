import nose

import master_data
import physical_distance

# compute the effective labor supply
raw_data = master_data.panel.minor_xs(2010)
clean_data = raw_data.sort('GDP_MP', ascending=False).drop([998, 48260])
population = clean_data['POP_MI'].values


def test_data_alignment():
    """Testing alignment of population and physical distance data."""
    condition = clean_data.index == physical_distance.geo_coords.index
    nose.tools.assert_true(condition.all())
