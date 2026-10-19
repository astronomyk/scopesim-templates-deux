import pytest

from scopesim_templates_deux import cluster
from scopesim_templates_deux.rc import Source


@pytest.mark.webtest
def test_actually_works():
    my_cluster = cluster(mass=900, distance=50000, core_radius=1)
    assert isinstance(my_cluster, Source)
