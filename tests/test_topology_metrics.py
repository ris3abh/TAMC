import numpy as np

from topology_metrics import topology_summary_vector, vietoris_rips_persistence


def test_vietoris_rips_persistence_runs():
    point_cloud = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    persistence = vietoris_rips_persistence(point_cloud, max_dimension=1)
    assert len(persistence) > 0


def test_topology_summary_vector_has_expected_keys():
    point_cloud = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    summary = topology_summary_vector(point_cloud, max_dimension=1)

    assert "h0_entropy" in summary
    assert "h0_total_persistence" in summary
    assert "h0_max_persistence" in summary
    assert "h1_entropy" in summary
    assert "h1_total_persistence" in summary
    assert "h1_max_persistence" in summary
