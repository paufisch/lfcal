"""Result records written before the switch to `ode-filters` 0.8 read back in the textbook length scale."""

from lfcal.paper.io import read_csv, write_csv


def test_legacy_length_scales_are_converted_on_reading(tmp_path):
    legacy = {"log_sigma_x2": 1.0, "log_xi_u": 0.5, "log_ell_u": 2.0, "dlog_ell_u": 0.4}
    current = {"log_ell_u": 1.0, "dlog_ell_u": 0.2, "length_scale": "textbook"}
    prefixed = {"obs_log_ell_u": 3.0, "mi_log_ell_u": 1.0}
    unrelated = {"rel_u": 0.1}
    write_csv(tmp_path / "records.csv", [legacy, current, prefixed, unrelated])
    legacy_read, current_read, prefixed_read, unrelated_read = read_csv(tmp_path / "records.csv")
    assert legacy_read == {"log_sigma_x2": 1.0, "log_xi_u": 0.5, "log_ell_u": 1.0, "dlog_ell_u": 0.2,
                           "length_scale": "textbook"}
    assert current_read == current
    assert prefixed_read == {"obs_log_ell_u": 1.5, "mi_log_ell_u": 0.5, "length_scale": "textbook"}
    assert unrelated_read == unrelated


def test_converted_records_are_not_converted_twice(tmp_path):
    write_csv(tmp_path / "legacy.csv", [{"log_ell_u": 2.0}])
    write_csv(tmp_path / "again.csv", read_csv(tmp_path / "legacy.csv"))
    assert read_csv(tmp_path / "again.csv") == [{"log_ell_u": 1.0, "length_scale": "textbook"}]
