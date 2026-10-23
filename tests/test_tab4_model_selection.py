"""Tests for uq_physicell.gui.tab4_model_selection.

Covers:
- check_consistency_dbs's handling of missing/legacy BO_Options (warn, not raise)
- dic_table keying surviving add/remove cycles without collisions or KeyErrors
- the "Remove" button targeting the correct row after an earlier row is removed
"""

import types
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QTableWidget
from PyQt5.QtCore import Qt

from uq_physicell.gui import tab4_model_selection as tab4


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _metadata_df(bo_options=None, obs_data_path="obs.csv", botorch_version="0.1.0"):
    return pd.DataFrame([{
        "BO_Options": bo_options,
        "ObsData_Path": obs_data_path,
        "botorch_version": botorch_version,
    }])


def _qois_df():
    return pd.DataFrame({"QoI_Name": ["q1", "q2"], "QOI_Function": ["f1", "f2"]})


class TestCheckConsistencyDbsBoOptions:
    def test_missing_bo_options_warns_instead_of_raising(self):
        """Regression test: a database migrated from a pre-BO_Options schema has
        BO_Options=NULL. That must not permanently lock it out of the comparison
        tab with a hard ValueError -- it should warn instead."""
        opts = '{"use_exponential_fitness": true}'
        df_prev = _metadata_df(bo_options=None)  # legacy database
        df_current = _metadata_df(bo_options=opts)
        warnings = tab4.check_consistency_dbs(df_prev, df_current, _qois_df(), _qois_df())
        assert any("BO_Options" in w or "BO setup" in w for w in warnings)

    def test_both_missing_bo_options_warns(self):
        df_prev = _metadata_df(bo_options=None)
        df_current = _metadata_df(bo_options=None)
        warnings = tab4.check_consistency_dbs(df_prev, df_current, _qois_df(), _qois_df())
        assert any("BO setup" in w for w in warnings)

    def test_score_defining_mismatch_still_raises(self):
        """A real, verifiable mismatch must still be a hard error."""
        df_prev = _metadata_df(bo_options='{"use_exponential_fitness": true}')
        df_current = _metadata_df(bo_options='{"use_exponential_fitness": false}')
        with pytest.raises(ValueError, match="use_exponential_fitness"):
            tab4.check_consistency_dbs(df_prev, df_current, _qois_df(), _qois_df())

    def test_consistent_bo_options_no_warning_about_bo_options(self):
        opts = '{"use_exponential_fitness": true, "summary_function": "s"}'
        df_prev = _metadata_df(bo_options=opts)
        df_current = _metadata_df(bo_options=opts)
        warnings = tab4.check_consistency_dbs(df_prev, df_current, _qois_df(), _qois_df())
        assert not any("BO setup" in w for w in warnings)


class TestDicTableKeying:
    """Regression tests for the model_id / dic_table desync bug: removing a row
    used to leave dic_table's keys stale, so a later add could either silently
    overwrite an unrelated model's data or KeyError."""

    def _make_main_window(self):
        main_window = SimpleNamespace()
        main_window.dic_table = {}
        main_window._next_model_id = 1
        main_window.db_table = QTableWidget()
        main_window.db_table.setColumnCount(5)
        return main_window

    def _add_file(self, main_window, path, num_params=2, score=1.23):
        with patch.object(tab4, "QFileDialog") as mock_dialog, \
             patch.object(tab4, "migrate_bo_database", return_value=[]), \
             patch.object(tab4, "load_parameter_space", return_value=pd.DataFrame(index=range(num_params))), \
             patch.object(tab4, "load_gp_models", return_value=pd.DataFrame({"Score": [score]})), \
             patch.object(tab4, "load_metadata", return_value=_metadata_df()), \
             patch.object(tab4, "load_qois", return_value=_qois_df()), \
             patch.object(tab4, "QMessageBox"):
            mock_dialog.getOpenFileNames.return_value = ([path], "")
            tab4.add_db_files_to_table(main_window)

    def test_add_three_remove_first_add_fourth_no_collision(self, qapp):
        """Load 3 files, remove the first row, then add a 4th: the new model must
        get a fresh id, not collide with (and silently overwrite) model 3's data."""
        main_window = self._make_main_window()
        self._add_file(main_window, "a.db", num_params=1, score=1.0)
        self._add_file(main_window, "b.db", num_params=2, score=2.0)
        self._add_file(main_window, "c.db", num_params=3, score=3.0)
        assert set(main_window.dic_table.keys()) == {1, 2, 3}

        # Remove the first row (model_id 1).
        remove_btn = main_window.db_table.cellWidget(0, 4)
        tab4.remove_db_row(main_window, remove_btn)
        assert set(main_window.dic_table.keys()) == {2, 3}
        assert main_window.db_table.rowCount() == 2

        # Model 3's data must be untouched.
        assert main_window.dic_table[3]["Database_path"] == "c.db"
        assert main_window.dic_table[3]["Num_Params"] == 3

        self._add_file(main_window, "d.db", num_params=4, score=4.0)
        # The new model gets id 4 -- model 3's entry must survive unmodified.
        assert set(main_window.dic_table.keys()) == {2, 3, 4}
        assert main_window.dic_table[3]["Database_path"] == "c.db"
        assert main_window.dic_table[4]["Database_path"] == "d.db"

    def test_remove_middle_row_then_add_does_not_raise_keyerror(self, qapp):
        """Regression test for the specific crash: removing a middle row used to
        leave a gap that a later add's "previous file" lookup (model_id - 1)
        could hit, raising an uncaught KeyError."""
        main_window = self._make_main_window()
        self._add_file(main_window, "a.db")
        self._add_file(main_window, "b.db")
        self._add_file(main_window, "c.db")

        # Remove the middle row (model_id 2).
        remove_btn = main_window.db_table.cellWidget(1, 4)
        tab4.remove_db_row(main_window, remove_btn)
        assert set(main_window.dic_table.keys()) == {1, 3}

        # Must not raise KeyError looking up "model_id - 1".
        self._add_file(main_window, "d.db")
        assert set(main_window.dic_table.keys()) == {1, 3, 4}

    def test_remove_button_targets_current_row_after_earlier_removal(self, qapp):
        """Regression test: a Remove button's row index must not go stale after an
        earlier row is removed (a plain captured row index would remove the wrong
        row once the table shifts rows up)."""
        main_window = self._make_main_window()
        self._add_file(main_window, "a.db")
        self._add_file(main_window, "b.db")
        self._add_file(main_window, "c.db")

        btn_a = main_window.db_table.cellWidget(0, 4)
        btn_c = main_window.db_table.cellWidget(2, 4)

        # Remove row 0 (model "a") -- rows 1,2 shift up to 0,1.
        tab4.remove_db_row(main_window, btn_a)
        assert main_window.db_table.rowCount() == 2

        # Now click the button that belongs to model "c" (now at row 1). It must
        # remove model "c", not whatever row a stale index would have pointed to.
        tab4.remove_db_row(main_window, btn_c)
        assert set(main_window.dic_table.keys()) == {2}
        assert main_window.dic_table[2]["Database_path"] == "b.db"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
