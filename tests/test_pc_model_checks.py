#!/usr/bin/env python
"""Tests for PhysiCell_Model's configuration checks: unknown .ini keys, a missing
rules file, rulesFile_ref without parameters_rules, and unmapped parameter names."""

import os

import pytest

from uq_physicell import PhysiCell_Model

XML = """<PhysiCell_settings>
  <parallel><omp_num_threads>1</omp_num_threads></parallel>
  <cell_definitions><cell_definition name="default"><phenotype>
    <death><model name="apoptosis"><death_rate>5e-5</death_rate></model></death>
  </phenotype></cell_definition></cell_definitions>
  <cell_rules><rulesets>
    <ruleset protocol="CBHG" version="3.0" format="csv" enabled="{enabled}">
      <folder>./config</folder><filename>{rules_file}</filename>
    </ruleset>
  </rulesets></cell_rules>
</PhysiCell_settings>
"""
APOPTOSIS = ".//cell_definitions/cell_definition[@name='default']/phenotype/death/model[@name='apoptosis']/death_rate"
SAT = "default,oxygen,increases,cycle entry,saturation"


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """A PhysiCell project layout: executable next to config/ (PhysiCell's working dir)."""
    (tmp_path / "config").mkdir()
    (tmp_path / "project").write_text("")  # placeholder executable, never run
    (tmp_path / "config" / "rules.csv").write_text("default,oxygen,increases,cycle entry,7.2e-04,21.5,4,0\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(model_dir, section_body, rules_file="rules.csv", enabled="true"):
    (model_dir / "config" / "settings.xml").write_text(XML.format(rules_file=rules_file, enabled=enabled))
    (model_dir / "uq.ini").write_text(
        "[M]\nexecutable = ./project\nconfigFile_ref = config/settings.xml\nnumReplicates = 1\n" + section_body
    )
    return "uq.ini"


def test_valid_config_loads(model_dir):
    ini = _write(model_dir, f'parameters = {{"{APOPTOSIS}": [None, "apoptosis_rate"]}}\n'
                            f'rulesFile_ref = config/rules.csv\nparameters_rules = {{"{SAT}": [None, "sat"]}}\n')
    model = PhysiCell_Model(ini, "M")
    assert model.parameters_rules_variable == {SAT: "sat"}


def test_swapped_key_name_is_rejected_with_suggestion(model_dir):
    # ex11's typo: rules_parameters was silently ignored, so the rules never changed
    ini = _write(model_dir, f'rulesFile_ref = config/rules.csv\nrules_parameters = {{"{SAT}": [None, "sat"]}}\n')
    with pytest.raises(ValueError, match="'rules_parameters' \\(did you mean 'parameters_rules'\\?\\)"):
        PhysiCell_Model(ini, "M")


def test_misspelled_key_is_rejected_with_suggestion(model_dir):
    ini = _write(model_dir, "numReplicate = 3\n")
    with pytest.raises(ValueError, match="did you mean 'numreplicates'"):
        PhysiCell_Model(ini, "M")


def test_missing_section_is_reported(model_dir):
    ini = _write(model_dir, "")
    with pytest.raises(ValueError, match=r"section \[Other\] not found"):
        PhysiCell_Model(ini, "Other")


def test_missing_rules_file_is_rejected(model_dir):
    # The XML's ruleset is resolved from the executable's folder; a missing file would mean no rules
    ini = _write(model_dir, "", rules_file="cell_rules_pressure.csv")
    with pytest.raises(ValueError, match="would run without any rules"):
        PhysiCell_Model(ini, "M")


def test_disabled_ruleset_is_not_checked(model_dir):
    ini = _write(model_dir, "", rules_file="does_not_exist.csv", enabled="false")
    PhysiCell_Model(ini, "M")


def test_existing_xml_rules_file_is_accepted(model_dir):
    ini = _write(model_dir, "")
    PhysiCell_Model(ini, "M")


def test_rules_ref_without_parameters_rules_is_rejected(model_dir):
    ini = _write(model_dir, "rulesFile_ref = config/rules.csv\n")
    with pytest.raises(ValueError, match="rules file would not be used"):
        PhysiCell_Model(ini, "M")


def test_check_parameter_names(model_dir):
    ini = _write(model_dir, f'parameters = {{"{APOPTOSIS}": [None, "apoptosis_rate"]}}\n'
                            f'rulesFile_ref = config/rules.csv\nparameters_rules = {{"{SAT}": [None, "sat"]}}\n')
    model = PhysiCell_Model(ini, "M")
    model.check_parameter_names(["apoptosis_rate", "sat"])  # XML and rules parameters both map
    with pytest.raises(ValueError, match=r"\['hfm'\] .* would never reach PhysiCell"):
        model.check_parameter_names(["apoptosis_rate", "sat", "hfm"], context="Prior parameters")
