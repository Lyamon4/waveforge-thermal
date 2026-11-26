"""Archive-dependent replays are optional; private run data is not packaged."""
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    selection = Path("artifacts/inputs/development_tasks.json")
    archive = Path("artifacts/mcl_stage51_20261028")
    requirements = {
        ("test_mcl_stage12.py", "test_stage12_geometry_audit_uses_new_radius_and_preserves_fixtures"):
            [Path("artifacts/mcl_stage11_20261017/run_003/qualification.json")],
        ("test_mcl_stage4_campaign.py", "test_actual_evaluation_record_is_json_serializable"):
            [selection],
        ("test_mcl_stage51.py", "test_optimizer_records_both_grids_and_accounts_selection"):
            [selection],
        ("test_mcl_stage52.py", "test_real_stage51_control_receipts_are_compatible"):
            [archive/"manifest.json", archive/"baseline_results.json"],
        ("test_mcl_stage52.py", "test_full_report_uses_all_pairs_and_writes_stage52_identity"):
            [archive/"STAGE5_1_AUDIT/results.json"],
        ("test_mcl_stage52.py", "test_neural_only_audit_matches_sealed_audit"):
            [selection, archive/"selection_audit.json", archive/"STAGE5_1_AUDIT/results.json"],
        ("test_mcl_stage52.py", "test_neural_audit_cannot_resurrect_invalid_nominal_selection"):
            [selection, archive/"selection_audit.json", archive/"STAGE5_1_AUDIT/results.json"],
    }
    for item in items:
        paths = requirements.get((item.path.name, item.originalname), ())
        if any(not path.is_file() for path in paths):
            item.add_marker(pytest.mark.skip(reason="optional archived campaign inputs not distributed"))
