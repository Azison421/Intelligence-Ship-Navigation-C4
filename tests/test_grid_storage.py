"""Regression tests for the local Unity-grid artifact and storage guard."""

from __future__ import annotations

import json
import shutil
import uuid

import pytest

from usvlib4ros.mapping import (
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    load_sidecar_artifact,
    occupancy_grid_payload,
    write_occupancy_grid,
)
from usvlib4ros.storage import (
    PROJECT_ROOT,
    StoragePolicyError,
    resolve_project_storage,
)


ARTIFACT = PROJECT_ROOT / "usvlib4ros" / "mapping" / "data" / "beihu_static_world_sidecar.json"


@pytest.fixture(scope="module")
def compiled():
    artifact, artifact_hash = load_sidecar_artifact(ARTIFACT)
    return compile_beihu_sidecar(
        artifact,
        source_artifact_hash=artifact_hash,
        session_id="grid-test",
        config=SidecarCompilerConfig(),
    )


def test_occupancy_payload_is_ros_compatible_and_preserves_unknown(compiled):
    payload = occupancy_grid_payload(compiled)
    assert payload["schema_version"] == "navalg-occupancy-grid-v1"
    assert payload["width"] * payload["height"] == len(payload["data"])
    assert set(payload["data"]) <= {-1, 0, 100}
    assert -1 in payload["data"]
    assert payload["row_order"] == "y_index_ascending_from_origin"
    assert payload["source_artifact_hash"] == compiled.snapshot.source_artifact_hash


def test_grid_writer_is_project_local_and_deterministic(compiled, tmp_path):
    # A temporary directory outside NavAIg must be rejected before writing.
    with pytest.raises(StoragePolicyError):
        write_occupancy_grid(compiled, tmp_path)

    output = PROJECT_ROOT / "artifacts" / ("test-grid-storage-" + uuid.uuid4().hex)
    try:
        paths = write_occupancy_grid(compiled, output)
        assert set(paths) == {"json", "pgm", "yaml"}
        assert all(path.is_file() for path in paths.values())
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert payload["width"] == compiled.snapshot.width
        assert paths["pgm"].read_bytes().startswith(b"P5\n")
        assert "source_artifact_hash" in paths["yaml"].read_text(encoding="utf-8")
    finally:
        shutil.rmtree(output, ignore_errors=True)


def test_storage_rejects_remote_or_external_paths(tmp_path):
    assert resolve_project_storage("beihu", category="maps").is_relative_to(PROJECT_ROOT)
    with pytest.raises(StoragePolicyError):
        resolve_project_storage(tmp_path / "ros-artifacts", category="maps")
    with pytest.raises(StoragePolicyError):
        resolve_project_storage(r"\\ros-host\share\grid", category="maps")
