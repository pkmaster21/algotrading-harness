"""The profile content hash across model changes: profiles frozen by older code keep
loading, and a new field cannot be added to what a profile holds without a decision."""

from __future__ import annotations

import hashlib
import json
import shutil
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from harness.profiles import (
    HASH_OMITTED_AT,
    FrozenProfile,
    ProfileTampered,
    _canonical,
    _drop_at,
    load_profile,
)
from tests.test_profiles import Rig

LEGACY = Path(__file__).parent / "legacy_profiles"
FIELDS_FILE = Path(__file__).parent / "profile_fields.json"

# Frozen with each commit's own code and test rig (a plumbing promotion), files untouched.
LEGACY_PROFILES = {
    "plumbing-6cbe5ca9": "7239e16, before #57 added universe.membership",
    "plumbing-2fcd05b1": "d0b77be, #57: membership hashed as null",
}


def _model_classes(annotation: Any) -> list[type[BaseModel]]:
    """The pydantic models inside an annotation (Optional, unions, tuples, lists)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type[BaseModel]] = []
    for arg in typing.get_args(annotation):
        found.extend(_model_classes(arg))
    return found


def _field_paths(model: type[BaseModel], prefix: str = "") -> dict[str, Any]:
    """Every field path under `model`, dotted, each with its default.
    Items of a tuple or list of models are walked as `<field>[]`."""
    paths: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        if info.is_required():
            paths[path] = "<required>"
        else:  # as JSON, so it compares with HASH_OMITTED_AT and the stored profiles
            paths[path] = to_jsonable_python(info.get_default(call_default_factory=True))
        origin = typing.get_origin(info.annotation)
        many = origin in (tuple, list) or (
            origin in (typing.Union, types.UnionType)
            and any(typing.get_origin(a) in (tuple, list) for a in typing.get_args(info.annotation))
        )
        for cls in _model_classes(info.annotation):
            paths.update(_field_paths(cls, f"{path}[]." if many else f"{path}."))
    return paths


def _current() -> dict[str, Any]:
    return {"fields": _field_paths(FrozenProfile), "hash_omitted_at": HASH_OMITTED_AT}


def test_the_profile_fields_match_the_pinned_list() -> None:
    """A field added to what a profile holds changes the hash of every frozen profile,
    unless it is listed in HASH_OMITTED_AT. Adding it here is that decision. To update
    the pin after deciding: `uv run python -m tests.test_profile_hash`."""
    pinned = json.loads(FIELDS_FILE.read_text())
    assert _current() == pinned, (
        "the fields a profile holds, their defaults, or the HASH_OMITTED_AT values differ "
        f"from {FIELDS_FILE.name}. A new field: list it in HASH_OMITTED_AT (or decide that "
        "every frozen profile should break), then rewrite the pin. A changed default of a "
        "field not in HASH_OMITTED_AT is safe (frozen profiles store that field), so just "
        "rewrite the pin. A changed default of a listed field is not: it changes what old "
        "profiles load with."
    )


def test_each_omitted_field_is_omitted_at_its_default() -> None:
    fields = _field_paths(FrozenProfile)
    for path, omitted in HASH_OMITTED_AT.items():
        assert path in fields, f"{path} is not a profile field"
        assert fields[path] != "<required>", f"{path} is required, so never at a default"
        assert fields[path] == omitted, f"{path}: default {fields[path]!r}, omitted at {omitted!r}"


@pytest.mark.parametrize("profile_id", sorted(LEGACY_PROFILES))
def test_a_profile_frozen_by_older_code_still_loads(profile_id: str) -> None:
    profile = load_profile(profile_id, profiles_dir=LEGACY)
    stored = json.loads((LEGACY / profile_id / "profile.json").read_text())
    assert profile.content_hash == stored["content_hash"]
    profile.verify()


@pytest.mark.parametrize("profile_id", sorted(LEGACY_PROFILES))
def test_an_edited_older_profile_is_still_refused(profile_id: str, tmp_path: Path) -> None:
    shutil.copytree(LEGACY / profile_id, tmp_path / profile_id)
    path = tmp_path / profile_id / "profile.json"
    data = json.loads(path.read_text())
    data["recipe"]["backtest"]["starting_cash"] += 1
    path.chmod(0o644)  # profiles are written read-only
    path.write_text(json.dumps(data))
    with pytest.raises(ProfileTampered, match="does not verify"):
        load_profile(profile_id, profiles_dir=tmp_path)


def test_a_new_profile_is_hashed_without_its_omitted_fields(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdout=False)
    profile = rig.promote("plumbing", plumbing=True)
    assert profile.recipe.universe.membership is None
    payload = profile.model_dump(mode="json", exclude={"profile_id", "content_hash"})
    for path, omitted in HASH_OMITTED_AT.items():
        _drop_at(payload, path, omitted)
    assert "membership" not in payload["recipe"]["universe"]
    assert profile.content_hash == hashlib.sha256(_canonical(payload)).hexdigest()
    assert rig.load(profile.profile_id) == profile


def test_drop_at_removes_a_nested_key_only_at_the_omitted_value() -> None:
    payload: dict[str, Any] = {"recipe": {"universe": {"membership": None, "name": "x"}}}
    _drop_at(payload, "recipe.universe.membership", None)
    assert payload == {"recipe": {"universe": {"name": "x"}}}

    kept: dict[str, Any] = {"recipe": {"universe": {"membership": {"file": "m.csv"}}}}
    _drop_at(kept, "recipe.universe.membership", None)
    assert kept == {"recipe": {"universe": {"membership": {"file": "m.csv"}}}}

    missing: dict[str, Any] = {"recipe": {"dates": {}}}
    _drop_at(missing, "recipe.universe.membership", None)
    assert missing == {"recipe": {"dates": {}}}


def test_drop_at_applies_a_list_path_to_each_item() -> None:
    payload: dict[str, Any] = {"a": {"b": [{"c": None, "d": 1}, {"c": 2}, {"d": 3}]}}
    _drop_at(payload, "a.b[].c", None)
    assert payload == {"a": {"b": [{"d": 1}, {"c": 2}, {"d": 3}]}}

    empty: dict[str, Any] = {"a": {"b": None}}  # an optional list that is not set
    _drop_at(empty, "a.b[].c", None)
    assert empty == {"a": {"b": None}}


if __name__ == "__main__":  # rewrite the pin, after deciding on any new field
    FIELDS_FILE.write_text(json.dumps(_current(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {FIELDS_FILE}")
