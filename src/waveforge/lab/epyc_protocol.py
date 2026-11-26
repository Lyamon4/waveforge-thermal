"""Prospectively locked EPYC-native protocol; never mutate old lab contracts."""

from pathlib import Path

import yaml

from .contracts import StackSpec, content_hash

# Canonical hash locked before any new development results are produced.
CONFIG_SHA256 = "31cd436d4d4138e9686939e997904f291ed115b11005433108c702ff31b75361"
CONFIG_PATH = Path(__file__).parents[3] / "configs/epyc_native_unet_v1.yaml"


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def validate_protocol(protocol: dict) -> dict:
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    raw = {
        k: v
        for k, v in protocol.items()
        if k not in ("protocol_sha256", "stack_sha256")
    }
    if content_hash(raw) != CONFIG_SHA256:
        raise ValueError("protocol differs from locked scope, values, keys, or types")
    expected = dict(
        raw, protocol_sha256=CONFIG_SHA256, stack_sha256=StackSpec().fingerprint()
    )
    if any(
        protocol.get(k, expected[k]) != expected[k]
        for k in ("protocol_sha256", "stack_sha256")
    ):
        raise ValueError("protocol or stack hash mismatch")
    return expected


def load_protocol(path: Path = CONFIG_PATH) -> dict:
    try:
        raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader)
        return validate_protocol(raw)
    except (TypeError, yaml.YAMLError) as exc:
        raise ValueError("invalid protocol") from exc
