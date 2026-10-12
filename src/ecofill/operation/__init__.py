"""Installed local operational API; no experiment imports or automatic search."""
from .config import OperationConfig, load_config, validate


def plan(config, *, output=None, transport=None):
    from .runner import make_operation_plan
    if transport is not None:
        config = load_config(config)
        transport.configure(config.to_dict())
        transport.preflight(config.to_dict(), {'config_identity': config.identity, 'mode': 'plan'})
    return make_operation_plan(config, output=output, source_reader=None if transport is None else transport.read_source)


def run(config, *, run_id, backends=None, transport=None, config_origin=None):
    from .runner import run_operation
    return run_operation(config, run_id=run_id, backends=backends, transport=transport, config_origin=config_origin)


def inspect(run_uri, *, transport=None):
    if str(run_uri).startswith('gs://'):
        if transport is None:
            raise ValueError('cloud inspect requires deployment transport')
        return transport.inspect(run_uri)
    from .bundle import inspect_bundle
    return inspect_bundle(run_uri)
