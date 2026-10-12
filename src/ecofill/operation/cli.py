"""Installed ecofill command, with no ambient backend or experiment selection."""
import argparse
import json
import os
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(prog='ecofill')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'plan', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        if name == 'plan':
            command.add_argument('--output', required=True)
        if name == 'run':
            command.add_argument('--run-id', required=True)
        if name != 'validate':
            command.add_argument('--config-generation', type=int)
            command.add_argument('--config-sha256')
    commands.add_parser('inspect').add_argument('--run-uri', required=True)
    export = commands.add_parser('export')
    export.add_argument('--config', required=True)
    export.add_argument('--destination', required=True)
    export.add_argument('--input-root', required=True)
    export.add_argument('--profile', required=True)
    export.add_argument('--artifact-manifest')
    export.add_argument('--model-directory')
    args = parser.parse_args(argv)
    from . import load_config, plan, run, inspect
    with tempfile.TemporaryDirectory(prefix='ecofill-cloud-') as staging:
        try:
            transport, origin = None, None
            if args.command in {'plan', 'run', 'inspect'}:
                remote_config = getattr(args, 'config', '').startswith('gs://')
                remote_output = getattr(args, 'run_uri', '').startswith('gs://')
                if remote_config or remote_output or os.environ.get('ECOFILL_CLOUD_POLICY'):
                    from .cloud import CloudTransport
                    from .cloud_config import CloudPolicy
                    from .config import read_document
                    policy = CloudPolicy(**read_document(os.environ['ECOFILL_CLOUD_POLICY']))
                    transport = CloudTransport(policy, staging=staging)
                if remote_config:
                    origin = {'uri': args.config, 'generation': args.config_generation, 'sha256': args.config_sha256}
                    config = transport.load_config(args.config, args.config_generation, args.config_sha256)
                else:
                    if getattr(args, 'config_generation', None) is not None or getattr(args, 'config_sha256', None) is not None:
                        raise ValueError('config generation/hash requires GCS config URI')
                    config = getattr(args, 'config', None)
            if args.command == 'validate':
                config = load_config(args.config)
                result = {'status': 'valid', 'config_identity': config.identity, 'resolved': config.to_dict()}
            elif args.command == 'plan':
                result = plan(config, output=args.output, transport=transport).to_dict()
            elif args.command == 'run':
                result = run(config, run_id=args.run_id, transport=transport, config_origin=origin)
            elif args.command == 'export':
                from .deployment import export_profile
                from .config import read_document
                result = export_profile(args.config, args.destination, input_root=args.input_root, profile=args.profile,
                                        artifact_paths=None if args.artifact_manifest is None else read_document(args.artifact_manifest),
                                        model_directory=args.model_directory)
            else:
                result = inspect(args.run_uri, transport=transport)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return {'failed': 1, 'partial': 2, 'in_progress': 3, 'incomplete': 3}.get(result.get('status'), 0)
        except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
            # Exception text can contain private paths; keep diagnostics typed.
            print(json.dumps({'status': 'error', 'error_type': type(exc).__name__}))
            return 1

if __name__ == '__main__':
    raise SystemExit(main())
