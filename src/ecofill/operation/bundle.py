"""Local write-once publication and independent, stdlib-only integrity readback."""
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class Writer:
    def __init__(self, root, byte_limit):
        self.root = Path(root)
        self.limit = byte_limit
        self.bytes = 0
        self.files = []
        self.reserve = 65536

    def write(self, name, payload, *, terminal=False):
        if isinstance(payload, dict) or isinstance(payload, list):
            payload = encoded(payload)
        if isinstance(payload, str):
            payload = payload.encode()
        if self.bytes + len(payload) > self.limit - (0 if terminal else self.reserve):
            raise BudgetStop('output_bytes')
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.publishing')
        with temporary.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        # link is an atomic, no-overwrite publication, unlike replace().
        os.link(temporary, path)
        temporary.unlink()
        self.bytes += len(payload)
        receipt = {'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        self.files.append(receipt)
        return receipt

    def csv(self, name, frame):
        text = frame.to_csv(index=True, index_label='timestamp', float_format='%.17g')
        return self.write(name, gzip.compress(text.encode(), mtime=0))

    def finish(self, status, summary):
        self.write('summary.json', summary, terminal=True)
        lines = ['# Ecofill operational result', '', f"Status: {status}. Datasets: {summary['datasets_processed']}.",
                 f"Original gaps: {summary['original_gaps']}; filled gaps: {summary['filled_gaps']}; filled cells: {summary['filled_cells']}.",
                 f"Elapsed seconds: {summary['elapsed_seconds']:.3f}; method calls: {summary['method_calls']}.",
                 'Natural-gap accuracy: unobserved.', '', 'Unresolved reasons: ' + json.dumps(summary['unresolved_reasons'], sort_keys=True),
                 'Detailed class/method counts, periods and artifacts: summary.json and manifest.json.']
        self.write('summary.md', '\n'.join(lines) + '\n', terminal=True)
        manifest = self.write('manifest.json', {'schema': 'ecofill.operation.manifest/1', 'files': self.files.copy()}, terminal=True)
        self.write('terminal.json', {'schema': 'ecofill.operation.terminal/1', 'status': status,
                                    'manifest_sha256': manifest['sha256']}, terminal=True)


class BudgetStop(RuntimeError):
    pass


def inspect_bundle(run_uri):
    """No ecofill/numpy/model imports needed by this module's reader."""
    root = Path(run_uri)
    if not (root / 'terminal.json').exists():
        return {'status': 'incomplete', 'terminal_present': False}
    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8'))
    terminal = read('terminal.json')
    if terminal.get('schema') != 'ecofill.operation.terminal/1' or terminal.get('status') not in {'complete', 'partial', 'failed'}:
        raise ValueError('invalid terminal')
    manifest_bytes = (root / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != terminal['manifest_sha256']:
        raise ValueError('terminal/manifest integrity mismatch')
    manifest = json.loads(manifest_bytes)
    if manifest['schema'] != 'ecofill.operation.manifest/1':
        raise ValueError('invalid manifest schema')
    seen = set()
    for item in manifest['files']:
        name = item['path']; rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name or name in seen:
            raise ValueError('invalid manifest path')
        seen.add(name)
        path = root.joinpath(*rel.parts)
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('manifest path escapes bundle')
        if path.stat().st_size != item['bytes']:
            raise ValueError('bundle byte mismatch')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                raise ValueError('bundle hash mismatch')
    if not {'request.json', 'config.resolved.json', 'summary.json', 'summary.md', 'provenance.json'} <= seen:
        raise ValueError('incomplete terminal manifest')
    summary = read('summary.json')
    if summary['status'] != terminal['status']:
        raise ValueError('summary/terminal status mismatch')
    return summary
