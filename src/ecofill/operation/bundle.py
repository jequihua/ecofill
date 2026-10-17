"""Local write-once publication and independent, stdlib-only integrity readback."""
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
import tempfile


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def artifact_path(root, name):
    """Canonical relative POSIX names, contained through existing parent links."""
    if (not isinstance(name, str) or not name or '\x00' in name or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute() or any(part in {'', '.', '..'} for part in name.split('/'))):
        raise ValueError('invalid artifact path')
    path = Path(root).joinpath(*name.split('/'))
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('artifact path escapes bundle')
    return path


class Writer:
    def __init__(self, root, byte_limit, *, publication='strict', scratch_limit=None, memory_limit=None):
        self.root = Path(root)
        self.limit = byte_limit
        self.bytes = 0
        self.files = []
        self.reserve = 65536
        self.publication = publication
        self.scratch_limit = byte_limit if scratch_limit is None else scratch_limit
        self.memory_limit = byte_limit * 8 if memory_limit is None else memory_limit

    def write(self, name, payload, *, terminal=False):
        path = artifact_path(self.root, name)
        if self.publication == 'single_writer':
            with self.staged() as staged:
                if isinstance(payload, (dict, list)):
                    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
                    for piece in encoder.iterencode(payload):
                        staged.write(piece.encode())
                else:
                    staged.write(payload.encode() if isinstance(payload, str) else payload)
                return self.publish_closed(name, staged, terminal=terminal)
        if isinstance(payload, dict) or isinstance(payload, list):
            payload = encoded(payload)
        if isinstance(payload, str):
            payload = payload.encode()
        if self.bytes + len(payload) > self.limit - (0 if terminal else self.reserve):
            raise BudgetStop('output_bytes')
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

    @contextmanager
    def staged(self):
        """A bounded encoder file in caller-owned local TMPDIR, never in the output root."""
        with tempfile.TemporaryDirectory(prefix='ecofill-encode-') as directory:
            path = Path(directory) / 'payload'
            if path.resolve().is_relative_to(self.root.resolve()):
                raise ValueError('encoding scratch must be separate from output')
            with path.open('w+b') as stream:
                limit, kind = min((self.scratch_limit, 'scratch_bytes'), (self.memory_limit // 2, 'memory_bytes'))
                yield LimitedFile(stream, limit, kind)

    def publish_closed(self, name, encoded_file, *, terminal=False):
        destination = artifact_path(self.root, name)
        encoded_file.flush()
        path = Path(encoded_file.name)
        encoded_file.close()  # Encoder closes before destination publication starts.
        size = path.stat().st_size
        if self.bytes + size > self.limit - (0 if terminal else self.reserve):
            raise encoding_budget('output_bytes')
        if self.publication != 'single_writer':
            return self.write(name, path.read_bytes(), terminal=terminal)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with path.open('rb') as source, destination.open('xb') as output:
            while block := source.read(1024**2):
                if output.write(block) != len(block):
                    raise OSError('short publication write')
                digest.update(block)
        if destination.stat().st_size != size:
            raise OSError('published size mismatch')
        with destination.open('rb') as check:
            if hashlib.file_digest(check, 'sha256').hexdigest() != digest.hexdigest():
                raise OSError('published hash mismatch')
        receipt = {'path': name, 'bytes': size, 'sha256': digest.hexdigest()}
        self.bytes += size
        self.files.append(receipt)
        return receipt

    def csv(self, name, frame):
        artifact_path(self.root, name)
        if self.publication == 'single_writer':
            with self.staged() as staged:
                with gzip.GzipFile(filename='', fileobj=staged, mode='wb', mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as text:
                        frame.to_csv(text, index=True, index_label='timestamp', float_format='%.17g')
                return self.publish_closed(name, staged)
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
        document = {'schema': 'ecofill.operation.manifest/1', 'files': self.files.copy()}
        if getattr(self, 'partition_index', None) is not None:
            document['datasets'] = self.partition_index
        manifest = self.write('manifest.json', document, terminal=True)
        self.write('terminal.json', {'schema': 'ecofill.operation.terminal/1', 'status': status,
                                    'manifest_sha256': manifest['sha256']}, terminal=True)


class BudgetStop(RuntimeError):
    pass


def encoding_budget(name):
    error = BudgetStop(name)
    error.diagnostic = {'code': 'encoding_exceeds_' + name, 'location': '/limits/' + name,
                        'explanation': 'closed-file encoding/publication exceeds the declared allowance; narrow rows or columns, or raise the limit within deployment policy'}
    return error


class LimitedFile:
    """File-like encoder sink that rejects growth before writing beyond local scratch."""
    def __init__(self, stream, limit, kind='scratch_bytes'):
        self.stream, self.limit, self.kind = stream, limit, kind

    def write(self, payload):
        if self.stream.tell() + len(payload) > self.limit:
            raise encoding_budget(self.kind)
        return self.stream.write(payload)

    def __getattr__(self, name):
        return getattr(self.stream, name)


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
        name = item['path']
        if name in seen:
            raise ValueError('invalid manifest path')
        seen.add(name)
        path = artifact_path(root, name)
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
