"""Offline cloud contract; deployment policy is supplied separately by the operator."""
from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import urlsplit

from .config import keys, integer, local_uri, sha, stamp, intervals, identifier, text


def gcs_uri(uri):
    if not isinstance(uri, str):
        raise ValueError('GCS URI required')
    parsed = urlsplit(uri)
    if (parsed.scheme != 'gs' or not re.fullmatch(r'[a-z0-9][a-z0-9._-]{1,221}[a-z0-9]', parsed.netloc)
            or parsed.query or parsed.fragment or not parsed.path.startswith('/')):
        raise ValueError('exact gs://bucket/object required')
    name = parsed.path[1:]
    if not name or any(p in {'', '.', '..'} for p in name.split('/')) or any(c in name for c in '\\%:\x00\r\n'):
        raise ValueError('unsafe GCS object path')
    return parsed.netloc, name


def table_id(uri):
    if not isinstance(uri, str) or not uri.startswith('bq://'):
        raise ValueError('explicit BigQuery table required')
    name = uri[5:]
    if not re.fullmatch(r'[a-z][a-z0-9-]{4,62}\.[A-Za-z_][A-Za-z0-9_]{0,1023}\.[A-Za-z_][A-Za-z0-9_]{0,1023}', name):
        raise ValueError('unsafe BigQuery table identifier')
    return name


def validate_artifact(value):
    text(value['uri'])
    if value['uri'].startswith('gs://'):
        gcs_uri(value['uri']); integer(value.get('generation'), 1, 2**63-1)
    else:
        local_uri(value['uri'])
        if 'generation' in value:
            raise ValueError('generation belongs only to GCS')


def validate_cloud(value):
    required = ('transfer_bytes', 'decoded_bytes', 'model_bytes', 'max_rows', 'max_pages', 'page_rows',
                'transport_attempts', 'per_query_bytes', 'total_query_bytes', 'snapshot_utc', 'billing_project', 'location', 'query_seconds')
    keys(value, required)
    for name in required[:9]:
        integer(value[name], 1, 2**53-1)
    integer(value['transport_attempts'], 1, 3)
    integer(value['page_rows'], 1, 10000)
    integer(value['query_seconds'], 1, 14400)
    if value['total_query_bytes'] < value['per_query_bytes']:
        raise ValueError('aggregate query cap must cover per-query cap')
    stamp(value['snapshot_utc'])
    identifier(value['billing_project']); identifier(value['location'])


def validate_source(source, cloud):
    uri = source['uri']
    if not isinstance(uri, str):
        raise ValueError('source URI required')
    if uri.startswith('bq://'):
        table_id(uri)
        if cloud is None or 'sha256' in source or 'generation' in source:
            raise ValueError('BigQuery uses explicit cloud snapshot, not a pre-extraction hash')
        bq = source.get('bigquery'); keys(bq, ('columns', 'intervals'))
        from .preparation import required_columns
        if (not isinstance(bq['columns'], list) or any(not isinstance(c, str) for c in bq['columns'])
                or len(bq['columns']) != len(set(bq['columns']))
                or set(bq['columns']) != set(required_columns(source))):
            raise ValueError('project exactly all typed source columns')
        for column in bq['columns']:
            if not isinstance(column, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,299}', column):
                raise ValueError('unsafe projected column')
        intervals(bq['intervals'])
        if not bq['intervals']:
            raise ValueError('finite support extraction intervals required')
        for metadata in source['instruments'].values():
            a, b = map(stamp, metadata['validity'])
            if not any(stamp(x) <= a and b <= stamp(y) for x, y in bq['intervals']):
                raise ValueError('extraction must cover full metadata support and gap boundaries')
    else:
        if 'bigquery' in source:
            raise ValueError('contradictory source modes')
        sha(source.get('sha256')); validate_artifact(source)


@dataclass(frozen=True)
class CloudPolicy:
    """Protected deployment allowlists and ceilings, never read from a run config."""
    read_prefixes: tuple
    output_prefixes: tuple
    tables: tuple
    max_transfer_bytes: int
    max_model_bytes: int
    max_total_query_bytes: int
    local_roots: tuple = ()

    def __post_init__(self):
        for prefixes in (self.read_prefixes, self.output_prefixes):
            for prefix in prefixes:
                gcs_uri(prefix)
        for table in self.tables:
            table_id('bq://' + table)
        for limit in (self.max_transfer_bytes, self.max_model_bytes, self.max_total_query_bytes):
            integer(limit, 1, 2**53-1)

    def allow(self, uri, *, output=False):
        bucket, name = gcs_uri(uri)
        for prefix in self.output_prefixes if output else self.read_prefixes:
            allowed_bucket, allowed_name = gcs_uri(prefix)
            if bucket == allowed_bucket and (name == allowed_name or name.startswith(allowed_name + '/')):
                return
        raise ValueError('URI outside deployment allowlist')

    def check(self, doc):
        def local(path):
            if not any(Path(path).resolve().is_relative_to(Path(root).resolve()) for root in self.local_roots):
                raise ValueError('mounted file outside deployment allowlist')
        cloud = doc['cloud']
        if (cloud['transfer_bytes'] > self.max_transfer_bytes or cloud['model_bytes'] > self.max_model_bytes
                or cloud['total_query_bytes'] > self.max_total_query_bytes):
            raise ValueError('configuration exceeds deployment ceilings')
        if doc['output']['uri'].startswith('gs://'):
            self.allow(doc['output']['uri'], output=True)
        for source in doc['sources'].values():
            if source['uri'].startswith('gs://'):
                self.allow(source['uri'])
            elif source['uri'].startswith('bq://') and table_id(source['uri']) not in self.tables:
                raise ValueError('table outside deployment allowlist')
            elif '://' not in source['uri']:
                local(source['uri'])
        for model in doc['models'].values():
            for artifact in model['artifacts']:
                if artifact['uri'].startswith('gs://'):
                    self.allow(artifact['uri'])
                else:
                    local(artifact['uri'])
