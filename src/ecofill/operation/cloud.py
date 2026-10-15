"""Bounded ADC transport around the installed runner. No inference resumption."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import time

from .bundle import Writer, BudgetStop, encoded
from .cloud_config import gcs_uri, table_id
from .config import digest, load_config, read_document, stamp
from .preparation import decode_source, read_source, required_columns


class ObjectMissing(FileNotFoundError):
    pass


class ObjectExists(RuntimeError):
    pass


class TransientIO(RuntimeError):
    pass


class Storage:
    """Direct Storage API, no ambient retry loop or transparent decompression."""
    def __init__(self):
        from google.cloud import storage
        self.client = storage.Client()

    def read(self, uri, generation, limit):
        from google.api_core import exceptions as errors
        from requests.exceptions import ConnectionError, Timeout
        bucket, name = gcs_uri(uri)
        blob = self.client.bucket(bucket).blob(name, generation=generation, chunk_size=256*1024)
        class Bounded(io.BytesIO):
            def write(self, value):
                if self.tell() + len(value) > limit:
                    raise BudgetStop('cloud transfer bytes')
                return super().write(value)
        try:
            blob.reload(timeout=60, retry=None)
            actual = int(blob.generation)
            if generation is not None and actual != generation:
                raise ValueError('GCS generation drift')
            if blob.size is None or blob.size > limit:
                raise BudgetStop('cloud object metadata size')
            stream = Bounded()
            blob.download_to_file(stream, raw_download=True, if_generation_match=actual, timeout=60, retry=None)
            return stream.getvalue(), actual
        except errors.NotFound as exc:
            raise ObjectMissing('object missing') from exc
        except errors.PreconditionFailed as exc:
            raise ValueError('GCS generation precondition failed') from exc
        except (errors.ServiceUnavailable, errors.DeadlineExceeded, errors.TooManyRequests, ConnectionError, Timeout, TimeoutError) as exc:
            raise TransientIO('GCS read outcome unknown') from exc

    def create(self, uri, payload):
        from google.api_core import exceptions as errors
        from requests.exceptions import ConnectionError, Timeout
        bucket, name = gcs_uri(uri)
        blob = self.client.bucket(bucket).blob(name)
        try:
            blob.upload_from_string(payload, if_generation_match=0, timeout=60, retry=None, checksum='crc32c')
            if blob.generation is None:
                raise TransientIO('GCS generation response absent')
            return int(blob.generation)
        except errors.PreconditionFailed as exc:
            raise ObjectExists('immutable object exists') from exc
        except (errors.ServiceUnavailable, errors.DeadlineExceeded, errors.TooManyRequests, ConnectionError, Timeout, TimeoutError) as exc:
            raise TransientIO('GCS write outcome unknown') from exc


class BigQuery:
    def __init__(self, project, location):
        from google.cloud import bigquery
        self.client = bigquery.Client(project=project, location=location)

    @staticmethod
    def config(read, dry_run=False):
        from google.cloud import bigquery
        params = [bigquery.ArrayQueryParameter(p['name'], 'STRING', p['value']) if p['type'] == 'ARRAY<STRING>'
                  else bigquery.ScalarQueryParameter(p['name'], p['type'], stamp(p['value']).to_pydatetime())
                  for p in read['parameters']]
        return bigquery.QueryJobConfig(query_parameters=params, dry_run=dry_run, use_query_cache=False,
                                      maximum_bytes_billed=read['maximum_bytes_billed'])

    def metadata(self, table):
        value = self.client.get_table(table, timeout=60, retry=None)
        return {'table': table, 'table_type': value.table_type, 'etag': value.etag,
                'schema': [field.to_api_repr() for field in value.schema]}

    def dry_run(self, read):
        job = self.client.query(read['sql'], job_config=self.config(read, True), retry=None,
                                job_retry=None, timeout=60)
        return int(job.total_bytes_processed)

    def submit(self, read):
        from google.api_core import exceptions as errors
        from requests.exceptions import ConnectionError, Timeout
        try:
            return self.client.query(read['sql'], job_config=self.config(read), job_id=read['job_id'],
                                     retry=None, job_retry=None, timeout=60)
        except (errors.DeadlineExceeded, errors.ServiceUnavailable, errors.Conflict, ConnectionError, Timeout, TimeoutError) as exc:
            raise TransientIO('BigQuery submission outcome unknown') from exc

    def get(self, read):
        return self.client.get_job(read['job_id'], retry=None, timeout=60)

    def verify(self, job, read):
        expected = self.config(read).to_api_repr()['query'] | {'query': read['sql']}
        actual = job.to_api_repr()['configuration']['query']
        if (job.job_id != read['job_id'] or any(actual.get(k) != v for k, v in expected.items())
                or job.project != self.client.project or job.location != self.client.location):
            raise ValueError('retrieved job differs from pinned query/configuration')

    def rows(self, job, cloud):
        iterator = job.result(page_size=cloud['page_rows'], timeout=cloud['query_seconds'], retry=None, job_retry=None)
        return iterator.total_rows, iterator.pages


def query_read(source, cloud, request):
    columns = required_columns(source)
    identities = sorted({name for k, m in source['instruments'].items() for name in [k, *m['aliases']]})
    params = [{'name': 'snapshot', 'type': 'TIMESTAMP', 'value': stamp(cloud['snapshot_utc']).isoformat()},
              {'name': 'identities', 'type': 'ARRAY<STRING>', 'value': identities}]
    predicates = []
    for i, (a, b) in enumerate(source['bigquery']['intervals']):
        params.extend([{'name': f'a{i}', 'type': 'TIMESTAMP', 'value': stamp(a).isoformat()},
                       {'name': f'b{i}', 'type': 'TIMESTAMP', 'value': stamp(b).isoformat()}])
        predicates.append(f"(`{source['timestamp_column']}` >= @a{i} AND `{source['timestamp_column']}` < @b{i})")
    sql = ('SELECT ' + ', '.join('`' + c + '`' for c in columns) + '\nFROM `' + table_id(source['uri']) +
           '` FOR SYSTEM_TIME AS OF @snapshot\nWHERE `' + source['identity_column'] +
           '` IN UNNEST(@identities) AND (' + ' OR '.join(predicates) + ')\nORDER BY ' +
           ', '.join('`' + source[k] + '`' for k in ('identity_column', 'timestamp_column', 'key_column')))
    read = {'sql': sql, 'parameters': params, 'table': table_id(source['uri']), 'columns': columns,
            'snapshot_utc': cloud['snapshot_utc'], 'maximum_bytes_billed': cloud['per_query_bytes']}
    read['job_id'] = 'ecofill_' + digest([request, read])
    return read


def billed_bytes(value):
    unit = 1024*1024
    if type(value) is not int or value < 0:
        raise ValueError('unknown query estimate/billing')
    return max(10*unit, ((value + unit-1)//unit)*unit)


class CloudWriter(Writer):
    def __init__(self, transport, uri, byte_limit):
        super().__init__(Path('.'), byte_limit)
        self.transport, self.uri = transport, uri

    def write(self, name, payload, *, terminal=False):
        if isinstance(payload, (dict, list)):
            if name == 'terminal.json':
                payload = payload | {'manifest_generation': self.files[-1]['generation']}
            payload = encoded(payload)
        elif isinstance(payload, str):
            payload = payload.encode()
        if self.bytes + len(payload) > self.limit - (0 if terminal else self.reserve):
            raise BudgetStop('output_bytes')
        receipt, _ = self.transport.create(self.uri + '/' + name, payload)
        receipt['path'] = name
        self.files.append(receipt); self.bytes += len(payload)
        return receipt


class CloudTransport:
    def __init__(self, policy, *, storage=None, bigquery=None, staging):
        self.policy, self.storage, self.bigquery = policy, storage, bigquery
        self.staging = Path(staging)
        self.doc, self.reads, self.writer = None, {}, None
        self.transferred, self.model_bytes, self.billed = 0, 0, 0
        self.config_bytes = 0
        self.config_origin, self.origin_config_identity = None, None

    def configure(self, doc):
        self.policy.check(doc)
        if self.origin_config_identity is not None and digest(doc) != self.origin_config_identity:
            raise ValueError('configuration differs from fetched config identity')
        self.doc = doc
        self.reads, self.source_cache, self.bq_inputs, self.writer = {}, {}, {}, None
        self.transferred, self.billed = self.config_bytes, 0
        if self.transferred > doc['cloud']['transfer_bytes']:
            raise BudgetStop('config transfer budget')
        if self.storage is None:
            self.storage = Storage()

    def fetch(self, uri, generation, sha256, limit, *, output=False):
        self.policy.allow(uri, output=output)
        if limit < 1:
            raise BudgetStop('cloud transfer budget exhausted')
        attempts = 1 if self.doc is None else self.doc['cloud']['transport_attempts']
        for i in range(attempts):
            try:
                payload, actual = self.storage.read(uri, generation, limit)
                break
            except TransientIO:
                if i+1 == attempts:
                    raise
                time.sleep(min(.1*2**i, 1))
        if len(payload) > limit or generation is not None and actual != generation:
            raise ValueError('GCS bound/generation mismatch')
        if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError('GCS hash mismatch')
        return payload, actual

    def load_config(self, uri, generation, sha256):
        self.policy.allow(uri)
        if self.storage is None:
            self.storage = Storage()
        from .config import integer, sha
        integer(generation, 1, 2**63-1); sha(sha256)
        payload, _ = self.fetch(uri, generation, sha256, min(self.policy.max_transfer_bytes, 16*1024*1024))
        # Use the same strict duplicate-key and YAML validation as offline callers.
        self.staging.mkdir(parents=True, exist_ok=True)
        path = self.staging / (sha256 + ('.yaml' if uri.endswith(('.yaml', '.yml')) else '.json'))
        if path.is_file():
            if path.read_bytes() != payload:
                raise ValueError('staged config drift')
        else:
            with path.open('xb') as stream:
                stream.write(payload)
        config = load_config(path)
        self.config_origin = {'uri': uri, 'generation': generation, 'sha256': sha256}
        self.origin_config_identity = config.identity
        self.config_bytes = len(payload); self.configure(config.to_dict())
        return config

    def create(self, uri, payload):
        self.policy.allow(uri, output=True)
        attempts = self.doc['cloud']['transport_attempts']
        for i in range(attempts):
            try:
                generation = self.storage.create(uri, payload)
                return {'generation': generation, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}, True
            except (ObjectExists, TransientIO) as exc:
                try:
                    saved, generation = self.fetch(uri, None, None, len(payload), output=True)
                except ObjectMissing:
                    if isinstance(exc, ObjectExists) or i+1 == attempts:
                        raise
                    time.sleep(min(.1*2**i, 1)); continue
                if saved != payload:
                    raise ValueError('immutable output conflict') from exc
                return {'generation': generation, 'bytes': len(saved), 'sha256': hashlib.sha256(saved).hexdigest()}, False
        raise TransientIO('write attempts exhausted')

    def claim(self, doc, run_id, request):
        uri = doc['output']['uri'] + '/runs/' + run_id
        payload = encoded(request)
        receipt, owned = self.create(uri + '/request.json', payload)
        if not owned:
            existing = self.inspect(uri)
            return None, ({'status': 'in_progress', 'run_uri': uri} if existing['status'] == 'incomplete' else existing)
        writer = CloudWriter(self, uri, doc['limits']['output_bytes'])
        writer.files.append(receipt | {'path': 'request.json'}); writer.bytes += len(payload)
        self.writer = writer
        return writer, None

    def inspect(self, uri):
        self.policy.allow(uri, output=True)
        if self.storage is None:
            self.storage = Storage()
        limit = self.policy.max_transfer_bytes
        try:
            terminal_payload, _ = self.fetch(uri + '/terminal.json', None, None, min(limit, 65536), output=True)
        except ObjectMissing:
            return {'status': 'incomplete', 'terminal_present': False}
        terminal = json.loads(terminal_payload)
        if terminal.get('schema') != 'ecofill.operation.terminal/1' or terminal.get('status') not in {'complete', 'partial', 'failed'}:
            raise ValueError('invalid terminal')
        manifest_payload, _ = self.fetch(uri + '/manifest.json', terminal['manifest_generation'],
                                         terminal['manifest_sha256'], limit, output=True)
        manifest = json.loads(manifest_payload)
        if manifest.get('schema') != 'ecofill.operation.manifest/1':
            raise ValueError('invalid manifest')
        seen, summary, total = set(), None, 0
        for item in manifest['files']:
            name = item['path']
            if name in seen or name in {'manifest.json', 'terminal.json'}:
                raise ValueError('duplicate/control manifest path')
            gcs_uri(uri + '/' + name); seen.add(name)
            total += item['bytes']
            if total > limit:
                raise BudgetStop('inspect transfer budget')
            payload, _ = self.fetch(uri + '/' + name, item['generation'], item['sha256'], item['bytes'], output=True)
            if len(payload) != item['bytes']:
                raise ValueError('manifest byte mismatch')
            if name == 'summary.json':
                summary = json.loads(payload)
        if not {'request.json', 'config.resolved.json', 'summary.json', 'summary.md', 'provenance.json'} <= seen:
            raise ValueError('incomplete terminal manifest')
        if summary['status'] != terminal['status']:
            raise ValueError('summary/terminal mismatch')
        return summary

    def preflight(self, doc, request):
        from .runner import dependency_refs
        names = sorted({ref['source'] for ref in dependency_refs(doc)})
        bq_sources = [doc['sources'][name] for name in names if doc['sources'][name]['uri'].startswith('bq://')]
        if not bq_sources:
            return
        cloud = doc['cloud']
        if self.bigquery is None:
            self.bigquery = BigQuery(cloud['billing_project'], cloud['location'])
        estimate, unique = 0, {}
        for source in bq_sources:
            key = digest(source)
            if key in self.reads:
                continue
            read = query_read(source, cloud, request)
            if read['job_id'] in unique:
                self.reads[key] = unique[read['job_id']]
                continue
            metadata = self.bigquery.metadata(read['table'])
            if metadata['table_type'] != 'TABLE':
                raise ValueError('only native BigQuery TABLE snapshot semantics supported')
            fields = {f['name']: f for f in metadata['schema']}
            if set(read['columns']) - set(fields) or any(fields[c].get('mode') == 'REPEATED' or
                    fields[c]['type'] not in {'STRING', 'TIMESTAMP', 'FLOAT', 'FLOAT64', 'INTEGER', 'INT64', 'BOOLEAN', 'BOOL', 'NUMERIC', 'BIGNUMERIC'} for c in read['columns']):
                raise ValueError('unsupported/missing BigQuery typed column')
            if fields[source['timestamp_column']]['type'] != 'TIMESTAMP':
                raise ValueError('BigQuery UTC TIMESTAMP column required')
            estimated = billed_bytes(self.bigquery.dry_run(read))
            if estimated > cloud['per_query_bytes']:
                raise BudgetStop('per-query billed-byte ceiling')
            estimate += estimated
            read.update(table_metadata=metadata, dry_run_billed_upper_bound=estimated)
            self.reads[key] = read
            unique[read['job_id']] = read
        if estimate > cloud['total_query_bytes']:
            raise BudgetStop('aggregate dry-run billed-byte ceiling')

    def read_source(self, source, limits):
        key = digest(source)
        if key not in self.source_cache:
            self.source_cache[key] = self._read_source(source, limits)
        return self.source_cache[key]

    def _read_source(self, source, limits):
        cloud = self.doc['cloud']
        if source['uri'].startswith('gs://'):
            remaining = min(cloud['transfer_bytes'] - self.transferred, limits['scratch_bytes'])
            # Simple input may omit hash/generation: the actual generation is captured and pinned,
            # and the downloaded bytes are hashed before use. Supplied expectations still apply.
            payload, generation = self.fetch(source['uri'], source.get('generation'), source.get('sha256'), remaining)
            self.transferred += len(payload)
            bounded = limits | {'scratch_bytes': min(limits['scratch_bytes'], cloud['decoded_bytes']),
                                'max_rows_per_series': min(limits['max_rows_per_series'], cloud['max_rows'])}
            result = decode_source(source, bounded, payload,
                                   compressed=source.get('compressed', source['uri'].endswith('.gz')),
                                   receipt={'uri': source['uri'], 'generation': generation})
            if result.receipt['rows'] > cloud['max_rows']:
                raise BudgetStop('GCS total source row ceiling')
            return result
        if not source['uri'].startswith('bq://'):
            return read_source(source, limits)
        read = self.reads[digest(source)]
        if read['job_id'] in self.bq_inputs:
            payload, receipt = self.bq_inputs[read['job_id']]
            return decode_source(source | {'sha256': hashlib.sha256(payload).hexdigest()}, limits, payload, receipt=receipt)
        if self.writer is not None:
            self.writer.write('queries/' + read['job_id'] + '/intent.json', read)
        try:
            job = self.bigquery.submit(read)
        except TransientIO:
            # Exactly one lookup, never a new submission. Unknown/missing stops the run.
            job = self.bigquery.get(read)
        self.bigquery.verify(job, read)
        total, pages = self.bigquery.rows(job, cloud)
        actual = int(job.total_bytes_billed)
        charged_bound = billed_bytes(actual)
        self.billed += charged_bound
        if self.writer is not None:
            self.writer.write('queries/' + read['job_id'] + '/returned.json',
                              {'job_id': read['job_id'], 'actual_billed_bytes': actual,
                               'budgeted_billed_bytes': charged_bound, 'total_rows': total})
        if charged_bound > cloud['per_query_bytes'] or self.billed > cloud['total_query_bytes']:
            raise BudgetStop('actual billed-byte ceiling')
        if total is None or total > cloud['max_rows']:
            raise BudgetStop('BigQuery total row ceiling')
        class BoundedText(io.StringIO):
            bytes_written = 0
            def write(self, value):
                self.bytes_written += len(value.encode('utf-8'))
                if self.bytes_written > min(cloud['decoded_bytes'], limits['scratch_bytes']):
                    raise BudgetStop('BigQuery decoded byte ceiling')
                return super().write(value)
        text = BoundedText(); writer = csv.writer(text, lineterminator='\n'); writer.writerow(read['columns'])
        count, page_count, page_receipts, previous = 0, 0, [], None
        for page in pages:
            page_count += 1
            if page_count > cloud['max_pages']:
                raise BudgetStop('BigQuery page ceiling')
            typed = []
            for row in page:
                values = [row[c] for c in read['columns']]
                lexical = [v.isoformat() if hasattr(v, 'isoformat') else '' if v is None else str(v) for v in values]
                mapping = dict(zip(read['columns'], lexical))
                order = (mapping[source['identity_column']], stamp(mapping[source['timestamp_column']]), mapping[source['key_column']])
                if previous is not None and order <= previous:
                    raise ValueError('BigQuery ordering/key duplicate drift')
                previous = order; writer.writerow(lexical); typed.append(lexical); count += 1
                if count > cloud['max_rows']:
                    raise BudgetStop('BigQuery decoded row/byte ceiling')
            payload = encoded({'columns': read['columns'], 'rows': typed})
            if len(typed) > cloud['page_rows'] or self.transferred + len(payload) > cloud['transfer_bytes']:
                raise BudgetStop('BigQuery page/transfer ceiling')
            self.transferred += len(payload)
            page_receipts.append({'rows': len(typed), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
            if self.writer is not None:
                self.writer.write(f"queries/{read['job_id']}/page-{page_count}.json", payload)
        if count != total:
            raise ValueError('BigQuery pagination incomplete')
        payload = text.getvalue().encode()
        receipt = read | {'total_rows': total, 'pages': page_receipts, 'actual_billed_bytes': actual,
                          'budgeted_billed_bytes': charged_bound,
                          'typed_csv': 'UTF-8; null empty; booleans True/False; numbers exact lexical; timestamp UTC'}
        self.bq_inputs[read['job_id']] = (payload, receipt)
        return decode_source(source | {'sha256': hashlib.sha256(payload).hexdigest()}, limits, payload, receipt=receipt)

    def backend_for(self, recipe, doc, supplied, cache):
        from .runner import backend_for
        name = recipe.get('backend')
        if name is None or name in supplied:
            return backend_for(recipe, doc, supplied, cache)
        staged = deepcopy(doc)
        model = staged['models'][name]
        for i, artifact in enumerate(model['artifacts']):
            if not artifact['uri'].startswith('gs://'):
                continue
            filename = gcs_uri(artifact['uri'])[1].split('/')[-1]
            if filename not in {'tabpfn-v3.5-20260909.safetensors', 'model.safetensors', 'config.json', 'LICENSE'}:
                raise ValueError('unsupported model artifact filename')
            path = self.staging / 'models' / name / digest(doc['models'][name])[:16] / filename
            if not path.is_file():
                bound = self.doc['cloud']['model_bytes'] - self.model_bytes
                try:
                    payload, _ = self.fetch(artifact['uri'], artifact['generation'], artifact['sha256'], bound)
                except ObjectMissing:
                    return None
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(payload)
                self.model_bytes += len(payload)
            artifact['uri'] = str(path)
            del artifact['generation']
        return backend_for(recipe, staged, supplied, cache)
