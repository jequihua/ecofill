"""Bounded local Parquet decoding and deterministic typed tables; Arrow is optional."""
import base64
import hashlib
import math
import struct
import sys

from .config import diagnostic, pointer

FORMAT = 'parquet'
ROW_GROUP = 4096
WRITER = {'compression': 'snappy', 'version': '2.6', 'data_page_version': '1.0',
          'use_dictionary': False, 'write_statistics': True, 'coerce_timestamps': 'us',
          'allow_truncated_timestamps': False, 'data_page_size': 65536, 'write_batch_size': 1024}


def arrow(where='/output/format'):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise diagnostic('parquet_dependency_missing', where,
                         'Parquet requires the optional extra: install ecofill[parquet] (pyarrow==25.0.1)') from None
    return pa, pq


def input_format(uri, declared=None, where='/format'):
    """No I/O or imports; retain legacy implicit CSV behavior."""
    suffix = uri.lower()
    if suffix.endswith(('.parquet.gz', '.parquet.gzip')):
        raise diagnostic('unsupported_format', where, 'Parquet has internal compression; supply a single .parquet file')
    value = declared if declared is not None else ('csv' if uri.startswith('bq://') else
        'parquet' if suffix.endswith('.parquet') else 'csv-gzip' if suffix.endswith('.gz') else 'csv')
    if not isinstance(value, str) or value not in {'csv', 'csv-gzip', 'parquet'}:
        raise diagnostic('unsupported_format', where, 'format must be csv, csv-gzip or parquet')
    if value == 'parquet' and '://' in uri:
        raise diagnostic('parquet_local_only', where, 'Parquet accepts one local file; materialize remote data upstream')
    if value == 'parquet' and any(char in uri for char in ('*', '?', '[')):
        raise diagnostic('single_parquet_file_required', where, 'supply one literal local file, without wildcard or directory discovery')
    if uri.startswith('bq://') and value != 'csv':
        raise diagnostic('unsupported_format', where, 'direct BigQuery uses the existing typed CSV snapshot transport')
    return value


def safe_footer(payload, ceiling):
    """Bound footer before Arrow and reject serialized extension handlers before deserialization."""
    if len(payload) < 12 or payload[:4] != b'PAR1' or payload[-4:] != b'PAR1':
        raise ValueError('invalid Parquet envelope')
    size = struct.unpack('<I', payload[-8:-4])[0]
    if size > ceiling or size > len(payload) - 12:
        raise ValueError('Parquet footer exceeds metadata limit')
    footer = payload[-8-size:-8]
    # Parquet KeyValue uses compact-Thrift string fields 1 and 2. Inspect only
    # the bounded, serialized IPC schema bytes, without invoking Arrow handlers.
    marker = b'\x18\x0cARROW:schema\x18'
    if b'ARROW:schema' in footer:
        if footer.count(b'ARROW:schema') != 1:
            raise ValueError('duplicate serialized Arrow schemas are unsupported')
        start = footer.find(marker)
        if start < 0:
            raise ValueError('unsupported serialized Arrow schema metadata')
        offset, length, shift = start + len(marker), 0, 0
        while True:
            if offset >= len(footer) or shift > 35:
                raise ValueError('invalid Arrow schema metadata')
            byte = footer[offset]; offset += 1
            length |= (byte & 127) << shift
            if not byte & 128:
                break
            shift += 7
        if length > ceiling or offset + length > len(footer):
            raise ValueError('Arrow schema metadata exceeds limit')
        schema = base64.b64decode(footer[offset:offset+length], validate=True)
        if b'ARROW:extension:' in schema:
            raise ValueError('Arrow extension columns are unsupported; export plain physical types')
    return size


def decode(source, limits, payload, receipt=None):
    from .preparation import SourceInput, bound, required_columns, source_name
    from .simple import MAX_SERIES
    where = pointer('datasets', source_name(source), 'format')
    pa, pq = arrow(where)
    identity = hashlib.sha256(payload).hexdigest()
    if source.get('sha256', identity) != identity:
        raise diagnostic('source_hash_mismatch', pointer('datasets', source_name(source), 'sha256'),
                         'input bytes differ from the expected sha256; nothing was executed')
    memory, scratch = limits['memory_bytes'], limits['scratch_bytes']
    metadata_limit = min(1024**2, scratch, max(1, memory // 8))
    try:
        safe_footer(payload, metadata_limit)
        reader = pq.ParquetFile(pa.BufferReader(payload), pre_buffer=False, memory_map=False,
                               thrift_string_size_limit=metadata_limit,
                               thrift_container_size_limit=min(65536, metadata_limit),
                               arrow_extensions_enabled=False, page_checksum_verification=True)
    except Exception as exc:
        raise diagnostic('invalid_parquet', where,
                         'invalid, oversized or extension-bearing Parquet metadata; export a bounded plain Parquet file') from exc
    schema = reader.schema_arrow
    columns = schema.names
    if len(columns) != len(set(columns)):
        raise diagnostic('invalid_parquet', where, 'Parquet requires unique top-level column names')
    selected = required_columns(source)
    if source['identity_column'] is None and 'station_id' in columns and 'station_id' not in selected:
        selected.append('station_id')
    absent = set(selected) - set(columns)
    if absent:
        raise diagnostic('missing_column', pointer('datasets', source_name(source), 'columns', min(absent)),
                         'configured column absent from Parquet schema; check selected column names')
    timestamp = source['timestamp_column']
    ids = {source['identity_column'], source['key_column'], 'station_id'} - {None}
    measures = {f['column'] for m in source['instruments'].values() for f in m['fields'].values()}
    types = {}
    for name in selected:
        kind = schema.field(name).type
        if pa.types.is_dictionary(kind):
            kind = kind.value_type
        types[name] = str(kind)
        scalar = pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind)
        text = pa.types.is_string(kind) or pa.types.is_large_string(kind)
        valid = scalar or text or pa.types.is_boolean(kind) or pa.types.is_null(kind)
        if name == timestamp:
            valid = text or pa.types.is_timestamp(kind)
        elif name in ids:
            valid = text or pa.types.is_integer(kind) or pa.types.is_null(kind)
        if name in measures:
            valid = valid and not pa.types.is_boolean(kind) and not pa.types.is_timestamp(kind)
        if not valid or isinstance(kind, pa.ExtensionType):
            raise diagnostic('unsupported_parquet_type', pointer('datasets', source_name(source), 'columns', name),
                             'selected type is unsupported; use flat numeric measurements, timestamp/string clocks and exact string/integer keys')
    series = MAX_SERIES if '*' in source['instruments'] else max(1, len(source['instruments']))
    max_rows = limits['max_rows_per_series'] * series
    metadata = reader.metadata
    if metadata.num_rows > max_rows or metadata.num_row_groups > min(max_rows, 65536):
        raise bound('source_rows_exceed_limit', 'max_rows_per_series', 'Parquet metadata exceeds row/group bound',
                    'select fewer series/rows upstream or raise limits.max_rows_per_series')
    compressed = decoded = 0
    for group in range(metadata.num_row_groups):
        row = metadata.row_group(group)
        for number in range(row.num_columns):
            column = row.column(number)
            if column.path_in_schema in selected:
                if column.total_uncompressed_size < 0 or column.total_compressed_size < 0:
                    raise diagnostic('invalid_parquet', where, 'invalid Parquet column size metadata')
                compressed += column.total_compressed_size
                decoded += column.total_uncompressed_size
    if compressed > len(payload):
        raise diagnostic('invalid_parquet', where, 'projected compressed column sizes exceed captured file bytes')
    estimate = len(payload) * 2 + decoded * 8 + metadata.num_rows * (128 + 96 * len(selected))
    if estimate > memory or decoded > scratch:
        raise bound('parquet_decoded_limit', 'memory_bytes' if estimate > memory else 'scratch_bytes',
                    'projected Parquet decoded-memory estimate exceeds the limit',
                    'narrow rows/selected columns upstream or raise the declared limit')
    rows, accumulated = [], len(payload) * 2
    try:
        for batch in reader.iter_batches(batch_size=1024, columns=selected, use_threads=False, use_pandas_metadata=False):
            if batch.nbytes * 8 + accumulated > memory or batch.nbytes > scratch:
                raise bound('parquet_batch_limit', 'memory_bytes', 'Parquet batch exceeds decoded memory limit', 'narrow the input')
            clock = batch.column(batch.schema.get_field_index(timestamp))
            if pa.types.is_dictionary(clock.type):
                clock = clock.dictionary_decode()
            clock_type = clock.type
            if pa.types.is_timestamp(clock_type) and clock_type.unit == 'ns':
                if any(v is not None and v % 1000 for v in clock.cast(pa.int64()).to_pylist()):
                    raise diagnostic('timestamp_precision_loss', pointer('datasets', source_name(source), 'timestamp'),
                                     'nanosecond timestamps must be exactly representable in microseconds; no truncation is allowed')
            for row in batch.to_pylist():
                accumulated += sys.getsizeof(row) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in row.items())
                if accumulated > memory or len(rows) >= max_rows:
                    raise bound('parquet_accumulation_limit', 'memory_bytes', 'typed Python rows exceed the memory/row limit', 'narrow the input')
                if row[timestamp] is None:
                    raise diagnostic('null_timestamp', pointer('datasets', source_name(source), 'timestamp'), 'timestamp cannot be null')
                if source['key_column'] is not None and row[source['key_column']] is None:
                    raise diagnostic('null_source_key', pointer('datasets', source_name(source), 'key'), 'configured source keys cannot be null')
                rows.append(row)
    except Exception as exc:
        if getattr(exc, 'diagnostic', None) is not None:
            raise
        raise diagnostic('invalid_parquet', where, 'Parquet column decoding failed; export a valid bounded plain file') from exc
    return SourceInput(payload, tuple(selected), rows, {'sha256': identity, 'bytes': len(payload), 'rows': len(rows),
        'columns': selected, 'source_policy': source['policy'], 'preparation_profile': source['preparation_profile'],
        'clock': 'UTC', 'format': 'parquet', 'decoding': {'arrow': pa.__version__, 'types': types,
        'metadata_limit_bytes': metadata_limit, 'projected_compressed_bytes': compressed,
        'projected_uncompressed_bytes': decoded, 'estimated_memory_bytes': estimate,
        'retained_memory_bytes': max(estimate, accumulated),
        'decimal': 'finite float64 conversion; original typed bytes retained', 'timestamp': 'lossless microseconds; UTC normalization'},
        'sort': 'stable ascending UTC; original typed bytes and row order retained separately'} | (receipt or {}))


def measurement(value, where):
    """Explicit conversion; keys never pass through this path."""
    from decimal import Decimal, InvalidOperation
    if isinstance(value, str):
        # The caller has already applied missing/nonfinite tokens and NUMBER.
        # Decimal retains integral identity without expanding an exponent to int.
        try:
            value = Decimal(value)
        except InvalidOperation:
            raise diagnostic('numeric_overflow', where,
                             'numeric exponent exceeds supported conversion; export a finite float64 measurement') from None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise diagnostic('unsupported_parquet_type', where, 'measurement must be numeric or an existing supported numeric string')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise diagnostic('numeric_overflow', where, 'measurement overflows float64; export a finite float64 measurement') from None
    if isinstance(value, Decimal) and value.is_finite():
        if not math.isfinite(result):
            raise diagnostic('numeric_overflow', where, 'finite measurement overflows float64; export a finite float64 measurement')
        if value == value.to_integral_value() and Decimal.from_float(result) != value:
            raise diagnostic('lossy_integer_measurement', where,
                             'integral measurement cannot be represented exactly in float64; retain it as a key or export an explicit converted measurement')
    if isinstance(value, int) and (not math.isfinite(result) or int(result) != value):
        raise diagnostic('lossy_integer_measurement', where, 'integer measurement cannot be represented exactly in float64; retain it as a key or export an explicit converted measurement')
    return result


def write_frame(writer, name, frame, kind):
    pa, pq = arrow()
    if len(frame) * max(1, len(frame.columns)) * 64 > writer.memory_limit:
        from .bundle import encoding_budget
        raise encoding_budget('memory_bytes')
    dtype = {'numeric': pa.float64(), 'mask': pa.bool_(), 'text': pa.string()}[kind]
    arrays = [pa.array(frame.index.to_pydatetime(), type=pa.timestamp('us', tz='UTC'))]
    for column in frame:
        arrays.append(pa.array(frame[column].tolist(), type=dtype, from_pandas=True))
    table = pa.Table.from_arrays(arrays, schema=pa.schema([pa.field('timestamp', pa.timestamp('us', tz='UTC'))] +
                                [pa.field(str(n), dtype) for n in frame.columns]))
    with writer.staged() as target:
        pq.write_table(table, target, row_group_size=ROW_GROUP, **WRITER)
        target.flush()
        target.seek(0)
        return writer.publish_closed(name, target)
