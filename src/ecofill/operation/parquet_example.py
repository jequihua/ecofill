"""Synthetic local Parquet fixture and independent Arrow/stdlib result reader.

Execute this installed file directly. Readback imports no ecofill or model code.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

MISSING = {0, 1, 100, 101, 102, 200, 201, 202, 203, *range(250, 256)}
FILLED = {100, 101, 102, 150, 200, 201, 202, 203}
START = datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def prepare(root, container=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    (root/'inputs').mkdir();(root/'output').mkdir();(root/'output').chmod(0o777)
    stamps=[START+timedelta(minutes=30*i) for i in range(300)]
    def source(grouped):
        rows=[]
        for station in ('001','002') if grouped else ('001',):
            for i in range(300):
                if i==150:continue
                rows.append({'timestamp':stamps[i], 'bar':None if i in MISSING else 1000. if grouped else 1000.+.5*i,
                             'station_id':station,'row_key':2**53+1+i+(1000 if station=='002' else 0) if grouped else f'{i:06d}',
                             'qc_bar':True,'unused_nested':[i]})
        return pa.Table.from_pylist(rows)
    for name,table in [('weather',source(False)),('stations',source(True)),
                       ('donors',pa.table({'timestamp':stamps,'d1':list(map(float,range(300))),
                                          'd2':[float(i%17) for i in range(300)]}))]:
        pq.write_table(table,root/'inputs'/f'{name}.parquet',compression='snappy',row_group_size=64)
    prefix='/inputs/' if container else 'inputs/'
    output='/output/' if container else 'output/'
    weather={'uri':prefix+'weather.parquet','cadence':'30min','key':'row_key','targets':['bar'],'method':'LIN','max_gap':'2h'}
    doc={'datasets':{'weather':weather,'stations':{'uri':prefix+'stations.parquet','cadence':'30min','key':'row_key',
        'group_by':'station_id','targets':['bar'],'method':'RF-X','max_gap':'2h',
        'predictors':[{'dataset':'donors','column':'d1'},{'dataset':'donors','column':'d2'}],
        'parameters':{'n_estimators':4,'seed':13}},
        'donors':{'uri':prefix+'donors.parquet','cadence':'30min','targets':[]}},
        'output':{'uri':output+'single','format':'parquet','publication':'single_writer'}}
    save(root/'inputs/config.json',doc)
    strict=deepcopy(doc);strict['output'].update(uri=output+'strict',publication='strict')
    save(root/'inputs/config-strict.json',strict)
    linear={'datasets':{'weather':weather},'output':{'uri':output+'linear','format':'parquet','publication':'single_writer'}}
    save(root/'inputs/config-linear.json',linear)
    engineer={'sources':{'weather':{'uri':weather['uri'],'cadence':'30min','format':'parquet','variables':{'bar':{'role':'response'}}}},
              'routes':{'weather':{'method':'linear','mode':'offline','max_gap':'2h'}},
              'output':{'uri':output+'engineer','format':'parquet','publication':'single_writer'}}
    save(root/'inputs/config-engineer.json',engineer)
    (root/'inputs/config.yaml').write_text('datasets:\n  weather:\n    uri: '+weather['uri']+'\n    cadence: 30min\n'
        '    key: row_key\n    targets: [bar]\n    method: LIN\n    max_gap: 2h\noutput:\n    uri: '+output+'yaml\n'
        '    format: parquet\n    publication: single_writer\n',encoding='utf-8',newline='\n')
    return {'partitions':3,'fills':24,'observations':852,'linear_fills':8,'linear_observations':284}


def readback(root, replay=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=Path(root)
    terminal=json.loads((root/'terminal.json').read_text())
    manifest_bytes=(root/'manifest.json').read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest()==terminal['manifest_sha256']
    manifest=json.loads(manifest_bytes)
    for item in manifest['files']:
        path=root/item['path'];assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size==item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    summary=json.loads((root/'summary.json').read_text())
    assert terminal['status']==summary['status']=='complete'
    entries=manifest['datasets'];assert len(entries) in (1,3)
    observations=fills=0
    for entry in entries:
        tables={}
        for name,path in entry['files'].items():
            file=pq.ParquetFile(root/path)
            assert file.metadata.num_rows==300
            table=pa.Table.from_batches(list(file.iter_batches(batch_size=512)))
            assert table.schema.field('timestamp').type==pa.timestamp('us','UTC')
            assert table['timestamp'].to_pylist()==[START+timedelta(minutes=30*i) for i in range(300)]
            expected=pa.float64() if name in ('raw','completed') else pa.string() if name in ('cell_states','dispositions') else pa.bool_()
            assert table.schema.field('bar').type==expected
            tables[name]=table['bar'].to_pylist()
            if replay is not None:
                assert (root/path).read_bytes()==(Path(replay)/path).read_bytes()
        grouped=entry['source']=='stations'
        for i in range(300):
            value=1000. if grouped else 1000.+.5*i
            if i in MISSING or i==150:
                assert tables['raw'][i] is None and tables['observed'][i] is False
                assert tables['reconstructed'][i] is (i in FILLED)
                if i in FILLED:
                    assert math.isclose(tables['completed'][i],value,rel_tol=0,abs_tol=1e-12);fills+=1
                else:
                    assert tables['completed'][i] is None and tables['dispositions'][i] in {'no_route','boundary_gap_unresolved'}
            else:
                assert tables['raw'][i]==tables['completed'][i]==value
                assert tables['observed'][i] is True and tables['reconstructed'][i] is False
                observations+=1
        assert tables['inserted'][150] is True
        reasons=json.loads((root/entry['source_keys']).read_text())['bar']
        keys={stamp:key for stamp,key,position in reasons['keys']}
        assert (START+timedelta(minutes=30*150)).isoformat() not in keys
        first=keys[START.isoformat()]
        if grouped:
            assert first==2**53+1+(1000 if entry['series']=='002' else 0) and type(first) is int
        elif 'generated' not in reasons['row_keys']:
            assert first=='000000'
    assert fills==8*len(entries) and observations==284*len(entries)
    assert summary['filled_cells']==fills
    return {'status':'complete','partitions':len(entries),'fills':fills,'observations':observations,
            'typed_tables':True,'replay_bytes_equal':True if replay is not None else None}


def replay_config(run, output, output_uri):
    run=Path(run);doc=json.loads((run/'config.resolved.json').read_text())
    for name,source in doc['sources'].items():
        assert source['format']=='parquet'
        source['uri']=str(run/'inputs'/(name+'.snapshot'))
    doc['output']['uri']=output_uri
    save(output,doc)
    return {'format':'parquet','snapshot_replay':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','readback','replay-config'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--container',action='store_true')
    parser.add_argument('--run');parser.add_argument('--output-uri');parser.add_argument('--replay')
    args=parser.parse_args()
    result=(prepare(args.output,args.container) if args.command=='prepare' else readback(args.output,args.replay)
            if args.command=='readback' else replay_config(args.run,args.output,args.output_uri))
    if args.command=='readback':
        assert not any(n=='ecofill' or n.startswith('ecofill.') for n in sys.modules), 'readback imported ecofill'
    print(json.dumps(result,sort_keys=True))
