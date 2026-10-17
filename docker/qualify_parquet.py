"""Small installed-candidate fixture/provenance/readback entry points; no native models."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import subprocess
import sys
import sysconfig


def writer_checks(root):
    """Publication entry points reject unsafe names without mutating destinations."""
    import gzip
    import os
    import pandas as pd
    from ecofill.operation.bundle import Writer
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    sentinel=root/'sentinel';sentinel.write_bytes(b'owned-adjacent-sentinel')
    unsafe=['','.', '..','./file','part/.','part/../sentinel','part//file','part/',
            str(sentinel), 'C:relative','C:/owned-test/sentinel','\\\\owned\\share\\sentinel',
            'part\\file','//owned/share','file\x00name',None,42]
    frame=pd.DataFrame({'bar':[1.,2.]},index=pd.date_range('2025-01-01',periods=2,tz='UTC'))
    negatives=0;link_results=[]
    for publication in ('strict','single_writer'):
        target=root/publication;writer=Writer(target,200000,publication=publication)
        def invoke(method,name):
            if method=='write':return writer.write(name,b'payload')
            if method=='csv':return writer.csv(name,frame)
            source=root/'encoder'
            with source.open('w+b') as encoded:
                encoded.write(b'closed');return writer.publish_closed(name,encoded)
        for method in ('write','csv','closed'):
            for name in unsafe:
                try:invoke(method,name)
                except ValueError as error:
                    assert str(error)=='invalid artifact path'
                else:raise AssertionError('unsafe publication name accepted')
                assert not target.exists() and writer.files==[] and writer.bytes==0
                assert sentinel.read_bytes()==b'owned-adjacent-sentinel'
                negatives+=1
        for method,suffix in [('write','bytes'),('csv','table.csv.gz'),('closed','closed')]:
            name='nested/'+suffix;receipt=invoke(method,name)
            data=(target/name).read_bytes();assert receipt['sha256']==hashlib.sha256(data).hexdigest()
            if method=='csv':assert gzip.decompress(data).decode()==frame.to_csv(index=True,index_label='timestamp',float_format='%.17g')
            else:assert data==(b'payload' if method=='write' else b'closed')
            before=(writer.bytes,writer.files.copy())
            try:invoke(method,name)
            except FileExistsError:pass
            else:raise AssertionError('existing artifact overwritten')
            assert (target/name).read_bytes()==data and (writer.bytes,writer.files)==before
        outside=root/(publication+'-adjacent');outside.mkdir();(outside/'keep').write_bytes(b'keep')
        link=target/'parent-link'
        try:link.symlink_to(outside,target_is_directory=True)
        except OSError as error:
            assert os.name=='nt',type(error).__name__
            link_results.append({'publication':publication,'status':'unavailable','error_type':type(error).__name__,
                                 'winerror':getattr(error,'winerror',None)})
        else:
            for method in ('write','csv','closed'):
                before=(writer.bytes,writer.files.copy(),sorted(p.name for p in target.iterdir()))
                try:invoke(method,'parent-link/escape')
                except ValueError as error:assert str(error)=='artifact path escapes bundle'
                else:raise AssertionError('escaping parent link accepted')
                assert (writer.bytes,writer.files,sorted(p.name for p in target.iterdir()))==before
                assert sorted(p.name for p in outside.iterdir())==['keep'] and (outside/'keep').read_bytes()==b'keep'
            link_results.append({'publication':publication,'status':'tested'})
    return {'unsafe_entry_point_cases':negatives,'parent_links':link_results,'adjacent_sentinel_unchanged':True}


def snapshot_checks(root):
    """Source keys, free-form metadata and shared physical captures through real API/CLI flows."""
    from contextlib import redirect_stdout
    from copy import deepcopy
    import csv
    from datetime import datetime, timedelta, timezone
    import io
    from unittest.mock import patch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ecofill.operation import load_config,plan,run,inspect,parquet_example
    from ecofill.operation.simple import resolve
    from ecofill.operation.preparation import read_source
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    stamps=[datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(minutes=30*i) for i in range(12)]
    def fixture(directory,bad=False,csv_input=False,offset=0):
        directory.mkdir();path=directory/('source.csv' if csv_input else 'source.parquet')
        values=['9007199254740993']*12 if bad else [float(i+offset) for i in range(12)]
        if csv_input:
            with path.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['timestamp','bar','other'])
                for i,stamp in enumerate(stamps):writer.writerow([stamp.isoformat(),values[i],100.+offset+i])
        else:pq.write_table(pa.table({'timestamp':stamps,'bar':values,'other':[100.+offset+i for i in range(12)]}),path)
        return path
    def simple(path,out,publication,targets=None):
        return {'datasets':{'source_key':{'uri':str(path),'cadence':'30min','targets':targets or ['bar'],
                'method':'LIN','max_gap':'1h'}},'output':{'uri':str(out),'format':'parquet','publication':publication}}
    def resolved(doc,prefix):
        value=resolve(load_config(doc),read_source)[0].to_dict()
        for source in value['sources'].values():
            for metadata in source['instruments'].values():
                for field in metadata['fields'].values():field['lineage']=prefix+'|'+field['column']
        return value
    checked=failed=shared=0
    for publication in ('strict','single_writer'):
        for index,prefix in enumerate(('metadata label','a//b','a/./b','../../escaped')):
            case=root/(publication+'-'+str(index));path=fixture(case)
            output=case/'output';(output/'runs').mkdir(parents=True)
            sentinel=output/'runs/escaped.snapshot';sentinel.write_bytes(b'adjacent-sentinel')
            doc=resolved(simple(path,output,publication),prefix)
            for channel in ('api','cli'):
                if channel=='api':result=run(doc,run_id=channel)
                else:
                    config=case/'config.json';config.write_text(json.dumps(doc))
                    with redirect_stdout(io.StringIO()) as out:
                        assert operation_cli(['run','--config',str(config),'--run-id',channel])==0,out.getvalue()
                    result=json.loads(out.getvalue())
                assert result['status']=='complete' and result['method_calls']==0
                bundle=output/'runs'/channel;assert inspect(bundle)==result
                assert [p.name for p in (bundle/'inputs').iterdir()]==['source_key.snapshot']
                assert (bundle/'inputs/source_key.snapshot').read_bytes()==path.read_bytes()
                manifest=json.loads((bundle/'input_manifest.json').read_text())['sources']
                assert manifest['source_key']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                saved=json.loads((bundle/'config.resolved.json').read_text())
                assert saved['sources']['source_key']['instruments']['source_key']['fields']['bar']['lineage']==prefix+'|bar'
                replay=deepcopy(saved);replay['sources']['source_key']['uri']=str(bundle/'inputs/source_key.snapshot')
                replay['output']['uri']=str(case/('replay-'+channel))
                copy=run(replay,run_id='copy');assert copy['status']=='complete'
                replay_bundle=case/('replay-'+channel)/'runs/copy';assert inspect(replay_bundle)==copy
                assert (bundle/'datasets/source_key-0/completed.parquet').read_bytes()==(replay_bundle/'datasets/source_key-0/completed.parquet').read_bytes()
                assert sentinel.read_bytes()==b'adjacent-sentinel'
                assert sorted(p.name for p in (output/'runs').iterdir())==sorted(['escaped.snapshot',*('api','cli')[:1 if channel=='api' else 2]])
                checked+=1
            bad_path=fixture(case/'bad',bad=True)
            bad_doc=resolved(simple(bad_path,case/'failed',publication),'../../escaped')
            original=bad_path.read_bytes()
            with patch('ecofill.operation.runner.execute_native',side_effect=AssertionError('model called')) as model:
                for channel in ('api','cli'):
                    if channel=='api':result=run(bad_doc,run_id=channel)
                    else:
                        config=case/'bad-config.json';config.write_text(json.dumps(bad_doc))
                        with redirect_stdout(io.StringIO()) as out:
                            assert operation_cli(['run','--config',str(config),'--run-id',channel])==1
                        result=json.loads(out.getvalue())
                    assert model.call_count==0
                    bad_bundle=case/'failed/runs'/channel
                    assert result['status']=='failed' and result['method_calls']==0 and result['failure_diagnostic']['code']=='lossy_integer_measurement'
                    assert inspect(bad_bundle)==result and (bad_bundle/'inputs/source_key.snapshot').read_bytes()==original
                    failed+=1
            assert sorted(p.name for p in (case/'failed/runs').iterdir())==['api','cli']
        # Distinct map keys whose free-form labels share a prefix; physical sharing is separately permitted.
        for same_file in (False,True):
            case=root/(publication+'-multiple-'+str(same_file));first=fixture(case);second=first if same_file else fixture(case/'second',offset=1000)
            if not same_file:assert first.read_bytes()!=second.read_bytes()
            doc=simple(first,case/'output',publication)
            doc['datasets']['second_key']={'uri':str(second),'cadence':'30min','targets':['other'],'method':'LIN','max_gap':'1h'}
            expanded=resolve(load_config(doc),read_source)[0].to_dict()
            for key,source in expanded['sources'].items():
                for metadata in source['instruments'].values():
                    for field in metadata['fields'].values():field['lineage']='shared-prefix|'+('physical' if same_file else key)+'|'+field['column']
            with patch('ecofill.operation.runner.read_source',wraps=read_source) as reader:
                result=run(expanded,run_id='two');assert reader.call_count==(1 if same_file else 2)
            assert result['status']=='complete';bundle=case/'output/runs/two';assert inspect(bundle)==result
            assert sorted(p.name for p in (bundle/'inputs').iterdir())==['second_key.snapshot','source_key.snapshot']
            for key,path in [('source_key',first),('second_key',second)]:
                assert (bundle/'inputs'/(key+'.snapshot')).read_bytes()==path.read_bytes()
                assert json.loads((bundle/'input_manifest.json').read_text())['sources'][key]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            shared+=1
        work=root/(publication+'-pending');parquet_example.prepare(work)
        doc=json.loads((work/'inputs/config.json').read_text())
        for source in doc['datasets'].values():source['uri']=str(work/source['uri'])
        doc['output'].update(uri=str(work/'output/single'),publication=publication)
        with patch('ecofill.operation.runner.read_source',wraps=read_source) as reader:
            result=run(doc,run_id='grouped');assert reader.call_count==3
        assert result['filled_cells']==24 and parquet_example.readback(work/'output/single/runs/grouped')['observations']==852
        assert sorted(p.name for p in (work/'output/single/runs/grouped/inputs').iterdir())==['donors.snapshot','stations.snapshot','weather.snapshot']
        # The engineer alias and resolved CSV retain the same keyed association.
        engineer=json.loads((work/'inputs/config-engineer.json').read_text())
        engineer['sources']['weather']['uri']=str(work/'inputs/weather.parquet')
        engineer['output'].update(uri=str(work/'output/engineer'),publication=publication)
        config=work/'engineer.json';config.write_text(json.dumps(engineer))
        with redirect_stdout(io.StringIO()) as out:assert operation_cli(['run','--config',str(config),'--run-id','alias'])==0
        assert json.loads(out.getvalue())['filled_cells']==8
        assert (work/'output/engineer/runs/alias/inputs/weather.snapshot').read_bytes()==(work/'inputs/weather.parquet').read_bytes()
        csv_path=fixture(root/(publication+'-csv'),csv_input=True)
        csv_doc=resolved(simple(csv_path,root/(publication+'-csv-output'),publication),'../../escaped')
        csv_doc['output'].pop('format');result=run(csv_doc,run_id='csv');assert result['status']=='complete'
        bundle=root/(publication+'-csv-output/runs/csv');assert inspect(bundle)==result
        assert (bundle/'inputs/source_key.snapshot').read_bytes()==csv_path.read_bytes()
    return {'lineage_api_cli_successes':checked,'conversion_failures':failed,'multiple_source_controls':shared,
            'pending_grouped_and_engineer':True,'csv_compatible':True,'outside_artifacts':0}


def operation_cli(args):
    from ecofill.operation.cli import main
    return main(args)


def publication_checks(root):
    """Run-level publication faults must never become missing-source or retry behavior."""
    from collections import Counter
    from contextlib import redirect_stdout
    from copy import deepcopy
    from datetime import datetime, timedelta, timezone
    import io
    from unittest.mock import patch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ecofill.operation import load_config, run, inspect
    from ecofill.operation.bundle import Writer
    from ecofill.operation.preparation import read_source
    from ecofill.operation.runner import prepare_operation, execute_native
    from ecofill.operation.simple import resolve

    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    stamps=[datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(minutes=30*i) for i in range(12)]
    paths={};originals={}
    for name,offset in [('a',0),('b',100)]:
        path=root/(name+'.parquet');paths[name]=path
        pq.write_table(pa.table({'timestamp':stamps,'bar':[None if i==5 else float(offset+i) for i in range(12)]}),path)
        originals[name]=path.read_bytes()

    def config(form,publication,output):
        doc={'datasets':{name:{'uri':str(path),'cadence':'30min','targets':['bar'],
                             'method':'LIN','max_gap':'1h'} for name,path in paths.items()},
             'output':{'uri':str(output),'format':'parquet','publication':publication}}
        return doc if form=='pending' else resolve(load_config(doc),read_source)[0].to_dict()

    counts=Counter();diagnostic={'code':'synthetic_snapshot_failure','location':'/owned-fixture',
                                'explanation':'injected snapshot publication failure'}
    original_write=Writer.write
    for form in ('pending','resolved'):
        for publication in ('strict','single_writer'):
            direct=config(form,publication,root/(form+'-'+publication+'-direct'))
            for key in ('a','b'):
                failure=FileNotFoundError('synthetic observer fault');seen=[];reads=[]
                def reader(source,limits):
                    reads.append(source['uri']);return read_source(source,limits)
                def observer(name,entry):
                    seen.append(name);assert entry.payload==originals[name]
                    if name==key:raise failure
                try:prepare_operation(direct,source_reader=reader,input_observer=observer)
                except FileNotFoundError as caught:assert caught is failure
                else:raise AssertionError('observer failure swallowed')
                assert seen==(['a'] if key=='a' else ['a','b']) and len(reads)==len(seen)
                counts['direct_observer_identity']+=1
            for error_type in (PermissionError,ValueError):
                failure=error_type('synthetic reader fault');seen=[]
                def reader(source,limits):
                    if source['uri']==str(paths['b']):raise failure
                    return read_source(source,limits)
                try:prepare_operation(direct,source_reader=reader,input_observer=lambda name,entry:seen.append(name))
                except error_type as caught:assert caught is failure
                else:raise AssertionError('reader failure swallowed')
                assert seen==['a'];counts['other_reader_failures']+=1

            cases=[('before','a',False,False),('after','a',True,False),
                   ('later-before','b',False,False),('later-after','b',True,False),
                   ('secondary-terminal','a',True,True),('missing',None,False,False),
                   ('control',None,False,False)]
            for case,key,after,terminal_fault in cases:
                for channel in ('api','cli'):
                    output=root/('-'.join((form,publication,case,channel)))
                    doc=config(form,publication,output)
                    if case=='missing':
                        section=doc['datasets'] if form=='pending' else doc['sources']
                        section['b']['uri']=str(root/'missing.parquet')
                    failure=FileNotFoundError('synthetic snapshot fault');failure.diagnostic=deepcopy(diagnostic)
                    snapshot_attempts=[];write_attempts=[];reads=[];injected=[]
                    def writer(self,name,payload,**kwargs):
                        write_attempts.append(name)
                        if name.startswith('inputs/') and name.endswith('.snapshot'):
                            snapshot_attempts.append(name)
                            if name=='inputs/'+str(key)+'.snapshot' and not injected:
                                injected.append(name)
                                if after:original_write(self,name,payload,**kwargs)
                                raise failure
                        if terminal_fault and name=='terminal.json':raise PermissionError('synthetic terminal fault')
                        return original_write(self,name,payload,**kwargs)
                    def reader(source,limits):
                        reads.append(source['uri']);return read_source(source,limits)
                    with patch.object(Writer,'write',writer),patch('ecofill.operation.runner.read_source',side_effect=reader),\
                         patch('ecofill.operation.runner.execute_native',wraps=execute_native) as model:
                        if channel=='api':result=run(doc,run_id='first')
                        else:
                            config_file=output.with_suffix('.json');config_file.write_text(json.dumps(doc))
                            with redirect_stdout(io.StringIO()) as printed:
                                code=operation_cli(['run','--config',str(config_file),'--run-id','first'])
                            result=json.loads(printed.getvalue())
                            assert code==(1 if key is not None else 2 if case=='missing' else 0)
                        if key is not None:
                            assert result['status']=='failed' and result['failure_type']=='FileNotFoundError'
                            assert result['failure_diagnostic']==diagnostic
                            assert result['method_calls']==0 and model.call_count==0
                            assert result['datasets_attempted']==0 and result['datasets_processed']==0
                            assert result['published_partitions']==[] and result['unresolved_reasons']=={}
                        else:
                            expected=1 if case=='missing' else 2
                            assert result['status']==('partial' if case=='missing' else 'complete')
                            assert result['failure_type'] is None and result['method_calls']==model.call_count==expected
                            assert result['datasets_processed']==expected and result['filled_cells']==expected
                    bundle=output/'runs/first'
                    expected_attempts=['inputs/a.snapshot']
                    if key=='b' or case=='control':expected_attempts.append('inputs/b.snapshot')
                    assert snapshot_attempts==expected_attempts
                    assert len(reads)==len(expected_attempts)+(case=='missing')
                    expected_snapshots=set()
                    if key is None:expected_snapshots={'a'} if case=='missing' else {'a','b'}
                    elif key=='b':expected_snapshots={'a','b'} if after else {'a'}
                    elif after:expected_snapshots={'a'}
                    actual=set(p.stem for p in (bundle/'inputs').iterdir()) if (bundle/'inputs').exists() else set()
                    assert actual==expected_snapshots
                    manifest=json.loads((bundle/'manifest.json').read_text())
                    records={item['path']:item for item in manifest['files']}
                    assert len(records)==len(manifest['files'])
                    assert {name for name in records if name.startswith('inputs/')}=={'inputs/'+n+'.snapshot' for n in expected_snapshots}
                    for name in expected_snapshots:
                        payload=(bundle/'inputs'/(name+'.snapshot')).read_bytes()
                        assert payload==originals[name]
                        assert records['inputs/'+name+'.snapshot']['sha256']==hashlib.sha256(payload).hexdigest()
                        assert records['inputs/'+name+'.snapshot']['bytes']==len(payload)
                    assert write_attempts[-1]=='terminal.json'
                    if terminal_fault:
                        assert result['terminal_published'] is False
                        assert result['secondary_failures']==[{'stage':'terminal','error_type':'PermissionError'}]
                        assert inspect(bundle)=={'status':'incomplete','terminal_present':False}
                        assert not (bundle/'terminal.json').exists()
                        assert json.loads((bundle/'provenance.json').read_text())['failure_type']=='FileNotFoundError'
                        counts['secondary_terminal_runs']+=1
                    else:
                        assert result['secondary_failures']==[] and inspect(bundle)==result
                        assert json.loads((bundle/'terminal.json').read_text())['status']==result['status']
                    if key is not None:
                        assert injected==['inputs/'+key+'.snapshot']
                        assert not (bundle/'plan.json').exists() and not (bundle/'input_manifest.json').exists()
                        counts['fault_runs']+=1
                    else:
                        input_manifest=json.loads((bundle/'input_manifest.json').read_text())['sources']
                        assert set(input_manifest)==expected_snapshots
                        for name in expected_snapshots:
                            assert input_manifest[name]['sha256']==hashlib.sha256(originals[name]).hexdigest()
                            completed=pq.ParquetFile(bundle/'datasets'/(name+'-0')/'completed.parquet').read().to_pydict()
                            assert completed['bar']==[float((0 if name=='a' else 100)+i) for i in range(12)]
                        counts['missing_reader_runs' if case=='missing' else 'no_fault_runs']+=1
                    assert all(path.read_bytes()==originals[name] for name,path in paths.items())
    return dict(counts)|{'snapshot_retry_attempts':0,'original_inputs_unchanged':True,
                        'forms':['pending','resolved'],'publications':['strict','single_writer'],'channels':['api','cli']}


def numeric_checks(root):
    """Real-file conversion regressions shared by source, installed-wheel and image checks."""
    from contextlib import redirect_stdout
    import csv
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal
    import io
    from unittest.mock import patch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ecofill.operation import plan, run, inspect
    from ecofill.operation.cli import main as cli
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    path=root/'input.parquet';stamps=[datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(minutes=30*i) for i in range(12)]
    def config(uri=path):
        return {'datasets':{'weather':{'uri':str(uri),'cadence':'30min','targets':['bar'],'key':'key',
                'method':'LIN','max_gap':'1h'}},'output':{'uri':str(root/'output'),'format':'parquet'}}
    def write(values,dictionary=False,keys=None):
        array=pa.array(values)
        if dictionary:array=array.dictionary_encode()
        pq.write_table(pa.table({'timestamp':stamps,'bar':array,'key':keys or ['00001','9007199254740993',*map(str,range(10))]}),path)
        return path.read_bytes()
    rejected=[('9007199254740993','lossy_integer_measurement'),('-9007199254740993','lossy_integer_measurement'),
              ('9007199254740993.0','lossy_integer_measurement'),('9.007199254740993e15','lossy_integer_measurement'),
              ('90071992547409930e-1','lossy_integer_measurement'),('-9.007199254740993e15','lossy_integer_measurement'),
              ('1e309','numeric_overflow'),('-1e309','numeric_overflow'),('1e1000000000','numeric_overflow'),
              ('1e999999999999999999999999999999','numeric_overflow')]
    failures=0
    for dictionary in (False,True):
        for n,(value,code) in enumerate(rejected):
            original=write([value]*12,dictionary)
            doc=config();identifier=f'bad-{int(dictionary)}-{n}'
            with patch('ecofill.operation.runner.execute_native',side_effect=AssertionError('model called')) as model:
                try:plan(doc)
                except ValueError as error:
                    assert error.diagnostic['code']==code
                    assert error.diagnostic['location']=='/datasets/weather/columns/bar'
                    assert value not in error.diagnostic['explanation'] and 'float64' in error.diagnostic['explanation']
                else:raise AssertionError('unsafe numeric string planned')
                result=run(doc,run_id=identifier)
                assert model.call_count==0
            assert result['status']=='failed' and result['method_calls']==0 and result['filled_cells']==0
            assert result['failure_type']=='ValueError' and result['failure_diagnostic']['code']==code
            bundle=root/'output/runs'/identifier
            assert inspect(bundle)==result and (bundle/'inputs/weather.snapshot').read_bytes()==original
            assert path.read_bytes()==original
            failures+=1
    controls=[('9007199254740992',9007199254740992.),('9007199254740994',9007199254740994.),
              ('-9007199254740992',-9007199254740992.),('9007199254740992.0',9007199254740992.),
              ('9.007199254740994e15',9007199254740994.),('1.25',1.25),('1.25e1',12.5),('-3.5e-1',-.35),
              (9007199254740994,9007199254740994.),(1.25,1.25),(Decimal('1.25'),1.25)]
    for n,(value,expected) in enumerate(controls):
        original=write([value]*12,isinstance(value,str),[2**53+1+i for i in range(12)])
        result=run(config(),run_id=f'control-{n}');assert result['status']=='complete' and result['method_calls']==0
        bundle=root/f'output/runs/control-{n}'
        assert pq.read_table(bundle/'datasets/weather-0/raw.parquet')['bar'].to_pylist()==[expected]*12
        assert (bundle/'inputs/weather.snapshot').read_bytes()==original
        audit=json.loads((bundle/'datasets/weather-0/source_reasons.json').read_text())['bar']
        assert audit['keys'][0][1]==2**53+1
    values=[None,'NA','nan','-inf','malformed','-9999','5','9007199254740993','1.25','2','3','4']
    original=write(values,True)
    table=pq.read_table(path).append_column('qc',pa.array([True]*6+[False]+[True]*5));pq.write_table(table,path);original=path.read_bytes()
    doc=config();doc['datasets']['weather'].update(missing={'tokens':['NA','9007199254740993'],'sentinels':[-9999]},
        columns={'bar':{'qc':{'column':'qc','accept':['True']}}})
    result=run(doc,run_id='precedence');assert result['status']=='complete'
    bundle=root/'output/runs/precedence'
    raw_values=pq.read_table(bundle/'datasets/weather-0/raw.parquet')['bar'].to_pylist()
    assert raw_values==[None,None,None,None,None,-9999.,5.,None,1.25,2.,3.,4.]
    observed=pq.read_table(bundle/'datasets/weather-0/observed.parquet')['bar'].to_pylist()
    assert observed==[False]*8+[True]*4
    assert pq.read_table(bundle/'datasets/weather-0/qc_rejected.parquet')['bar'].to_pylist()==[False]*5+[True,True]+[False]*5
    audit=json.loads((bundle/'datasets/weather-0/source_reasons.json').read_text())['bar']
    assert audit['keys'][0][1]=='00001' and audit['keys'][1][1]=='9007199254740993'
    assert [row[1] for row in audit['source_reasons']][:5]==[['original_missing'],['original_missing'],
        ['nonfinite_or_invalid'],['nonfinite_or_invalid'],['invalid_numeric']]
    assert (bundle/'inputs/weather.snapshot').read_bytes()==original
    # The equivalent CSV parsing retains its legacy rounding and overflow-as-missing policy.
    csv_path=root/'legacy.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['timestamp','bar','key'])
        for i,stamp in enumerate(stamps):writer.writerow([stamp.isoformat(),['9007199254740993','1e309','-1e309',*['1.25']*9][i],str(i)])
    result=run(config(csv_path),run_id='csv-control');assert result['status']=='complete'
    assert pq.read_table(root/'output/runs/csv-control/datasets/weather-0/raw.parquet')['bar'].to_pylist()==[9007199254740992.,None,None]+[1.25]*9
    bad=write(['9007199254740993']*12);doc=config();cfg=root/'bad.json';cfg.write_text(json.dumps(doc))
    with redirect_stdout(io.StringIO()) as output:
        assert cli(['run','--config',str(cfg),'--run-id','cli-bad'])==1
    assert json.loads(output.getvalue())['failure_diagnostic']['code']=='lossy_integer_measurement'
    assert path.read_bytes()==bad
    return {'rejected_numeric_cases':failures,'control_cases':len(controls),'precedence_and_exact_keys':True,'legacy_csv_unchanged':True}


DEPENDENCY_CHILD = r'''
import importlib.abc,importlib.metadata,json,socket,sys
from pathlib import Path
sys.path[:0]=sys.argv[1:3]
root=Path(sys.argv[3]);phase=sys.argv[4];blocked=[]
versions=json.loads(sys.argv[5]);real_version=importlib.metadata.version
def version(name):
    if name not in versions:return real_version(name)
    if versions[name] is None:raise importlib.metadata.PackageNotFoundError(name)
    return versions[name]
importlib.metadata.version=version
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'pyarrow','torch','tabpfn','timesfm','sklearn','scipy','google','huggingface_hub','safetensors'}:
            blocked.append(name)
            raise ImportError('simulated unavailable optional import')
sys.meta_path.insert(0,Deny())
def denied(*a,**k):raise AssertionError('network denied')
socket.socket.connect=socket.create_connection=socket.getaddrinfo=denied
from contextlib import redirect_stdout
from copy import deepcopy
import io
from ecofill.operation import load_config,run,inspect
from ecofill.operation.cli import main
from ecofill.operation import runner
actual=runner.execute_native;calls=[]
def counted(*a,**k):calls.append(True);return actual(*a,**k)
runner.execute_native=counted
docs=json.loads((root/'cases.json').read_text())
def cli(*args):
    with redirect_stdout(io.StringIO()) as out:code=main(list(args))
    return code,json.loads(out.getvalue())
for name,doc in docs.items():
    if doc['output'].get('format')!='parquet':continue
    before=len(blocked);load_config(doc)
    assert cli('validate','--config',str(root/(name+'.json')))[0]==0
    assert len(blocked)==before
    out=Path(doc['output']['uri'])
    for channel in ('api','cli'):
        before=len(calls)
        if phase=='existing':
            assert run(doc,run_id=channel)['status']=='complete'
            assert cli('inspect','--run-uri',str(out/'runs'/channel))[0]==0
            changed=deepcopy(doc);changed['datasets']['weather']['max_gap']='2h'
            try:run(changed,run_id=channel)
            except ValueError as error:assert str(error)=='run identity conflict'
            else:raise AssertionError('identity conflict bypassed')
            assert len(calls)==before
            continue
        if channel=='api':
            try:run(doc,run_id=channel)
            except ValueError as error:
                assert error.diagnostic['code']=='parquet_dependency_missing'
                assert error.diagnostic['location']=='/output/format'
                assert 'ecofill[parquet]' in error.diagnostic['explanation']
            else:raise AssertionError('missing extra accepted')
        else:
            code,result=cli('run','--config',str(root/(name+'.json')),'--run-id',channel)
            assert code==1 and result['code']=='parquet_dependency_missing' and result['location']=='/output/format'
        assert len(calls)==before and not out.exists()
if phase=='blocked':
    for name,doc in docs.items():
        if doc['output'].get('format')=='parquet':continue
        for channel in ('api','cli'):
            before=len(calls);out=Path(doc['output']['uri'])/'runs'/channel
            if channel=='api':result=run(doc,run_id=channel)
            else:
                code,result=cli('run','--config',str(root/(name+'.json')),'--run-id',channel)
                assert code==(1 if name.startswith('input-') else 0)
            assert inspect(out)==result and cli('inspect','--run-uri',str(out))[0]==(1 if name.startswith('input-') else 0)
            if name.startswith('input-'):
                assert result['status']=='failed' and result['failure_type']=='ValueError' and result['method_calls']==0
                assert result['failure_diagnostic']['code']=='parquet_dependency_missing' and len(calls)==before
                assert result['failure_diagnostic']['location']=='/datasets/weather/format' and result['secondary_failures']==[]
            else:assert result['status']=='complete' and result['filled_cells']==1
    assert len(calls)==4
assert not any(name.split('.')[0]=='pyarrow' for name in sys.modules)
print(json.dumps({'phase':phase,'arrow_absent_simulation':True,'failure_model_calls':0,'orphan_claims':0,'blocked_imports':len(blocked)}))
'''


def dependency_checks(root):
    """Missing-import simulation in fresh interpreters; never uninstall dependencies."""
    import csv
    from contextlib import redirect_stdout
    from datetime import datetime, timedelta, timezone
    import pyarrow as pa
    import pyarrow.parquet as pq
    import ecofill
    import pandas
    import io
    from ecofill.operation import run, inspect
    from ecofill.operation.cli import main as operation_cli
    from ecofill.operation.runner import provenance as operation_provenance
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    stamps=[datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(minutes=30*i) for i in range(12)]
    values=[float(i) if i!=5 else None for i in range(12)]
    pq.write_table(pa.table({'timestamp':stamps,'bar':values}),root/'input.parquet')
    with (root/'input.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['timestamp','bar'])
        for stamp,value in zip(stamps,values):writer.writerow([stamp.isoformat(),'' if value is None else value])
    docs={}
    for publication in ('strict','single_writer'):
        for input_format,output_format in [('csv','parquet'),('parquet','parquet'),('parquet','csv'),('csv','csv')]:
            name=('output-' if output_format=='parquet' else 'input-' if input_format=='parquet' else 'csv-')+publication+'-'+input_format
            doc={'datasets':{'weather':{'uri':str(root/('input.'+input_format)),'cadence':'30min','targets':['bar'],
                         'method':'LIN','max_gap':'1h'}},'output':{'uri':str(root/name),'publication':publication}}
            if output_format=='parquet':doc['output']['format']='parquet'
            docs[name]=doc;(root/(name+'.json')).write_text(json.dumps(doc))
    (root/'cases.json').write_text(json.dumps(docs))
    def child(phase):
        command=[sys.executable,'-I','-S','-B','-c',DEPENDENCY_CHILD,
                 str(Path(ecofill.__file__).resolve().parent.parent),str(Path(pandas.__file__).resolve().parent.parent),str(root),phase]
        # The public verifier deliberately hides optional metadata. Keep the same
        # caller-visible version view across subprocesses so identity is comparable.
        command.append(json.dumps(operation_provenance()['libraries']))
        result=subprocess.run(command,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        return json.loads(result.stdout)
    blocked=child('blocked')
    for name,doc in docs.items():
        if name.startswith('csv-'):continue
        for channel in ('api','cli'):
            if name.startswith('input-'):
                previous=inspect(Path(doc['output']['uri'])/'runs'/channel)
                assert run(doc,run_id=channel)==previous
                identifier=channel+'-fresh'
            else:identifier=channel
            if channel=='cli':
                with redirect_stdout(io.StringIO()) as output:
                    code=operation_cli(['run','--config',str(root/(name+'.json')),'--run-id',identifier])
                assert code==0,output.getvalue()
                result=json.loads(output.getvalue())
            else:result=run(doc,run_id=identifier)
            assert result['status']=='complete' and result['filled_cells']==1
            assert run(doc,run_id=identifier)==result
        # Existing empty claims remain incomplete; the correction never repairs them.
        old=Path(doc['output']['uri'])/'runs/old-incomplete';old.mkdir()
        assert run(doc,run_id='old-incomplete')['status']=='in_progress' and list(old.iterdir())==[]
    existing=child('existing')
    return {'simulated_missing_arrow':blocked,'completed_idempotency_without_arrow':existing,
            'restored_import_reruns':True,'old_incomplete_untouched':True}


def example():
    path=Path(sysconfig.get_path('purelib'))/'ecofill/operation/parquet_example.py'
    assert path.is_file()
    return path


def provenance():
    import ecofill
    import pyarrow
    root=Path(importlib.metadata.distribution('ecofill').locate_file('ecofill')).resolve()
    assert Path(ecofill.__file__).resolve().parent==root
    assert str(root).startswith('/usr/local/lib/python3.12/site-packages/')
    assert pyarrow.__version__=='25.0.1'
    lock=Path('/opt/ecofill/requirements.lock').read_bytes()
    inventory=json.loads(Path('/opt/ecofill/dependency-inventory.json').read_text())
    for package in inventory['packages']:
        if package['name']!='wheel':assert importlib.metadata.version(package['name'])==package['version']
    import os
    wheel=Path(os.environ['ECOFILL_WHEEL_PATH'])
    return {'package':str(root),'pyarrow':pyarrow.__version__,'uid':os.getuid(),
            'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),
            'lock_sha256':hashlib.sha256(lock).hexdigest(),'foundation_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','readback','replay-config','provenance','numeric-checks','dependency-checks','snapshot-checks','writer-checks','publication-checks'])
    parser.add_argument('--output');parser.add_argument('--run');parser.add_argument('--output-uri');parser.add_argument('--replay')
    args=parser.parse_args()
    if args.command=='provenance':result=provenance()
    elif args.command=='numeric-checks':result=numeric_checks(args.output)
    elif args.command=='dependency-checks':result=dependency_checks(args.output)
    elif args.command=='snapshot-checks':result=snapshot_checks(args.output)
    elif args.command=='writer-checks':result=writer_checks(args.output)
    elif args.command=='publication-checks':result=publication_checks(args.output)
    else:
        installed=runpy.run_path(str(example()))
        if args.command=='prepare':result=installed['prepare'](args.output,container=True)
        elif args.command=='replay-config':result=installed['replay_config'](args.run,args.output,args.output_uri)
        else:
            # A fresh interpreter reads numbers and hashes with Arrow/stdlib only.
            command=[sys.executable,'-I','-B',str(example()),'readback','--output',args.output]
            if args.replay:command+=['--replay',args.replay]
            child=subprocess.run(command,check=True,capture_output=True,text=True)
            result=json.loads(child.stdout)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
