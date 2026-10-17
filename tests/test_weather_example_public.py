"""Clearly synthetic Davis-schema software checks; no archived measurements or foundation models."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest


def synthetic_checks(configs, root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ecofill.operation import load_config, plan, run, inspect
    from ecofill.operation.cli import main
    root=Path(root);root.mkdir(parents=True,exist_ok=False);configs=Path(configs)
    originals={n:(configs/n).read_bytes() for n in ('davis-weather.json','davis-weather-rf.json')}
    docs={n:json.loads(v) for n,v in originals.items()}
    for name in originals:
        with redirect_stdout(io.StringIO()) as captured:code=main(['validate','--config',str(configs/name)])
        assert code==0 and json.loads(captured.getvalue())['status']=='valid'
    main_doc=docs['davis-weather.json'];spec=main_doc['datasets']['davis_weather']
    assert spec['select']==['RBMNN-AB-01','RBRL-COL-01'] and spec['interval']==spec['validity']
    smooth=['temp_out','humidity','dew_point','heat_index','bar'];fast=['rain','rain_rate','solar_rad','solar_energy','et']
    assert set(spec['targets'])==set(smooth+fast) and spec['method']=='LIN' and spec['max_gap']=='48h'
    assert all(spec['targets'][n]=={'method':'LIN','max_gap':'2h'} for n in fast)
    assert all(d['limits']['max_rows_per_series']==400000 for d in docs.values())
    rows=[]
    for station in ('RBMNN-AB-01','RBRL-COL-01','SYNTHETIC-OTHER'):
        for i in range(96):
            record={'primary_key':f'SYNTHETIC-{station}-{i:08d}','station_id':station,
                'timestamp':datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(hours=i),
                'data_source':'SYNTHETIC','wind_speed':7.25,'wind_chill':12.5}
            record.update({n:(int(i+20) if n in ('humidity','solar_rad') else float(i+20)) for n in smooth+fast})
            record['solar_energy']=None
            if station=='RBMNN-AB-01':
                for n in smooth:
                    if i==0 or i==5 or 10<=i<13 or 20<=i<33 or 40<=i<89:record[n]=None
                if i==5 or 10<=i<13:record['rain']=None
            rows.append(record)
    for i in range(200):
        record={n:20. for n in smooth+fast};record.update(humidity=50,solar_rad=0,solar_energy=None,
            primary_key=f'SYNTHETIC-RF-{i:08d}',station_id='RBRL-COL-01',
            timestamp=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(hours=i),
            data_source='SYNTHETIC',wind_speed=7.25,wind_chill=12.5)
        if i==100:record['bar']=None
        rows.append(record)
    names=['primary_key','station_id','timestamp','temp_out','humidity','dew_point','wind_speed','wind_chill',
           'heat_index','bar','rain','rain_rate','solar_rad','solar_energy','et','data_source']
    schema=pa.schema([(n,pa.string() if n in ('primary_key','station_id','data_source') else
                      pa.timestamp('us',tz='UTC') if n=='timestamp' else pa.int64() if n in ('humidity','solar_rad') else pa.float64()) for n in names])
    table=pa.Table.from_pylist(rows,schema=schema);path=root/'whole-synthetic.parquet';pq.write_table(table,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    def relocate(doc,label):
        value=deepcopy(doc);value['datasets']['davis_weather']['uri']=str(path);value['output']['uri']=str(root/label);return value
    doc=relocate(main_doc,'main');planned=plan(doc).to_dict();assert not planned['budget_exceeded']
    # Across-station duplicate timestamps are legitimate; within-station duplicates are not.
    duplicate=root/'duplicate.parquet';pq.write_table(pa.concat_tables([table,table.slice(0,1)]),duplicate)
    bad=deepcopy(doc);bad['datasets']['davis_weather']['uri']=str(duplicate)
    try:plan(bad)
    except ValueError:pass
    else:raise AssertionError('within-station duplicate accepted')
    wrong=deepcopy(doc);wrong['datasets']['davis_weather']['cadence']='2h'
    try:plan(wrong)
    except ValueError:pass
    else:raise AssertionError('wrong cadence accepted')
    missing=root/'missing.parquet';pq.write_table(table.drop(['heat_index']),missing)
    bad=deepcopy(doc);bad['datasets']['davis_weather']['uri']=str(missing)
    try:plan(bad)
    except ValueError:pass
    else:raise AssertionError('missing required measurement accepted')
    edited=deepcopy(doc);edited['datasets']['davis_weather'].update(select=['SYNTHETIC-OTHER'],
        interval=['2025-01-02T00:00:00Z','2025-01-03T00:00:00Z'],validity=['2025-01-02T00:00:00Z','2025-01-03T00:00:00Z'])
    selected=plan(edited).to_dict();assert len(selected['partitions'])==1
    local=root/'main.json';local.write_text(json.dumps(doc))
    with redirect_stdout(io.StringIO()) as captured:code=main(['run','--config',str(local),'--run-id','synthetic'])
    summary=json.loads(captured.getvalue());assert code==0 and summary['status']=='complete'
    assert summary['method_calls']==16 and summary['filled_gaps']==16 and summary['filled_cells']==86
    bundle=root/'main/runs/synthetic';assert inspect(bundle)==summary
    assert hashlib.sha256((bundle/'inputs/davis_weather.snapshot').read_bytes()).hexdigest()==digest
    manifest=json.loads((bundle/'manifest.json').read_bytes());assert len(manifest['datasets'])==2
    classes={n:set() for n in smooth};reasons=set()
    for entry in manifest['datasets']:
        assert entry['series'] in spec['select']
        completed=pq.ParquetFile(bundle/entry['files']['completed']).read()
        observed=pq.ParquetFile(bundle/entry['files']['observed']).read().to_pydict()
        assert set(completed.column_names)=={'timestamp',*smooth,*fast}
        assert completed.schema.field('humidity').type==pa.float64()
        values=completed.to_pydict()
        expected={r['timestamp']:r for r in rows if r['station_id']==entry['series']}
        for i,stamp in enumerate(values['timestamp']):
            for name in smooth+fast:
                if observed[name][i]:assert values[name][i]==float(expected[stamp][name])
        audits=json.loads((bundle/Path(entry['files']['completed']).parent/'gaps.json').read_bytes())
        for gap in audits:
            assert gap['variable'] not in ('wind_speed','wind_chill')
            if gap['complete']:
                assert gap['duration_seconds']<=(48 if gap['variable'] in smooth else 2)*3600
                if gap['variable'] in smooth:classes[gap['variable']].add(gap['duration_class'])
            else:reasons.add(gap['final_reason'])
    assert all(v=={'short','medium','long'} for v in classes.values())
    assert {'gap_exceeds_limit','boundary_gap_unresolved','no_route'}<=reasons
    rf=relocate(docs['davis-weather-rf.json'],'rf');rf_plan=plan(rf).to_dict()
    assert rf_plan['method_call_upper_bound']==2 and len(rf_plan['gaps'])==1
    recipe=rf_plan['gaps'][0]['recipes'][0]
    assert recipe['method']=='RF-X' and [p['field'] for p in recipe['predictors']]==['temp_out','humidity']
    assert recipe['parameters']=={'max_gap_seconds':172800,'n_estimators':128,'max_depth':12,'min_samples_leaf':5,
                                 'seed':0,'n_jobs':1,'window_days':180,'min_rows':168,'max_rows':4096}
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    assert all((configs/n).read_bytes()==v for n,v in originals.items())
    assert pq.ParquetFile(path).read(columns=['wind_speed','wind_chill']).to_pydict()=={
        'wind_speed':[7.25]*len(rows),'wind_chill':[12.5]*len(rows)}
    return {'synthetic':True,'columns':16,'native_lin_calls':16,'native_rf_fits':0,'rf_plan_gaps':1,
            'three_classes_for_smooth_targets':True,'config_bytes_unchanged':True,'input_unchanged':True}


class WeatherExampleTests(unittest.TestCase):
    def test_exact_configs_and_synthetic_table_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            result=synthetic_checks(Path(__file__).resolve().parents[1]/'examples/operation',Path(temporary)/'synthetic')
        self.assertEqual(result['native_lin_calls'],16)
        self.assertEqual(result['native_rf_fits'],0)
        self.assertTrue(result['three_classes_for_smooth_targets'])
