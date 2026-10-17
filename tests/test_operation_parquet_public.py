"""Public installed-wheel local Parquet smoke; no native model or Google access."""
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ecofill.operation import load_config, parquet_example as example
from ecofill.operation.cli import main


class PublicParquetTests(unittest.TestCase):
    def test_snapshot_publication_failures_and_missing_reader(self):
        checked=self.correction('publication_checks')
        self.assertEqual(checked['fault_runs'],40)
        self.assertEqual(checked['missing_reader_runs'],8)
        self.assertEqual(checked['no_fault_runs'],8)
        self.assertEqual(checked['secondary_terminal_runs'],8)
        self.assertEqual(checked['snapshot_retry_attempts'],0)

    def test_source_key_snapshots_and_free_form_metadata(self):
        # Grouped RF runs in its own installed process; the core public guard
        # continues to prohibit optional model imports in the parent suite.
        import ecofill
        import pandas
        code=r'''import importlib.abc,json,runpy,socket,sys
from pathlib import Path
sys.path[:0]=sys.argv[1:3]
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'torch','tabpfn','timesfm','google','huggingface_hub','safetensors'}:
            raise ImportError('foundation imports denied')
sys.meta_path.insert(0,Deny())
def denied(*a,**k):raise AssertionError('network denied')
socket.socket.connect=socket.create_connection=socket.getaddrinfo=denied
import ecofill
assert Path(ecofill.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
print(json.dumps(runpy.run_path(sys.argv[3])['snapshot_checks'](sys.argv[4])))
'''
        with tempfile.TemporaryDirectory() as temporary:
            child=subprocess.run([sys.executable,'-I','-S','-B','-c',code,
                str(Path(ecofill.__file__).resolve().parent.parent),str(Path(pandas.__file__).resolve().parent.parent),
                str(Path(__file__).resolve().parents[1]/'docker/qualify_parquet.py'),str(Path(temporary)/'snapshot')],
                capture_output=True,text=True)
            self.assertEqual(child.returncode,0,child.stderr)
            checked=json.loads(child.stdout)
        self.assertEqual(checked['lineage_api_cli_successes'],16)
        self.assertEqual(checked['outside_artifacts'],0)

    def test_local_writer_names_and_parent_links(self):
        checked=self.correction('writer_checks')
        self.assertEqual(checked['unsafe_entry_point_cases'],102)
        self.assertTrue(checked['adjacent_sentinel_unchanged'])

    def correction(self,name):
        import runpy
        harness=runpy.run_path(str(Path(__file__).resolve().parents[1]/'docker/qualify_parquet.py'))
        with tempfile.TemporaryDirectory() as temporary:
            return harness[name](Path(temporary)/name)

    def test_numeric_string_conversion_and_csv_control(self):
        checked=self.correction('numeric_checks')
        self.assertEqual(checked['rejected_numeric_cases'],20)
        self.assertTrue(checked['precedence_and_exact_keys'])

    def test_missing_arrow_runs_and_restored_reruns(self):
        checked=self.correction('dependency_checks')
        self.assertEqual(checked['simulated_missing_arrow']['orphan_claims'],0)
        self.assertTrue(checked['completed_idempotency_without_arrow']['arrow_absent_simulation'])

    def test_installed_cli_arrow_readback_and_resolved_replay(self):
        with tempfile.TemporaryDirectory() as temporary,chdir(temporary):
            example.prepare('work')
            with chdir('work'):
                def cli(*args):
                    with redirect_stdout(io.StringIO()) as output:
                        code=main(list(args))
                    self.assertEqual(code,0,output.getvalue())
                    return json.loads(output.getvalue())
                self.assertEqual(cli('validate','--config','inputs/config-linear.json')['input_facts'],'pending')
                cli('plan','--config','inputs/config-linear.json','--output','plan')
                result=cli('run','--config','inputs/config-linear.json','--run-id','first')
                self.assertEqual(result['filled_cells'],8)
                self.assertEqual(cli('inspect','--run-uri','output/linear/runs/first'),result)
                reader=subprocess.run([sys.executable,'-I','-B',str(Path(example.__file__).resolve()),'readback',
                    '--output','output/linear/runs/first'],check=True,capture_output=True,text=True)
                self.assertEqual(json.loads(reader.stdout)['observations'],284)
                example.replay_config('output/linear/runs/first','replay.json','output/replay')
                self.assertEqual(cli('run','--config','replay.json','--run-id','copy')['filled_cells'],8)
                self.assertTrue(example.readback('output/linear/runs/first','output/replay/runs/copy')['replay_bytes_equal'])

    def test_offline_format_and_remote_guards(self):
        doc={'datasets':{'weather':{'uri':'unavailable.parquet','cadence':'30min','targets':['bar'],
                                  'method':'LIN','max_gap':'2h'}},
             'output':{'uri':'new-output','format':'parquet','publication':'single_writer'}}
        config=load_config(doc)
        self.assertTrue(config.pending)
        self.assertEqual(config.to_dict()['sources']['weather']['format'],'parquet')
        doc['datasets']['weather']['uri']='gs://synthetic-bucket/input.parquet'
        with self.assertRaises(ValueError) as caught:load_config(doc)
        self.assertEqual(caught.exception.diagnostic['code'],'parquet_local_only')
