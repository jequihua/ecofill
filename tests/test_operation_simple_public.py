from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from ecofill.operation import plan, run, simple_example as example  # Installed with the package.
from ecofill.operation.cli import main


class PublicSimpleOperationTests(unittest.TestCase):
    def test_concise_weather_example_validates_offline_plans_and_runs_lin(self):
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            example.prepare('work')
            with chdir('work'):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(['validate', '--config', 'config.json']), 0)
                validated = json.loads(output.getvalue())
                self.assertEqual((validated['status'], validated['input_facts']), ('valid', 'pending'))
                planned = plan('config.json').to_dict()
                self.assertEqual((planned['planned_gaps'], planned['method_call_upper_bound']), (7, 4))
                self.assertEqual(run('config.json', run_id='first')['filled_cells'], 10)
                checked = example.readback('output/runs/first')
        self.assertEqual((checked['observations_checked'], checked['unresolved']),
                         (460, {'boundary_gap_unresolved': 2, 'no_route': 1}))

    def test_engineer_spelling_matches_concise_and_bad_options_explain_safely(self):
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            example.prepare('work')
            with chdir('work'):
                engineer = plan('config-engineer.json').config.to_dict()
                concise = json.loads(Path('config.json').read_text())
                concise['datasets'] = {'davis_weather': concise['datasets']['weather'] | {'enabled': True, 'columns': {
                    'temp_out': {'unit': 'degC'}, 'bar': {'unit': 'hPa'}, 'dew_point': {'unit': 'degC'}}}}
                resolved = plan(concise).config.to_dict()
                engineer.pop('resolution'); resolved.pop('resolution')
                self.assertEqual(engineer, resolved)
                bad = json.loads(Path('config.json').read_text())
                bad['datasets']['weather'].update(cadence='half an hour', secret_token='do-not-echo')
                Path('bad.json').write_text(json.dumps(bad))
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(['validate', '--config', 'bad.json']), 1)
        error = json.loads(output.getvalue())
        self.assertEqual((error['status'], error['error_type'], error['code'], error['location']),
                         ('error', 'ValueError', 'unknown_field', '/datasets/weather'))
        self.assertNotIn('do-not-echo', output.getvalue())
        self.assertNotIn('secret_token', output.getvalue())


if __name__ == '__main__':
    unittest.main()
