from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from ecofill.operation import load_config, plan
from ecofill.operation.cli import main


class PublicOperationTests(unittest.TestCase):
    def test_four_family_public_fixture_validates_and_plans_with_saved_budget(self):
        root=Path(__file__).resolve().parents[1]
        doc=load_config(root/'examples/operation/config.json').to_dict()
        for source in doc['sources'].values(): source['uri']=str(root/source['uri'])
        planned=plan(doc).to_dict()
        self.assertEqual(planned['planned_gaps'],12)
        self.assertEqual(planned['method_call_upper_bound'],15)
        self.assertFalse(planned['budget_exceeded'])
        self.assertEqual(len(planned['partitions']),4)

    def test_installed_cli_offline_validation_and_dataset_edit_need_no_model(self):
        root=Path(__file__).resolve().parents[1]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['validate','--config',str(root/'examples/operation/config.json')]),0)
        self.assertEqual(json.loads(output.getvalue())['status'],'valid')
        doc=load_config(root/'examples/operation/config.json').to_dict()
        doc['datasets']['ctd']['enabled']=False
        del doc['datasets']['weather']
        for source in doc['sources'].values(): source['uri']=str(root/source['uri'])
        planned=plan(doc).to_dict()
        self.assertEqual(set(planned['inputs']),{'flux','biomet'})
        self.assertEqual(len(planned['partitions']),2)
