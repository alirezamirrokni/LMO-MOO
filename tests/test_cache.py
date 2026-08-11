from __future__ import annotations
import copy, tempfile, unittest
from pathlib import Path
from src.cache import CacheManager

class CacheTests(unittest.TestCase):
    def _cfg(self):
        return {
            'experiment': {'epochs': 200, 'average_last_epochs': 10, 'methods': ['A'], 'datasets': ['cityscapes'], 'seeds': [1]},
            'model': {'width': 1.0},
            'data': {'cityscapes': {'root': 'city', 'batch_size': 8}, 'nyuv2': {'root': 'nyu', 'batch_size': 2}},
            'methods': {'A': {'kind': 'alg1', 'gamma': 1e-4, 'by_dataset': {'cityscapes': {'eta': .5}, 'nyuv2': {'eta': .1}}}, 'B': {'kind': 'alg4', 'gamma': 2e-4}},
        }
    def test_unrelated_selection_and_method_changes_do_not_invalidate(self):
        with tempfile.TemporaryDirectory() as td:
            c1=self._cfg(); f1=CacheManager(Path(td),'main_comparison',c1).fingerprint('A','cityscapes',1)
            c2=copy.deepcopy(c1); c2['experiment']['methods'].append('B'); c2['experiment']['seeds'].append(2); c2['methods']['B']['gamma']=9e-4; c2['data']['nyuv2']['root']='other'
            f2=CacheManager(Path(td),'main_comparison',c2).fingerprint('A','cityscapes',1)
            self.assertEqual(f1,f2)
    def test_other_dataset_override_does_not_invalidate(self):
        with tempfile.TemporaryDirectory() as td:
            c1=self._cfg(); f1=CacheManager(Path(td),'main_comparison',c1).fingerprint('A','cityscapes',1)
            c2=copy.deepcopy(c1); c2['methods']['A']['by_dataset']['nyuv2']['eta']=.9
            f2=CacheManager(Path(td),'main_comparison',c2).fingerprint('A','cityscapes',1)
            self.assertEqual(f1,f2)

    def test_current_dataset_override_does_invalidate(self):
        with tempfile.TemporaryDirectory() as td:
            c1=self._cfg(); f1=CacheManager(Path(td),'main_comparison',c1).fingerprint('A','cityscapes',1)
            c2=copy.deepcopy(c1); c2['methods']['A']['by_dataset']['cityscapes']['eta']=.9
            f2=CacheManager(Path(td),'main_comparison',c2).fingerprint('A','cityscapes',1)
            self.assertNotEqual(f1,f2)

    def test_relevant_changes_do_invalidate(self):
        with tempfile.TemporaryDirectory() as td:
            c1=self._cfg(); f1=CacheManager(Path(td),'main_comparison',c1).fingerprint('A','cityscapes',1)
            for mutate in ('method','data','epochs'):
                c2=copy.deepcopy(c1)
                if mutate=='method': c2['methods']['A']['gamma']=2e-4
                elif mutate=='data': c2['data']['cityscapes']['root']='other'
                else: c2['experiment']['epochs']=201
                f2=CacheManager(Path(td),'main_comparison',c2).fingerprint('A','cityscapes',1)
                self.assertNotEqual(f1,f2)

if __name__=='__main__': unittest.main()
