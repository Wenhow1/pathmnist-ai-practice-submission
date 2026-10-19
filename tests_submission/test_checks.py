"""New synthetic positive/negative tests; not original training or model tests."""
from pathlib import Path
import tempfile,unittest,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'checks'))
from verify_submission import cm_metrics,safe_path,same,load_json,selection,digest
class Checks(unittest.TestCase):
 def identity(self):return [[1 if i==j else 0 for j in range(9)] for i in range(9)]
 def test_identity(self):self.assertEqual(cm_metrics(self.identity())['macro_f1'],1)
 def test_count(self):self.assertEqual(cm_metrics(self.identity())['n'],9)
 def test_errors(self):
  m=self.identity();m[0][0]=0;m[0][1]=1;self.assertLess(cm_metrics(m)['macro_f1'],1)
 def test_empty(self):
  with self.assertRaises(ValueError):cm_metrics([[0]*9 for _ in range(9)])
 def test_wrong_shape(self):
  with self.assertRaises(ValueError):cm_metrics([[1]])
 def test_negative(self):
  m=self.identity();m[0][0]=-1
  with self.assertRaises(ValueError):cm_metrics(m)
 def test_bool_count(self):
  m=self.identity();m[0][0]=True
  with self.assertRaises(ValueError):cm_metrics(m)
 def test_float_count(self):
  m=self.identity();m[0][0]=1.0
  with self.assertRaises(ValueError):cm_metrics(m)
 def test_metric(self):same(.5,.50000000001)
 def test_metric_error(self):
  with self.assertRaises(ValueError):same(.5,.8)
 def test_nonfinite(self):
  with self.assertRaises(ValueError):same(float('nan'),1)
 def test_boolean_metric(self):
  with self.assertRaises(ValueError):same(True,1)
 def test_safe_path(self):
  with tempfile.TemporaryDirectory() as td:self.assertEqual(safe_path(Path(td),'a/b'),Path(td).resolve()/'a/b')
 def test_parent_path(self):
  with self.assertRaises(ValueError):safe_path(Path('/tmp'),'../secret')
 def test_absolute_path(self):
  with self.assertRaises(ValueError):safe_path(Path('/tmp'),'/etc/passwd')
 def test_windows_path(self):
  with self.assertRaises(ValueError):safe_path(Path('/tmp'),'C:/x')
 def test_symlink_escape(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'escape').symlink_to('/etc',target_is_directory=True)
   with self.assertRaises(ValueError):safe_path(root,'escape/passwd')
 def test_json_nan(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x.json';p.write_text('{"x": NaN}')
   with self.assertRaises(ValueError):load_json(p)
 def test_digest_change(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x';p.write_text('old');v=digest(p);p.write_text('new');self.assertNotEqual(v,digest(p))
 def test_validation_selection_not_test(self):
  rows=[]
  for g,v,t in [('A',.996,10),('B',.9986,12),('C',.992,1),('D',.9978,2)]:
   for seed in (42,2026,3407):rows.append({'group':g,'id':f'{g}_seed{seed}','val_macro_f1':v+(seed==2026)*.0001,'cpu_median_ms':t,'macro_f1':1 if g=='C' else 0})
  s=selection(rows);self.assertEqual(s['admissible'],['A','B','D']);self.assertEqual(s['run_id'],'D_seed2026')
 def test_missing_seeds(self):
  with self.assertRaises(ValueError):selection([])
if __name__=='__main__':unittest.main()
