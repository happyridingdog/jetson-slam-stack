import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from editor_core import apply_strokes,brush_command,save_pcd,save_json
from cloud_editor_node import CloudEditor

class EditorTests(unittest.TestCase):
    def stroke(self,**extra):
        return brush_command(dict(op='paint',stroke='one',x=0.,y=0.,radius=.2,max_height=.35,all_heights=False,**extra))
    def test_near_ground_brush_preserves_floor_high_obstacle_and_outside_points(self):
        p=np.array([[0,0,0],[0,0,.12],[0,0,.5],[.3,0,.12]])
        self.assertEqual(apply_strokes(p,np.zeros(4),[self.stroke()]).tolist(),[True,False,True,True])
    def test_all_height_mode_and_undo(self):
        s=self.stroke();s['all_heights']=True;p=np.array([[0.,0.,2.],[.5,0.,.1],[0.,0.,-3.],[0.,0.,8.]])
        self.assertEqual(apply_strokes(p,np.zeros(len(p)),[s]).tolist(),[False,True,False,False])
        self.assertTrue(apply_strokes(p,np.zeros(len(p)),[]).all())
    def test_fast_drag_has_no_gaps(self):
        s=self.stroke();s['centers']=[[0,0],[2,0]]
        p=np.array([[1.,0.,.12],[1.,.3,.12]])
        self.assertEqual(apply_strokes(p,np.zeros(2),[s]).tolist(),[False,True])
    def test_local_floor_reference(self):
        p=np.array([[0,0,.2],[0,0,.5]])
        self.assertEqual(apply_strokes(p,np.array([.1,.1]),[self.stroke()]).tolist(),[False,True])
    def test_invalid_brush_rejected(self):
        for key,val in [('radius',-1.),('x',float('nan')),('max_height',30.),('all_heights','yes')]:
            c=dict(stroke='a',x=0.,y=0.,radius=.2,max_height=.35,all_heights=False);c[key]=val
            with self.assertRaises(ValueError):brush_command(c)
    def test_export_and_edit_record_preserve_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'source.pcd';source.write_bytes(b'original')
            a=np.array([[1,2,3],[4,5,6]],dtype=np.float32);save_pcd(p/'clean.pcd',a)
            raw=(p/'clean.pcd').read_bytes();header,payload=raw.split(b'DATA binary\n')
            self.assertIn(b'POINTS 2',header);np.testing.assert_array_equal(np.frombuffer(payload,dtype='<f4').reshape(-1,3),a)
            save_json(p/'mask.json',{'strokes':[self.stroke()]});self.assertEqual(json.loads((p/'mask.json').read_text())['strokes'][0]['id'],'one');self.assertEqual(source.read_bytes(),b'original')
    def test_navigation_running_refuses_edit(self):
        n=SimpleNamespace(idle=lambda:False,say=Mock(),get_logger=lambda:Mock(),blocked_stroke=None,strokes=[])
        CloudEditor.on_command(n,SimpleNamespace(data=json.dumps(dict(op='paint',stroke='test'))))
        self.assertEqual(n.strokes,[]);self.assertEqual(n.blocked_stroke,'test')
    def test_released_stroke_clears_costmap_even_after_cloud_was_published(self):
        n=SimpleNamespace(persist=Mock(),strokes=[{'id':'test'}],changed=False,clear_due=0.,say=Mock(),get_logger=lambda:Mock())
        CloudEditor.on_command(n,SimpleNamespace(data=json.dumps(dict(op='end',stroke='test'))))
        self.assertGreater(n.clear_due,0.);n.persist.assert_called_once()
if __name__=='__main__':unittest.main()
