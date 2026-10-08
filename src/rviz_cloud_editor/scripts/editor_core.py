"""Pure geometry and atomic storage for reversible static-cloud brush edits."""
import json
import os
from pathlib import Path
import numpy as np


def apply_strokes(points, ground, strokes):
    keep = np.ones(len(points), dtype=bool)
    height = points[:,2]-ground
    for stroke in strokes:
        centers = np.asarray(stroke['centers'], dtype=float)
        if not len(centers):
            continue
        r = float(stroke['radius'])
        eligible = keep & (points[:,0] >= centers[:,0].min()-r) & (points[:,0] <= centers[:,0].max()+r)
        eligible &= (points[:,1] >= centers[:,1].min()-r) & (points[:,1] <= centers[:,1].max()+r)
        if not stroke['all_heights']:
            eligible &= (height >= .06) & (height <= stroke['max_height'])
        indices = np.flatnonzero(eligible)
        q = points[indices,:2]
        hit = np.zeros(len(q), dtype=bool)
        # Capsules join consecutive cursor positions, even after a slow GUI frame.
        for i,b in enumerate(centers):
            a = centers[max(0,i-1)]
            v = b-a
            length = float(v@v)
            t = np.clip((q-a)@v/length,0.,1.) if length > 1e-12 else np.zeros(len(q))
            hit |= np.sum((q-(a+t[:,None]*v))**2,axis=1) <= r*r
        keep[indices[hit]] = False
    return keep


def brush_command(cmd):
    values = [float(cmd[k]) for k in ('x','y','radius','max_height')]
    if not np.isfinite(values).all():
        raise ValueError('笔刷参数必须为有限数值')
    x,y,r,h = values
    if abs(x)>10000 or abs(y)>10000 or not .05<=r<=1.5 or not .08<=h<=2.:
        raise ValueError('笔刷参数超出范围')
    if not isinstance(cmd.get('all_heights'),bool) or not isinstance(cmd.get('stroke'),str) or not cmd['stroke']:
        raise ValueError('无效笔画')
    return {'id':cmd['stroke'],'centers':[[x,y]],'radius':r,'max_height':h,'all_heights':cmd['all_heights']}


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def save_pcd(path, points):
    a = np.asarray(points,dtype='<f4')
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():
        raise ValueError('无效点云，未保存')
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
              f'WIDTH {len(a)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(a)}\nDATA binary\n').encode()
    atomic_bytes(path,header+a.tobytes())


def save_json(path, value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
