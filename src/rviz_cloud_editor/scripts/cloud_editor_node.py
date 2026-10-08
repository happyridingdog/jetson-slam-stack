#!/usr/bin/env python3
"""Edit only the saved-map obstacle stream; live obstacle sensing stays independent."""
import copy
import hashlib
import json
import time
from array import array
from datetime import datetime
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from nav2_msgs.srv import ClearEntireCostmap
from editor_core import apply_strokes,brush_command,save_json,save_pcd


def xyz_records(msg):
    offsets={f.name:f.offset for f in msg.fields if f.name in ('x','y','z') and f.datatype==7 and f.count==1}
    if set(offsets)!=set(('x','y','z')):raise ValueError('编辑器要求 float32 XYZ 点云')
    dtype=np.dtype({'names':['x','y','z'],'formats':[('>' if msg.is_bigendian else '<')+'f4']*3,'offsets':[offsets[k] for k in ('x','y','z')],'itemsize':msg.point_step})
    return np.ndarray((msg.height,msg.width),dtype=dtype,buffer=msg.data,strides=(msg.row_step,msg.point_step)).reshape(-1)


def points_of(msg):
    p=xyz_records(msg)
    return np.column_stack([p[k] for k in ('x','y','z')])


def select(msg,keep):
    # Select raw records, preserving arbitrary extra fields and endianness.
    records=np.ndarray((msg.height,msg.width,msg.point_step),dtype=np.uint8,buffer=msg.data,strides=(msg.row_step,msg.point_step,1)).reshape(-1,msg.point_step)
    result=copy.copy(msg);result.height=1;result.width=int(np.count_nonzero(keep));result.row_step=result.width*result.point_step;result.data=array('B',records[keep].tobytes())
    return result


class CloudEditor(Node):
    def __init__(self):
        super().__init__('map_cloud_editor')
        self.declare_parameter('input_topic','/nav_map_points_unedited')
        self.declare_parameter('output_topic','/nav_map_points')
        self.declare_parameter('map_path','')
        self.declare_parameter('output_directory','/home/jetson/.jszr/map/manual_edits')
        self.declare_parameter('require_navigation_idle',True)
        self.map_path=str(self.get_parameter('map_path').value)
        self.output=Path(str(self.get_parameter('output_directory').value))
        self.require_idle=bool(self.get_parameter('require_navigation_idle').value)
        self.strokes=[];self.full=None;self.full_points=None;self.fingerprint=None;self.state_path=None
        self.plane=None;self.surface=None;self.source=None;self.source_points=None;self.source_sig=None
        self.mask=None;self.changed=False;self.clear_due=0.;self.state={};self.state_time=0.;self.message='等待地图和地面模型';self.last_publish=0.;self.blocked_stroke=None
        latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.publisher=self.create_publisher(PointCloud2,str(self.get_parameter('output_topic').value),1)
        self.status=self.create_publisher(String,'/map_editor/status',latched)
        self.create_subscription(PointCloud2,str(self.get_parameter('input_topic').value),self.on_source,1)
        self.create_subscription(PointCloud2,'/map_points_3d',self.on_full,1)
        self.create_subscription(Vector3Stamped,'/navigation/ground_plane',self.on_plane,latched)
        self.create_subscription(PointCloud2,'/navigation/ground_surface',self.on_surface,latched)
        self.create_subscription(String,'/navigation/status',self.on_navigation,latched)
        self.create_subscription(String,'/map_editor/command',self.on_command,20)
        self.pause=self.create_client(Trigger,'/navigation/cancel')
        self.clear={k:self.create_client(ClearEntireCostmap,f'/{k}_costmap/clear_entirely_{k}_costmap') for k in ('local','global')}
        self.clear_pending=[];self.create_timer(.2,self.tick);self.create_timer(2.,self.publish_status)
        self.publish_status()

    def publish_status(self):
        self.status.publish(String(data=self.message))

    def say(self,text):
        self.message=text;self.publish_status()

    def on_navigation(self,msg):
        try:self.state=json.loads(msg.data);self.state_time=time.monotonic()
        except ValueError:pass

    def idle(self):
        return not self.require_idle or (time.monotonic()-self.state_time<2. and self.state and not self.state.get('active',True) and not self.state.get('queued',True))

    def on_plane(self,msg):
        if msg.header.frame_id=='map' and np.isfinite([msg.vector.x,msg.vector.y,msg.vector.z]).all():
            self.plane=np.array([msg.vector.x,msg.vector.y,msg.vector.z]);self.mask=None

    def on_surface(self,msg):
        if msg.header.frame_id!='map':return
        a=points_of(msg);self.surface=None
        if len(a) and np.isfinite(a).all():
            idx=np.floor(a[:,:2]/.5).astype(int);origin=idx.min(0);idx-=origin;shape=idx.max(0)+1
            if np.prod(shape)<1000000:
                grid=np.full(shape,np.nan);grid[idx[:,0],idx[:,1]]=a[:,2];self.surface=(origin,grid)
        self.mask=None

    def ground(self,a):
        if self.plane is None:raise ValueError('等待地面模型')
        g=a[:,:2]@self.plane[:2]+self.plane[2]
        if self.surface is not None:
            origin,grid=self.surface;idx=np.floor(a[:,:2]/.5).astype(int)-origin;valid=(idx>=0).all(1)&(idx<grid.shape).all(1);rows=np.flatnonzero(valid);v=grid[idx[valid,0],idx[valid,1]];finite=np.isfinite(v);g[rows[finite]]=np.maximum(g[rows[finite]],v[finite])
        return g

    def on_full(self,msg):
        if msg.header.frame_id!='map':return
        a=points_of(msg)
        if not np.isfinite(a).all():self.say('地图包含无效点，暂不允许编辑');return
        digest=hashlib.sha256(a.astype('<f4').tobytes()).hexdigest()
        if digest==self.fingerprint:
            self.full=msg;self.full_points=a;return
        if self.fingerprint:self.persist()
        self.full=msg;self.full_points=a
        self.fingerprint=digest;self.state_path=self.output/('edits_'+digest[:16]+'.json');self.strokes=[];self.mask=None
        if self.state_path.exists():
            try:
                saved=json.loads(self.state_path.read_text())
                if saved['fingerprint']!=digest:raise ValueError('地图指纹不匹配')
                for s in saved['strokes']:
                    for x,y in s['centers']:brush_command({'stroke':s['id'],'x':x,'y':y,**{k:s[k] for k in ('radius','max_height','all_heights')}})
                self.strokes=saved['strokes']
            except Exception as e:self.say('编辑记录未载入：'+str(e));return
        self.say(f'地图已加载，已恢复 {len(self.strokes)} 笔编辑。停车后可涂抹。')

    def on_source(self,msg):
        if msg.header.frame_id!='map':return
        a=points_of(msg);sig=hashlib.sha256(a.astype('<f4').tobytes()).digest()
        self.source=msg
        if sig!=self.source_sig:self.source_points=a;self.source_sig=sig;self.mask=None
        self.publish_cloud()

    def publish_cloud(self):
        if self.source is None:return
        # Never restamp a stale observation while the saved-map source is gone.
        age=(self.get_clock().now().nanoseconds-self.source.header.stamp.sec*10**9-self.source.header.stamp.nanosec)/1e9
        if not -.05<=age<=5.:return
        if not self.strokes:self.publisher.publish(self.source);return
        if self.plane is None:return
        if self.mask is None:self.mask=apply_strokes(self.source_points,self.ground(self.source_points),self.strokes)
        self.publisher.publish(select(self.source,self.mask));self.last_publish=time.monotonic()

    def persist(self):
        if self.state_path:
            backup=self.output/('original_'+self.fingerprint[:16]+'.pcd')
            if self.strokes and self.full_points is not None and not backup.exists():
                save_pcd(backup,self.full_points)
            save_json(self.state_path,{'fingerprint':self.fingerprint,'source_map':self.map_path,'strokes':self.strokes})

    def on_command(self,msg):
        try:
            cmd=json.loads(msg.data);op=cmd.get('op')
            if op=='pause':
                if not self.require_idle:
                    self.say('当前为离线地图编辑，无需暂停导航');return
                if not self.pause.service_is_ready():raise ValueError('导航暂停服务不可用')
                self.pause.call_async(Trigger.Request());self.say('已请求暂停导航，请等待车辆停稳后涂抹');return
            if op not in ('paint','end','undo','save'):return
            if op=='end':
                self.persist()
                if self.strokes and self.strokes[-1]['id']==cmd.get('stroke'):self.clear_due=time.monotonic()+.5
                return
            if not self.idle():
                if op=='paint':self.blocked_stroke=cmd.get('stroke')
                raise ValueError('正在导航或状态未就绪，请先点击“暂停导航”并等待车辆停稳')
            if self.full is None or self.plane is None:raise ValueError('等待完整地图及地面模型')
            if op=='paint':
                stroke=brush_command(cmd)
                if self.blocked_stroke==stroke['id']:raise ValueError('请松开鼠标后重新涂抹')
                if len(self.strokes)>2000:raise ValueError('编辑笔画过多，请另存地图后重新加载')
                if self.strokes and self.strokes[-1]['id']==stroke['id']:
                    previous=self.strokes[-1]
                    if any(previous[k]!=stroke[k] for k in ('radius','max_height','all_heights')):raise ValueError('请松开鼠标后修改笔刷')
                    previous['centers'].append(stroke['centers'][0])
                else:self.strokes.append(stroke)
                self.mask=None;self.changed=True
                self.say(f'已编辑 {len(self.strokes)} 笔；松开鼠标后更新障碍层。Ctrl+Z 可撤销。')
            elif op=='undo':
                if self.strokes:self.strokes.pop();self.mask=None;self.changed=True;self.persist();self.clear_due=time.monotonic()+.5
                self.say(f'已撤销，剩余 {len(self.strokes)} 笔')
            elif op=='save':
                keep=apply_strokes(self.full_points,self.ground(self.full_points),self.strokes)
                dest=self.output/('map_cleaned_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.pcd')
                save_pcd(dest,self.full_points[keep]);self.persist();save_json(dest.with_suffix('.json'),{'source_map':self.map_path,'fingerprint':self.fingerprint,'removed_points':int(np.sum(~keep)),'remaining_points':int(np.sum(keep)),'edit_record':str(self.state_path)})
                self.say(f'已另存 {int(np.sum(keep))} 点，擦除 {int(np.sum(~keep))} 点：\n{dest}')
        except Exception as e:self.say(str(e));self.get_logger().warning(str(e))

    def tick(self):
        if self.changed:
            self.publish_cloud();self.changed=False
        if self.clear_due and time.monotonic()>=self.clear_due and self.idle():
            self.publish_cloud()
            if all(c.service_is_ready() for c in self.clear.values()):
                self.clear_pending=[c.call_async(ClearEntireCostmap.Request()) for c in self.clear.values()];self.clear_due=0.
        self.clear_pending=[f for f in self.clear_pending if not f.done()]


def main():
    rclpy.init();node=CloudEditor()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        node.persist();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
