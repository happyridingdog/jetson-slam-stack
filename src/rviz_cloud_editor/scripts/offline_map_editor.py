#!/usr/bin/env python3
"""Static PCD + ground filtering + brush editing; no sensor/localization/navigation nodes."""
import hashlib
import importlib.util
import json
from array import array
from pathlib import Path
import numpy as np
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy
from sensor_msgs.msg import PointCloud2,PointField
from geometry_msgs.msg import Vector3Stamped,TransformStamped
from tf2_ros import StaticTransformBroadcaster
from ament_index_python.packages import get_package_prefix
from cloud_editor_node import CloudEditor,points_of


def read_xyz_pcd(path):
    with Path(path).open('rb') as f:
        header={}
        while True:
            line=f.readline()
            if not line:raise ValueError('PCD header incomplete')
            words=line.decode('ascii').strip().split()
            if not words or words[0].startswith('#'):continue
            header[words[0]]=words[1:]
            if words[0]=='DATA':break
        if header['FIELDS']!=['x','y','z'] or header['SIZE']!=['4']*3 or header['TYPE']!=['F']*3 or header['DATA']!=['binary']:
            raise ValueError('Offline editor expects binary XYZ float32 PCD')
        a=np.frombuffer(f.read(),dtype='<f4').reshape(-1,3).copy()
        if len(a)!=int(header['POINTS'][0]) or not np.isfinite(a).all():raise ValueError('Invalid PCD payload')
        return a


def message(a,node):
    m=PointCloud2();m.header.frame_id='map';m.header.stamp=node.get_clock().now().to_msg();m.height=1;m.width=len(a)
    m.fields=[PointField(name=k,offset=i*4,datatype=7,count=1) for i,k in enumerate(('x','y','z'))]
    m.point_step=12;m.row_step=12*len(a);m.is_dense=True;m.data=array('B',np.asarray(a,dtype='<f4').tobytes());return m


def main():
    rclpy.init();editor=CloudEditor()
    if editor.require_idle:raise RuntimeError('Offline mode requires require_navigation_idle:=false')
    editor.declare_parameter('min_height_above_ground',.08);editor.declare_parameter('max_height_above_ground',2.)
    source=Path(editor.map_path);cloud=read_xyz_pcd(source);chosen=source
    # Online PCL voxelization may change point ordering / one boundary voxel.
    # Restore its exact original snapshot so existing brush records still match.
    for record in sorted(editor.output.glob('edits_*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        data=json.loads(record.read_text())
        if data.get('source_map')!=str(source):continue
        snapshot=editor.output/('original_'+data['fingerprint'][:16]+'.pcd')
        if snapshot.exists():
            candidate=read_xyz_pcd(snapshot)
            if hashlib.sha256(candidate.astype('<f4').tobytes()).hexdigest()==data['fingerprint']:
                cloud=candidate;chosen=snapshot;break
    full=message(cloud,editor)
    helper=Path(get_package_prefix('lightweight_fusion_bringup'))/'lib/lightweight_fusion_bringup/nav_cloud_retimestamp.py'
    spec=importlib.util.spec_from_file_location('offline_ground_filter',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);Filter=mod.NavCloudRetimestamp
    xyz=Filter.xyz_view(full);plane=Filter.fit_ground_plane(xyz,-.35);surface=Filter.fit_ground_surface(xyz,plane)
    ground_msg=Vector3Stamped();ground_msg.header=full.header;ground_msg.vector.x,ground_msg.vector.y,ground_msg.vector.z=plane
    editor.on_plane(ground_msg);surface_msg=message(surface,editor);editor.on_surface(surface_msg)
    h=cloud[:,2]-editor.ground(cloud);low=float(editor.get_parameter('min_height_above_ground').value);high=float(editor.get_parameter('max_height_above_ground').value)
    visible=message(cloud[(h>=low)&(h<=high)],editor)
    qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ground_pub=editor.create_publisher(Vector3Stamped,'/navigation/ground_plane',qos);surface_pub=editor.create_publisher(PointCloud2,'/navigation/ground_surface',qos)
    ground_pub.publish(ground_msg);surface_pub.publish(surface_msg)
    tf=StaticTransformBroadcaster(editor);tr=TransformStamped();tr.header=full.header;tr.child_frame_id='offline_map_origin';tr.transform.rotation.w=1.;tf.sendTransform(tr)
    def publish():
        stamp=editor.get_clock().now().to_msg();full.header.stamp=stamp;visible.header.stamp=stamp;editor.on_full(full);editor.on_source(visible)
    publish();editor.say(f'离线地图编辑：已恢复 {len(editor.strokes)} 笔，直接涂抹即可。\n显示离地 {low:.2f}～{high:.2f} m 点云；定位、导航和雷达均未启动。')
    editor.create_timer(1.,publish)
    editor.get_logger().info(f'OFFLINE ONLY: source={source}; snapshot={chosen}; full={len(cloud)}; ground_filtered={visible.width}; restored_strokes={len(editor.strokes)}')
    try:rclpy.spin(editor)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        editor.persist();editor.destroy_node()
        if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
