import importlib.util
from pathlib import Path
import struct
import numpy as np
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
spec=importlib.util.spec_from_file_location('navcloud',Path(__file__).resolve().parents[1]/'scripts/nav_cloud_retimestamp.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Cloud=module.NavCloudRetimestamp

class NavigationCloud(unittest.TestCase):
    def cloud(self):
        fields=[PointField(name=n,offset=4*i,datatype=PointField.FLOAT32,count=1)
                for i,n in enumerate(['x','y','z','intensity'])]
        return PointCloud2(header=Header(frame_id='sensor',stamp=Time(sec=10)),height=1,width=3,
            point_step=16,row_step=48,fields=fields,data=struct.pack('<12f',-.21,0.,.3,11., .6,0.,-.1,22., float('nan'),0.,0.,33.))
    def node(self):
        n=Cloud.__new__(Cloud)
        n.target_frame='';n.max_cloud_age_s=.5;n.point_stride=1;n.exclude_footprint=True;n.base_frame='base_link'
        n.height_enabled=False
        n.dropped=n.forwarded=0;n.publisher=Mock();n.tf_buffer=Mock()
        tf=TransformStamped();tf.transform.rotation.w=1.;tf.transform.translation.x=.21
        n.tf_buffer.lookup_transform.return_value=tf
        n.get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10050000000))
        n.get_logger=lambda:Mock()
        return n
    def test_live_sensor_frame_stamp_and_intensity_survive_filtering(self):
        n=self.node();c=self.cloud();n.on_cloud(c)
        output=n.publisher.publish.call_args.args[0]
        self.assertEqual(output.header.frame_id,'sensor')
        self.assertEqual(output.header.stamp,c.header.stamp)
        self.assertEqual(output.width,1)
        x,y,z,intensity=struct.unpack('<4f',bytes(output.data))
        self.assertAlmostEqual(x,.6,places=6);self.assertEqual(intensity,22.)
        self.assertEqual(c.width,3)
    def test_stale_cloud_is_not_restamped_as_fresh(self):
        n=self.node();c=self.cloud();c.header.stamp.sec=9;n.on_cloud(c)
        n.publisher.publish.assert_not_called()
    def test_zero_stamp_is_rejected(self):
        n=self.node();c=self.cloud();c.header.stamp.sec=0;n.on_cloud(c)
        n.publisher.publish.assert_not_called()
    def test_static_cloud_is_not_footprint_filtered(self):
        n=self.node();n.exclude_footprint=False;c=self.cloud();n.on_cloud(c)
        self.assertEqual(n.publisher.publish.call_args.args[0].width,2)
    def test_empty_scan_is_valid_and_keeps_sensor_origin(self):
        n=self.node();c=self.cloud();c.width=0;c.row_step=0;c.data=b'';n.on_cloud(c)
        self.assertEqual(n.publisher.publish.call_args.args[0].width,0)

    def height_node(self):
        n=self.node();n.height_enabled=True;n.height_frame='map';n.estimate_ground=False
        n.ground_plane=(0.,0.,-.35);n.min_height=.08;n.max_height=2.;n.exclude_footprint=False
        return n

    def points(self, xyz, frame='map'):
        c=self.cloud();c.header.frame_id=frame;c.width=len(xyz);c.row_step=16*len(xyz)
        c.data=struct.pack('<'+'f'*4*len(xyz),*[v for i,p in enumerate(xyz) for v in (*p,float(i+1))])
        return c

    def test_ground_and_roof_removed_but_low_and_hanging_obstacles_kept(self):
        n=self.height_node();c=self.points([(1.,0.,-.35),(1.,0.,-.23),(1.,0.,1.2),(1.,0.,4.5)])
        n.on_cloud(c);out=n.publisher.publish.call_args.args[0]
        self.assertEqual(Cloud.xyz_view(out)['intensity'].tolist(),[2.,3.])
        self.assertEqual(out.header,c.header)

    def test_sensor_translation_and_tilt_do_not_change_height_classification(self):
        n=self.height_node();tf=TransformStamped();angle=.12
        tf.transform.rotation.y=float(np.sin(angle/2));tf.transform.rotation.w=float(np.cos(angle/2));tf.transform.translation.z=.65
        n.tf_buffer.lookup_transform.return_value=tf
        r=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
        world=np.array([[1.,0.,-.35],[1.,0.,-.23],[1.,0.,1.2],[1.,0.,4.5]])
        sensor=(world-[0.,0.,.65])@r;c=self.points(sensor,'sensor');n.on_cloud(c)
        out=n.publisher.publish.call_args.args[0]
        self.assertEqual(Cloud.xyz_view(out)['intensity'].tolist(),[2.,3.])
        self.assertEqual(out.header,c.header)
        np.testing.assert_allclose(Cloud.xyz_view(out)['z'],sensor[1:3,2],rtol=1e-6)

    def test_missing_ground_model_does_not_publish_unfiltered_cloud(self):
        n=self.height_node();n.ground_plane=None;n.on_cloud(self.points([(1.,0.,-.35)]))
        n.publisher.publish.assert_not_called()

    def test_robust_saved_floor_fit_preserves_small_tilt(self):
        x,y=np.meshgrid(np.linspace(-3,3,30),np.linspace(-3,3,30));x=x.ravel();y=y.ravel()
        floor=.006*x-.003*y-.29
        points=np.vstack((np.c_[x,y,floor],np.c_[x,y,floor+4.8],np.c_[x[:60],y[:60],floor[:60]+.2]))
        xyz=Cloud.xyz_view(self.points(points));plane=Cloud.fit_ground_plane(xyz,-.35)
        np.testing.assert_allclose(plane,[.006,-.003,-.29],atol=.001)

    def test_roof_only_map_cannot_be_mistaken_for_floor(self):
        xyz=Cloud.xyz_view(self.points([(float(i%20),float(i//20),4.5) for i in range(400)]))
        with self.assertRaises(ValueError):Cloud.fit_ground_plane(xyz,-.35)

    def local_surface(self, node, points):
        surface = Cloud.fit_ground_surface(Cloud.xyz_view(self.points(points)), node.ground_plane)
        node.on_ground_surface(self.points(surface))
        return surface

    def test_local_floor_warp_removed_without_erasing_low_box_or_pole(self):
        n=self.height_node();n.ground_plane=(0.,0.,0.)
        x,y=np.meshgrid(np.arange(0.,8.,.1),np.arange(-3.,3.,.1));x=x.ravel();y=y.ravel()
        floor=.11*np.clip((x-1.)/5.,0.,1.)
        # A 12 cm box covers its underlying ground returns.
        box=(x>3.)&(x<4.)&(y>1.)&(y<2.)
        points=np.c_[x,y,floor+np.where(box,.12,0.)]
        self.local_surface(n,points)
        c=self.points([(6.25,0.,.12),(6.25,0.,.23),(3.5,1.5,.175),(6.25,0.,4.9)])
        n.on_cloud(c)
        self.assertEqual(Cloud.xyz_view(n.publisher.publish.call_args.args[0])['intensity'].tolist(),[2.,3.])

    def test_disconnected_twelve_cm_platform_is_not_a_floor_seed(self):
        n=self.height_node();n.ground_plane=(0.,0.,0.)
        x,y=np.meshgrid(np.arange(-4.,4.,.1),np.arange(-4.,4.,.1));x=x.ravel();y=y.ravel()
        z=np.where((np.abs(x)<1.5)&(np.abs(y)<1.5),.12,0.)
        self.local_surface(n,np.c_[x,y,z]);n.on_cloud(self.points([(0.,0.,.12)]))
        self.assertEqual(n.publisher.publish.call_args.args[0].width,1)

    def test_sparse_floor_and_wrong_frame_surface_do_not_raise_ground(self):
        n=self.height_node();n.ground_plane=(0.,0.,0.)
        surface=self.local_surface(n,[(float(i),0.,.1) for i in range(20)])
        self.assertEqual(len(surface),0)
        n.on_ground_surface(self.points([(0.,0.,.1)],frame='odom'))
        self.assertIsNone(n.ground_surface)
        n.on_cloud(self.points([(0.,0.,.12)]));self.assertEqual(n.publisher.publish.call_args.args[0].width,1)

    def test_shared_local_surface_preserves_live_frame_and_stamp(self):
        n=self.height_node();n.ground_plane=(0.,0.,0.)
        x,y=np.meshgrid(np.arange(0.,8.,.1),np.arange(-3.,3.,.1));x=x.ravel();y=y.ravel()
        self.local_surface(n,np.c_[x,y,.11*np.clip((x-1.)/5.,0.,1.)])
        tf=TransformStamped();tf.transform.rotation.w=1.;tf.transform.translation.z=.5
        n.tf_buffer.lookup_transform.return_value=tf
        c=self.points([(6.25,0.,-.38),(6.25,0.,-.27)],frame='sensor');n.on_cloud(c)
        out=n.publisher.publish.call_args.args[0];self.assertEqual(out.header,c.header)
        self.assertEqual(Cloud.xyz_view(out)['intensity'].tolist(),[2.])

    def test_latched_surface_arriving_before_plane_is_kept(self):
        from geometry_msgs.msg import Vector3Stamped
        n=self.height_node();n.ground_plane=None
        n.on_ground_surface(self.points([(0.25,0.25,-.25)]))
        model=n.ground_surface
        plane=Vector3Stamped();plane.header.frame_id='map';plane.vector.z=-.35
        n.on_ground_plane(plane)
        self.assertIs(n.ground_surface,model)

    def test_delayed_tf_retries_same_scan_without_restamping(self):
        n=self.height_node();tf=TransformStamped();tf.transform.rotation.w=1.
        n.tf_buffer.lookup_transform.side_effect=[module.ExtrapolationException('TF still arriving'),tf]
        c=self.points([(1.,0.,-.23)],'sensor');n.on_cloud(c)
        n.publisher.publish.assert_not_called();self.assertIs(n.pending_cloud,c)
        n.forward_pending_cloud()
        self.assertEqual(n.publisher.publish.call_args.args[0].header,c.header)
        self.assertIsNone(n.pending_cloud)

if __name__=='__main__':unittest.main()
