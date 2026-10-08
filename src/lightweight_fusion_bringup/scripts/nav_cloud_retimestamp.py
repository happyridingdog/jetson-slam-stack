#!/usr/bin/env python3
"""Prepare navigation clouds without falsifying sensor pose or measurement time.

Live scans retain their acquisition stamp and sensor frame so costmap raytracing
starts at the lidar, not map origin. Static map clouds can be thinned separately.
"""

import copy
import zlib
from array import array
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Vector3Stamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


class NavCloudRetimestamp(Node):
    def __init__(self):
        super().__init__("nav_cloud_retimestamp")
        self.declare_parameter("input_topic", "/body_points")
        self.declare_parameter("output_topic", "/nav_body_points")
        self.declare_parameter("target_frame", "")
        self.declare_parameter("exclude_robot_footprint", True)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("max_cloud_age_s", 0.5)
        self.declare_parameter("point_stride", 2)
        self.declare_parameter("reliable_cloud", False)
        self.declare_parameter("height_filter_enabled", False)
        self.declare_parameter("height_filter_frame", "map")
        self.declare_parameter("estimate_ground_plane", False)
        self.declare_parameter("local_ground_enabled", False)
        self.declare_parameter("ground_seed_z", -0.35)
        self.declare_parameter("min_height_above_ground", 0.08)
        self.declare_parameter("max_height_above_ground", 2.0)
        self.height_enabled = bool(self.get_parameter("height_filter_enabled").value)
        self.height_frame = str(self.get_parameter("height_filter_frame").value)
        self.estimate_ground = bool(self.get_parameter("estimate_ground_plane").value)
        self.local_ground_enabled = bool(self.get_parameter("local_ground_enabled").value)
        self.ground_surface = None
        self.ground_seed_z = float(self.get_parameter("ground_seed_z").value)
        self.min_height = float(self.get_parameter("min_height_above_ground").value)
        self.max_height = float(self.get_parameter("max_height_above_ground").value)
        if not 0.0 < self.min_height < self.max_height:
            raise ValueError('ground-relative obstacle height limits must satisfy 0 < min < max')
        self.ground_plane = None
        self.ground_signature = None
        plane_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plane_pub = self.create_publisher(Vector3Stamped, '/navigation/ground_plane', plane_qos) if self.estimate_ground else None
        self.plane_sub = self.create_subscription(Vector3Stamped, '/navigation/ground_plane', self.on_ground_plane, plane_qos) if self.height_enabled and not self.estimate_ground else None
        self.surface_pub = self.create_publisher(PointCloud2, '/navigation/ground_surface', plane_qos) if self.estimate_ground and self.local_ground_enabled else None
        self.surface_sub = self.create_subscription(PointCloud2, '/navigation/ground_surface', self.on_ground_surface, plane_qos) if self.height_enabled and self.local_ground_enabled and not self.estimate_ground else None
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.exclude_footprint = bool(self.get_parameter("exclude_robot_footprint").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.max_cloud_age_s = max(0.1, float(self.get_parameter("max_cloud_age_s").value))
        self.point_stride = max(1, int(self.get_parameter("point_stride").value))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latest_only_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=(ReliabilityPolicy.RELIABLE if self.get_parameter("reliable_cloud").value
                         else ReliabilityPolicy.BEST_EFFORT),
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(
            PointCloud2, self.output_topic, latest_only_qos
        )
        self.subscription = self.create_subscription(
            PointCloud2, self.input_topic, self.on_cloud, latest_only_qos
        )
        self.forwarded = 0
        self.dropped = 0
        self.pending_cloud = None
        # TF and points arrive on independent subscriptions. Retry the newest
        # scan without blocking the executor that must receive its transform.
        self.retry_timer = self.create_timer(0.02, self.forward_pending_cloud) if self.height_enabled and not self.estimate_ground else None
        self.get_logger().info(
            f"costmap cloud {self.input_topic} -> {self.output_topic} in {self.target_frame}; "
            f"preserving measurement stamp, point_stride={self.point_stride}, "
            f"dropping age > {self.max_cloud_age_s:.1f}s"
        )

    def on_cloud(self, cloud: PointCloud2):
        self.pending_cloud = cloud
        self.forward_pending_cloud()

    def forward_pending_cloud(self):
        cloud = self.pending_cloud
        if cloud is None:
            return
        stamp_ns = int(cloud.header.stamp.sec) * 1_000_000_000 + int(cloud.header.stamp.nanosec)
        now = self.get_clock().now()
        age_s = (now.nanoseconds - stamp_ns) * 1e-9
        if stamp_ns <= 0 or not cloud.header.frame_id or age_s > self.max_cloud_age_s or age_s < -0.05:
            self.pending_cloud = None
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 50 == 0:
                self.get_logger().warning(
                    f"dropping stale/future cloud age={age_s:.3f}s (count={self.dropped})"
                )
            return

        try:
            from geometry_msgs.msg import TransformStamped
            identity = TransformStamped()
            identity.transform.rotation.w = 1.0
            costmap_cloud = self.transform_xyz(cloud, identity, self.point_stride)
            # Keep sensor geometry for raytracing; only transform when requested.
            output_frame = self.target_frame or cloud.header.frame_id
            if self.exclude_footprint:
                base_tf = self.tf_buffer.lookup_transform(
                    self.base_frame, cloud.header.frame_id, rclpy.time.Time())
                base_points = self.transform_xyz(costmap_cloud, base_tf, 1)
                xyz = self.xyz_view(base_points)
                keep = np.isfinite(xyz['x']) & np.isfinite(xyz['y']) & np.isfinite(xyz['z'])
                keep &= ~((np.abs(xyz['x']) <= 0.31) & (np.abs(xyz['y']) <= 0.235))
                costmap_cloud = self.select_points(costmap_cloud, keep)
            if self.height_enabled:
                # Apply a single ground-relative band BEFORE either costmap
                # transforms the data to its own frame. Absolute z thresholds
                # in map and odom disagree when map->odom has tilt/translation.
                reference_cloud = costmap_cloud
                if cloud.header.frame_id != self.height_frame:
                    height_tf = self.tf_buffer.lookup_transform(
                        self.height_frame, cloud.header.frame_id,
                        rclpy.time.Time.from_msg(cloud.header.stamp))
                    reference_cloud = self.transform_xyz(costmap_cloud, height_tf, 1)
                xyz = self.xyz_view(reference_cloud)
                if self.estimate_ground:
                    # Structured XYZ clouds may have uninitialized padding.
                    # Only geometry determines whether the floor needs refitting.
                    signature = zlib.crc32(np.column_stack([xyz[k] for k in ('x','y','z')]).tobytes())
                    if self.ground_plane is None or signature != self.ground_signature:
                        self.ground_plane = None
                        self.ground_surface = None
                        plane_msg = Vector3Stamped()
                        plane_msg.header = copy.deepcopy(reference_cloud.header)
                        plane_msg.header.frame_id = self.height_frame
                        try:
                            self.ground_plane = self.fit_ground_plane(xyz, self.ground_seed_z)
                        except ValueError:
                            plane_msg.vector.x = plane_msg.vector.y = plane_msg.vector.z = float('nan')
                            self.plane_pub.publish(plane_msg)
                            raise
                        plane_msg.vector.x, plane_msg.vector.y, plane_msg.vector.z = self.ground_plane
                        self.plane_pub.publish(plane_msg)
                        if self.local_ground_enabled:
                            surface = self.fit_ground_surface(xyz, self.ground_plane)
                            surface_msg = PointCloud2()
                            surface_msg.header = copy.deepcopy(plane_msg.header)
                            surface_msg.height, surface_msg.width = 1, len(surface)
                            surface_msg.fields = [PointField(name=k, offset=4*i, datatype=PointField.FLOAT32, count=1) for i,k in enumerate(('x','y','z'))]
                            surface_msg.point_step, surface_msg.row_step = 12, 12*len(surface)
                            surface_msg.is_dense = True
                            surface_msg.data = array('B', surface.astype('<f4').tobytes())
                            self.on_ground_surface(surface_msg)
                            self.surface_pub.publish(surface_msg)
                        self.ground_signature = signature
                        self.get_logger().info('Ground plane z = %.5f*x + %.5f*y + %.5f; obstacle band %.2f..%.2f m above ground' % (*self.ground_plane, self.min_height, self.max_height))
                if self.ground_plane is None:
                    raise ValueError('waiting for a valid saved-map ground plane')
                a, b, c = self.ground_plane
                ground = a*xyz['x'] + b*xyz['y'] + c
                height = xyz['z'] - self.local_ground_height(xyz, ground, getattr(self, 'ground_surface', None))
                costmap_cloud = self.select_points(costmap_cloud,
                    (height >= self.min_height) & (height <= self.max_height))
            if output_frame != cloud.header.frame_id:
                transform = self.tf_buffer.lookup_transform(
                    output_frame, cloud.header.frame_id, rclpy.time.Time.from_msg(cloud.header.stamp))
                costmap_cloud = self.transform_xyz(costmap_cloud, transform, 1)
        except (LookupException, ConnectivityException, ExtrapolationException):
            # Keep its real stamp and finite maximum age; newer scans replace
            # this one, so neither TF delay nor loss creates an unbounded queue.
            return
        except Exception as exc:
            self.pending_cloud = None
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 50 == 0:
                self.get_logger().warning(
                    f"drop navigation cloud ({cloud.header.frame_id}): {exc}"
                )
            return

        costmap_cloud.header.frame_id = output_frame
        self.pending_cloud = None
        self.publisher.publish(costmap_cloud)
        self.forwarded += 1

    def on_ground_plane(self, msg):
        plane = (msg.vector.x, msg.vector.y, msg.vector.z)
        # Transient-local topics can arrive in either order on startup.
        if self.ground_plane is not None and plane != self.ground_plane:
            self.ground_surface = None
        self.ground_plane = plane if msg.header.frame_id == self.height_frame and np.all(np.isfinite(plane)) else None

    def on_ground_surface(self, msg):
        self.ground_surface = None
        if msg.header.frame_id != self.height_frame or not msg.width:
            return
        xyz = self.xyz_view(msg)
        if not all(np.all(np.isfinite(xyz[k])) for k in ('x','y','z')):
            return
        indices = np.floor(np.column_stack((xyz['x'], xyz['y']))/.5).astype(int)
        origin = indices.min(axis=0)
        indices -= origin
        shape = indices.max(axis=0)+1
        if np.prod(shape) > 1000000:
            return
        heights = np.full(shape, np.nan, dtype=np.float32)
        heights[indices[:,0], indices[:,1]] = xyz['z']
        self.ground_surface = (origin, heights)

    @staticmethod
    def local_ground_height(xyz, baseline, surface):
        """Use the same saved-map floor in both filters; unknown cells keep the plane."""
        if surface is None:
            return baseline
        origin, heights = surface
        indices = np.floor(np.column_stack((xyz['x'], xyz['y']))/.5).astype(int)-origin
        valid = np.all(indices >= 0, axis=1) & np.all(indices < heights.shape, axis=1)
        rows = np.flatnonzero(valid)
        floor = heights[indices[valid,0], indices[valid,1]]
        finite = np.isfinite(floor)
        result = baseline.copy()
        # Never introduce new low obstacles by lowering the old floor model.
        result[rows[finite]] = np.maximum(result[rows[finite]], floor[finite])
        return result

    @staticmethod
    def fit_ground_surface(xyz, plane):
        """Follow supported, gently varying floor cells connected to the fitted plane.

        A raised isolated patch cannot seed the floor model. Steps >3.5 cm per
        0.5 m cell break connectivity, preserving low boxes and obstacle tops.
        Sparse or unsupported regions retain the conservative global plane.
        """
        from scipy.ndimage import convolve
        a, b, c = plane
        height = xyz['z']-(a*xyz['x']+b*xyz['y']+c)
        candidate = (height > -.12) & (height < .20)
        if np.count_nonzero(candidate) < 100:
            return np.empty((0,3), dtype=np.float32)
        residual = height[candidate]
        cells = np.floor(np.column_stack((xyz['x'][candidate], xyz['y'][candidate]))/.5).astype(int)
        origin = cells.min(axis=0)
        cells -= origin
        shape = cells.max(axis=0)+1
        if np.prod(shape) > 1000000:
            return np.empty((0,3), dtype=np.float32)
        ids = cells[:,0]*shape[1]+cells[:,1]
        order = np.argsort(ids)
        unique, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
        grid = np.full(shape, np.nan)
        for index, start, count in zip(unique, starts, counts):
            if count >= 4:
                grid.flat[index] = np.quantile(residual[order[start:start+count]], .25)
        connected = np.zeros(shape, dtype=bool)
        queue = deque(map(tuple, np.argwhere(np.isfinite(grid) & (np.abs(grid) < .025))))
        for cell in queue:
            connected[cell] = True
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                xx, yy = x+dx, y+dy
                if (0 <= xx < shape[0] and 0 <= yy < shape[1] and not connected[xx,yy]
                        and np.isfinite(grid[xx,yy]) and abs(grid[xx,yy]-grid[x,y]) <= .035):
                    connected[xx,yy] = True
                    queue.append((xx,yy))
        weights = convolve(connected.astype(float), np.ones((3,3)), mode='constant')
        total = convolve(np.where(connected, grid, 0.), np.ones((3,3)), mode='constant')
        supported = connected & (weights >= 5)
        indices = np.argwhere(supported)
        xy = (indices+origin+.5)*.5
        z = a*xy[:,0]+b*xy[:,1]+c+total[supported]/weights[supported]
        return np.column_stack((xy,z)).astype(np.float32)

    @staticmethod
    def fit_ground_plane(xyz, seed_z):
        """Robust plane for the current indoor floor, never a ceiling/table.

        The saved map's floor seed constrains candidate height. Require broad
        horizontal support, a gentle slope and a majority of floor inliers;
        do not silently invent a plane when the map lacks floor observations.
        """
        candidate = (np.abs(xyz['z']-seed_z) < 0.25)
        p = np.column_stack([xyz[k][candidate] for k in ('x','y','z')]).astype(np.float64)
        if len(p) < 100:
            raise ValueError('saved map has insufficient floor points near the configured ground seed')
        design = np.column_stack((p[:,:2], np.ones(len(p))))
        keep = np.abs(p[:,2]-np.median(p[:,2])) < 0.12
        for _ in range(6):
            if np.count_nonzero(keep) < 100:
                raise ValueError('insufficient ground-plane inliers')
            coeff = np.linalg.lstsq(design[keep], p[keep,2], rcond=None)[0]
            residual = p[:,2] - (coeff[0]*p[:,0]+coeff[1]*p[:,1]+coeff[2])
            keep = np.abs(residual) < 0.05
        spread = np.linalg.eigvalsh(np.cov(p[keep,:2], rowvar=False))
        if np.mean(keep) < 0.55 or spread[0] < 0.05 or np.hypot(*coeff[:2]) > 0.15:
            raise ValueError('saved map floor is unsupported, nonplanar or too steep for this height filter')
        return tuple(float(v) for v in coeff)

    @staticmethod
    def xyz_view(cloud):
        offsets = {}
        for field in cloud.fields:
            if field.name in ("x", "y", "z"):
                if field.datatype != PointField.FLOAT32 or field.count != 1:
                    raise ValueError(f"unsupported {field.name} field layout")
                offsets[field.name] = field.offset
        if set(offsets) != {"x", "y", "z"}:
            raise ValueError("cloud needs float32 x/y/z fields")
        if cloud.point_step <= 0 or cloud.row_step < cloud.width * cloud.point_step:
            raise ValueError("invalid PointCloud2 strides")

        endian = ">" if cloud.is_bigendian else "<"
        types = {1:'i1',2:'u1',3:'i2',4:'u2',5:'i4',6:'u4',7:'f4',8:'f8'}
        names, formats, all_offsets = [], [], []
        for field in cloud.fields:
            if field.datatype not in types or field.count < 1:
                raise ValueError('unsupported point field')
            base = np.dtype(endian+types[field.datatype])
            if field.offset < 0 or field.offset+base.itemsize*field.count > cloud.point_step:
                raise ValueError('point field exceeds point stride')
            names.append(field.name)
            formats.append(base if field.count == 1 else (base, (field.count,)))
            all_offsets.append(field.offset)
        dtype = np.dtype({'names':names,'formats':formats,'offsets':all_offsets,
                          'itemsize':cloud.point_step})
        source = np.ndarray(
            shape=(cloud.height, cloud.width), dtype=dtype, buffer=cloud.data,
            strides=(cloud.row_step, cloud.point_step)
        )
        return source.reshape(-1)

    @staticmethod
    def select_points(cloud, keep):
        points = NavCloudRetimestamp.xyz_view(cloud)[keep].copy()
        result = copy.deepcopy(cloud)
        result.height, result.width = 1, int(points.size)
        result.row_step = result.width*result.point_step
        # rclpy accepts uint8 arrays directly. Passing bytes validates every
        # byte in Python and used to add ~200 ms to a normal live scan.
        result.data = array('B', points.tobytes())
        result.is_dense = True
        return result

    @staticmethod
    def transform_xyz(cloud: PointCloud2, transform, point_stride: int) -> PointCloud2:
        """Thin and transform XYZ, preserving all other fields and source header."""
        points = NavCloudRetimestamp.xyz_view(cloud)[::point_stride].copy()
        points = points[np.isfinite(points['x']) & np.isfinite(points['y']) & np.isfinite(points['z'])]
        q = transform.transform.rotation
        scale = 2.0 / max(1e-18, q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        xx, yy, zz = q.x*q.x*scale, q.y*q.y*scale, q.z*q.z*scale
        xy, xz, yz = q.x*q.y*scale, q.x*q.z*scale, q.y*q.z*scale
        wx, wy, wz = q.w*q.x*scale, q.w*q.y*scale, q.w*q.z*scale
        r00, r01, r02 = 1.0-(yy+zz), xy-wz, xz+wy
        r10, r11, r12 = xy+wz, 1.0-(xx+zz), yz-wx
        r20, r21, r22 = xz-wy, yz+wx, 1.0-(xx+yy)
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z
        x = points["x"].copy()
        y = points["y"].copy()
        z = points["z"].copy()
        points["x"] = r00*x + r01*y + r02*z + tx
        points["y"] = r10*x + r11*y + r12*z + ty
        points["z"] = r20*x + r21*y + r22*z + tz
        transformed = PointCloud2()
        transformed.header = copy.deepcopy(cloud.header)
        transformed.height = 1
        transformed.width = int(points.size)
        transformed.fields = cloud.fields
        transformed.is_bigendian = cloud.is_bigendian
        transformed.point_step = cloud.point_step
        transformed.row_step = transformed.width * transformed.point_step
        transformed.is_dense = cloud.is_dense
        transformed.data = array('B', points.tobytes())
        return transformed


def main(args=None):
    rclpy.init(args=args)
    node = NavCloudRetimestamp()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
