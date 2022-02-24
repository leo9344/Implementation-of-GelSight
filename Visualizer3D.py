import open3d
from open3d import *
import numpy.matlib
import numpy as np

class Visualizer():
    def __init__(self, n, m):
        self.n, self.m = n,m
        self.points = np.zeros([self.n * self.m, 3])
        print(f'self.n: {self.n}, self.m: {self.m}')
        self.init_open3D()

    def init_open3D(self):
        x = np.arange(self.n)
        y = np.arange(self.m)
        self.X, self.Y = np.meshgrid(x, y)
        # Z = (X ** 2 + Y ** 2) / 10
        Z = np.sin(self.X)

        self.points[:, 0] = np.ndarray.flatten(self.X) / self.n
        self.points[:, 1] = np.ndarray.flatten(self.Y) / self.m

        print("z1",np.shape(Z))
        self.depth2points(Z)
        print(np.shape(self.points))
        # exit(0)

        # points = np.random.rand(1,3)

        # self.pcd = PointCloud()

        self.pcd = open3d.geometry.PointCloud()

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        # self.pcd.colors = Vector3dVector(np.zeros([self.n, self.m, 3]))

        self.vis = open3d.visualization.Visualizer()
        # self.vis.get_render_option()
        # opt.background_color = np.asarray([0,0,0])
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-10)
        print("fov", self.ctr.get_field_of_view())
        self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.35)
        self.ctr.rotate(0, -400)  # mouse drag in x-axis, y-axis
        self.vis.update_renderer()

    def depth2points(self, Z):
        print("inside",np.shape(Z))
        self.points[:, 2] = np.ndarray.flatten(Z)

    def update(self, Z):
        self.depth2points(Z)

        dx, dy = np.gradient(Z)
        dx, dy = dx * 100, dy * 100

        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([self.points.shape[0], 3])

        colors = np.zeros([self.points.shape[0], 3])

        for _ in range(3):
            colors[:, _] = np_colors
        # print("COLORS", colors)

        self.pcd.points = open3d.utility.Vector3dVector(self.points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        try:
            self.vis.update_geometry()
        except:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()