from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection,PolyCollection
import numpy as np
from typing import List
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import colormaps, _api
from matplotlib.animation import FuncAnimation
from functools import partial
import math
import cartopy.crs as ccrs


class Node:
    """
    This is a base class Node for delft3d_net.
    """
    def __init__(self, loc: List[float], **kwargs):
        self.__loc = loc
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return

    def get_loc(self):
        return self.__loc

    def set_loc(self, loc: list) -> None:
        self.__loc = loc
        return

    def add_attr(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return

    def __str__(self):
        return f"Node locates at {self.__loc}."


class Edge:
    """
    This is a base class Edge for delft3d_net
    """
    def __init__(self, edge_loc: List[float], edge_nodes: List[Node], **kwargs):
        """

        Parameters:
        ---------- 
        edge_loc: Characteristic positions of edges, such as midpoints.
        edge_nodes: Two node that makes up the edge.
        **kwargs: Set attributes for the edge, eg: id=1, color='k'
        Returns:
        ----------
        """
        self.__edge_loc = edge_loc
        self.__egde_nodes = edge_nodes
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return

    def get_edge_loc(self):
        return self.__edge_loc

    def get_edge_nodes(self):
        return self.__edge_nodes

    def set_edge_loc(self, edge_loc: List[float]) -> None:
        self.__edge_loc = edge_loc
        return

    def set_edge_nodes(self, edge_nodes: List[float]) -> None:
        self.__edge_nodes = edge_nodes
        return

    def add_attr(self, **kwargs):
        for key, value in kwargs:
            self.__setattr__(key, value)
        return


class Face:
    """
    This is a base class Face for delft3d_net.
    """
    def __init__(self, face_loc: List[float], face_nodes_id: List[int], **kwargs):
        self.__face_loc = face_loc
        self.__face_nodes_id = face_nodes_id
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return

    def get_face_loc(self):
        return self.__face_loc

    def get_face_nodes_id(self):
        return self.__face_nodes_id

    def set_face_loc(self, face_loc):
        self.__face_loc = face_loc
        return

    def set_face_nodes_id(self, face_nodes_id):
        self.__face_nodes_id = face_nodes_id
        return

    def add_attr(self, **kwargs):
        for key, value in kwargs:
            self.__setattr__(key, value)
            return

    def __str__(self):
        return f"Face locates at {self.__face_loc} with nodes {self.__face_nodes_id}"


class Collection:
    """
    Base Collection.
    """
    def __init__(self):
        return

    def get_item(self, id: int):
        return self.items[id]


class NodeCollection(Collection):
    """
    A node collection.
    """
    def __init__(self, nodes_loc:List, nodes_num):
        self.nodes_loc = np.column_stack(nodes_loc)
        self.nodes_num = nodes_num
        self.nodes = np.apply_along_axis(Node, axis=1, arr=self.nodes_loc)
        return

    def get_nodes(self):
        return self.nodes


class EdgeCollection(Collection):
    def __init__(self):
        pass


class FaceCollection(Collection):
    def __init__(self, nodes, faces_loc: List[float], faces_nodes_id: List[int], faces_num: int):
        """
        Iinital function

        Parameters
        ----------

        nodes: NodeCollection
        faces_loc: A list of faces location, format as [[x0,x1,x2], [y0, y1, y2]]
        faces_nodes: A list of nodes comprised of faces, format as [[node_id1, node_id2, node_id3], [node_id3], [node_id4], [node_id5]]
        faces_num: number of nodes.
        """
        self.nodes = nodes
        self.faces_loc = np.column_stack(faces_loc)
        _, self.faceloc_dim = self.faces_loc.shape
        self.faces_nodes_id = faces_nodes_id
        self.faces_num = faces_num
        ids = np.arange(0, self.faces_num, 1)
        params = np.column_stack([self.faces_loc, self.faces_nodes_id, ids])
        self.faces = np.apply_along_axis(self.create_face, axis=1, arr=params)

    def create_face(self, params):
        return Face(face_loc=params[0:self.faceloc_dim], face_nodes_id=params[self.faceloc_dim+1:-1], id=params[-1])

    def get_node_xy(self, id):
        return self.nodes.nodes_loc[id-1, 0:2]

    def createPatchCollection(self, **kwargs) -> PatchCollection:
        polygons = []
        for i in range(self.faces_nodes_id.shape[0]):
            nodes_id = self.faces_nodes_id[i][~self.faces_nodes_id[i].mask]
            pol_xy = self.get_node_xy(nodes_id)
            polygons.append(Polygon(pol_xy))
        patchcollection = PatchCollection(polygons,  **kwargs)
        return patchcollection


class ncNetParser:
    """
    This is a net file parser for delft3d.
    """
    def __init__(self, filepath):
        self._filepath = filepath
        self._parse()
        return

    def _parse(self):
        with Dataset(self._filepath, 'r') as dataset:
            self.mesh2d_nNodes = dataset.dimensions['mesh2d_nNodes'].size
            self.mesh2d_nEdges = dataset.dimensions['mesh2d_nEdges'].size
            self.mesh2d_nFaces = dataset.dimensions['mesh2d_nFaces'].size
            self.mesh2d_node_x = dataset.variables['mesh2d_node_x'][:]
            self.mesh2d_node_y = dataset.variables['mesh2d_node_y'][:]
            self.mesh2d_node_z = dataset.variables['mesh2d_node_z'][:]
            self.mesh2d_edge_x = dataset.variables['mesh2d_edge_x'][:]
            self.mesh2d_edge_y = dataset.variables['mesh2d_edge_y'][:]
            self.mesh2d_edge_nodes = dataset.variables['mesh2d_edge_nodes'][:]
            self.mesh2d_face_x = dataset.variables['mesh2d_face_x'][:]
            self.mesh2d_face_y = dataset.variables['mesh2d_face_y'][:]
            self.mesh2d_face_nodes = dataset.variables['mesh2d_face_nodes'][:]
        return

    def createNodes(self) -> NodeCollection:
        nodes = NodeCollection(nodes_loc=[self.mesh2d_node_x, self.mesh2d_node_y, self.mesh2d_node_z],
                               nodes_num=self.mesh2d_nNodes)
        return nodes

    def createFaces(self) -> FaceCollection:
        nodes = self.createNodes()
        faces = FaceCollection(nodes, faces_loc=[self.mesh2d_face_x, self.mesh2d_face_y],
                               faces_nodes_id=self.mesh2d_face_nodes, faces_num=self.mesh2d_nFaces)
        return faces


class ncResultViewer(ncNetParser):
    """
    This is a parser for visualzie the result compute by delft3d-fm
    """
    def __init__(self, filepath, result=False, src_crs=None, des_crs=None):
        self._filepath = filepath
        self._parse(result, src_crs, des_crs)
        # super().__init__(filepath)

    def _parse(self, result=False, src_crs=None, des_crs=None):
        with Dataset(self._filepath, 'r') as dataset:
            # net parameters
            self.mesh2d_nNodes = dataset.dimensions['mesh2d_nNodes'].size
            self.mesh2d_nEdges = dataset.dimensions['mesh2d_nEdges'].size
            self.mesh2d_nFaces = dataset.dimensions['mesh2d_nFaces'].size
            self.mesh2d_node_x = dataset.variables['mesh2d_node_x'][:]
            self.mesh2d_node_y = dataset.variables['mesh2d_node_y'][:]
            self.mesh2d_node_z = dataset.variables['mesh2d_node_z'][:]
            self.mesh2d_edge_x = dataset.variables['mesh2d_edge_x'][:]
            self.mesh2d_edge_y = dataset.variables['mesh2d_edge_y'][:]
            self.mesh2d_edge_nodes = dataset.variables['mesh2d_edge_nodes'][:]
            self.mesh2d_face_x = dataset.variables['mesh2d_face_x'][:]
            self.mesh2d_face_y = dataset.variables['mesh2d_face_y'][:]
            self.mesh2d_face_nodes = dataset.variables['mesh2d_face_nodes'][:]
            if result:
            # results parameters
                self.mesh2d_s1 = dataset.variables['mesh2d_s1'][:]  # Water level
                self.mesh2d_ucx = dataset.variables['mesh2d_ucx'][:]
                self.mesh2d_ucy = dataset.variables['mesh2d_ucy'][:]
                self.mesh2d_ucmag = dataset.variables['mesh2d_ucmag'][:]
                self.mesh2d_waterdepth = dataset.variables['mesh2d_waterdepth'][:]  # Water depth at pressure points
            if src_crs and des_crs:
                node_loc = des_crs.transform_points(src_crs, self.mesh2d_node_x, self.mesh2d_node_y)
                self.mesh2d_node_x, self.mesh2d_node_y = node_loc[:, 0], node_loc[:, 1]
                edge_loc = des_crs.transform_points(src_crs, self.mesh2d_edge_x, self.mesh2d_edge_y)
                self.mesh2d_edge_x, self.mesh2d_edge_y = edge_loc[:, 0], edge_loc[:, 1]
                face_loc = des_crs.transform_points(src_crs, self.mesh2d_face_x, self.mesh2d_face_y)
                self.mesh2d_face_x, self.mesh2d_face_y = face_loc[:, 0], face_loc[:, 1]

        return

    def get_bed_level(self):
        bed_level = []
        for i in range(self.mesh2d_face_nodes.shape[0]):
            nodes_id = self.mesh2d_face_nodes[i][~self.mesh2d_face_nodes[i].mask]
            z_mean = self.mesh2d_node_z[nodes_id - 1].mean()
            bed_level.append(z_mean)
        self.bed_level = bed_level
        return

    def plot(self, ax, variable, time_step=0, cmap='jet', delta_t=1, **kwargs):
        """
        This is a function to visualize the variable in the net.

        Parameters
        -----
        ax: Axes
        variable: water level, ucx, ucy, ucmag, water depth, bed level
        time_step: 0~max_time_step, use 0 as default, if time_step==-1, then all the time_step will be shown.
        cmap: colorsmap to use, use 'jet' as default
        """
        if variable == 'water level':
            data = self.mesh2d_s1
        elif variable == 'ucx':
            data = self.mesh2d_ucx
        elif variable == 'ucy':
            data = self.mesh2d_ucy
        elif variable == 'ucmag':
            data = self.mesh2d_ucmag
        elif variable == 'water depth':
            data = self.mesh2d_waterdepth
        elif variable == 'bed level':
            self.get_bed_level()
            data = self.bed_level
            #norm = Normalize(np.abs(data).min(), np.abs(data).max())
            norm = Normalize(100, 300)
            cmap = colormaps[cmap]
            sm = ScalarMappable(norm=norm, cmap=cmap)

            colors = sm.to_rgba(data)
            faces = self.createFaces()
            patch_collection = faces.createPatchCollection(linewidth=0, edgecolors='none', facecolors=colors, **kwargs)
            ax.add_collection(patch_collection)
            ax.autoscale_view()
            return patch_collection, sm
        else:
            raise NotImplementedError("Interface to plot the {} hasn't implement yet!".format(variable))
        
        norm = Normalize(np.abs(data).min(), np.abs(data).max())
        cmap = colormaps[cmap]
        sm = ScalarMappable(norm=norm, cmap=cmap)
        
        if time_step == -1:
            colors = sm.to_rgba(np.abs(data[0, :]))
            faces = self.createFaces()
            patch_collection = faces.createPatchCollection(linewidth=0, edgecolors='k', facecolors=colors, **kwargs)
            ax.add_collection(patch_collection)
            text = ax.text(0.5, 1, 'Time: {:.2f}s'.format(0 * delta_t), transform=ax.transAxes,
                           ha='center', va='bottom')
            ax.autoscale_view()
            fig = ax.figure
            def update(frames, patch_collection, text):
                colors = sm.to_rgba(data[frames, :])
                patch_collection.set_facecolor(colors)
                text.set_text(f'Time: {frames*delta_t:.2f}s')
                return
            obj = FuncAnimation(fig, partial(update, patch_collection=patch_collection, text=text), frames=data.shape[0])
        else:
            norm = Normalize(np.abs(data[time_step, :]).min(), np.abs(data[time_step]).max())
            sm = ScalarMappable(norm=norm, cmap=cmap)
            colors = sm.to_rgba(np.abs(data[time_step, :]))
            faces = self.createFaces()
            patch_collection = faces.createPatchCollection(linewidth=0, edgecolors=colors, facecolors=colors, **kwargs)
            ax.add_collection(patch_collection)
            text = ax.text(0.5, 1, 'Time: {:.2f}s'.format(time_step * delta_t), transform=ax.transAxes,
                           ha='center', va='bottom')
            ax.autoscale_view()
            obj = patch_collection
        return obj, sm

    def quiver(self, ax, time_step=0, width=None, scale=None, delta_t=1, cmap='jet', **kwargs):
        ucmag = np.hypot(self.mesh2d_ucx, self.mesh2d_ucy)
        norm = Normalize(ucmag.min(), ucmag.max())
        cmap = colormaps[cmap]
        sm = ScalarMappable(norm=norm, cmap=cmap)
        if time_step == -1:
            ucmag = np.hypot(self.mesh2d_ucx[0, :], self.mesh2d_ucy[0, :])
            colors = sm.to_rgba(ucmag)
            x = self.mesh2d_face_x
            y = self.mesh2d_face_y
            ucx = self.mesh2d_ucx[0, :]
            ucy = self.mesh2d_ucy[0, :]
            data = [x, y, ucx, ucy]
            quiver = Quiver(ax, data, width=width, scale=scale, color=colors, **kwargs)
            ax.add_collection(quiver)
            text = ax.text(0.5, 1, 'Time: {:.2f}s'.format(0 * delta_t), transform=ax.transAxes,
                           ha='center', va='bottom')
            ax.autoscale()
            fig = ax.figure

            def update(frames, quiver, text):
                ucmag = np.hypot(self.mesh2d_ucx[frames, :], self.mesh2d_ucy[frames, :])
                colors = sm.to_rgba(ucmag)
                ucx = self.mesh2d_ucx[frames, :]
                ucy = self.mesh2d_ucy[frames, :]
                quiver.update_uvc(ucx, ucy, colors)
                text.set_text('Time: {:.2f}s'.format(frames * delta_t))
                return
            obj = FuncAnimation(fig, partial(update, quiver=quiver, text=text), frames=self.mesh2d_ucmag.shape[0])
            return obj, sm
        else:
            norm = Normalize(ucmag[time_step, :].min(), ucmag[time_step, :].max())
            sm = ScalarMappable(norm=norm, cmap=cmap)
            colors = sm.to_rgba(self.mesh2d_ucmag[time_step, :])
            x = self.mesh2d_face_x
            y = self.mesh2d_face_y
            ucx = self.mesh2d_ucx[time_step, :]
            ucy = self.mesh2d_ucy[time_step, :]
            data = [x, y, ucx, ucy]
            quiver = Quiver(ax, data, width=width, scale=scale, color=colors, **kwargs)
            ax.add_collection(quiver)
            text = ax.text(0.5, 1, 'Time: {:.2f}s'.format(time_step * delta_t), transform=ax.transAxes,
                           ha='center', va='bottom')
            ax.autoscale()
            return quiver, sm

class Quiver(PolyCollection):
    """
    Specialized PolyCollection for arrows.
    """
    _PIVOT_VALS = ('tail', 'middle', 'tip')

    def __init__(self, ax, data, 
                 scale=None, headwidth=3, headlength=5, headaxislength=4.5,
                 minshaft=1, minlength=1, units='width', width=None,
                 color='k', pivot='tail', **kwargs):
        """
        The constructor takes one required argument, an Axes 
        instance, followed by the args and kwargs described:
        """
        x, y, u, v = data[0], data[1], data[2], data[3]
        self.x = x
        self.y = y
        self.xy = np.column_stack((x, y))
        self.N = len(x)
        self._axes = ax
        self.scale = scale
        self.headwidth = headwidth
        self.headlength = float(headlength)
        self.headaxislength = headaxislength
        self.minshaft = minshaft
        self.minlength = minlength
        self.units = units

        self._auto_fit_width(width)
        self.pivot = pivot
        _api.check_in_list(self._PIVOT_VALS, pivot=self.pivot)

        self.transform = kwargs.pop('transform', ax.transData)
        kwargs.setdefault('facecolors', color)
        kwargs.setdefault('linewidths', (0,))
        super().__init__([], closed=False, **kwargs)
        self.update_uvc(u, v, color)

    def _auto_fit_width(self, width):
        if width is None:
            yspan = self.y.max() - self.y.min()
            sn = np.clip(math.sqrt(self.N), 8, 25)
            self.width = 0.06 * yspan / sn
            print("Autofit width is {:.4f}.".format(self.width))
        else:
            self.width = width

    def _get_angles_lengths(self, u, v):
        angles = np.arctan2(v, u)
        if self.scale == None:
            lengths = np.hypot(u, v)
            minsh = self.headlength + self.minshaft
            lenmean = lengths.mean()
            if lenmean == 0:
                scale = 1
                lengths = lengths * scale
            else:
                self.scale = minsh/lengths.mean()
                print("Autofit scale is {:.4f}.".format(self.scale))
                lengths = self.scale * lengths
        else:
            lengths = np.hypot(u, v) * self.scale # u,v值太小时要实现一定的放缩，否则全部都是六边形
        return angles, lengths

    def _make_verts(self, u, v):
        angles, lengths = self._get_angles_lengths(u, v)
        x, y = self._h_arrows(lengths)
        theta = angles.reshape((-1, 1))
        xy = (x + y * 1j) * np.exp(1j * theta) * self.width
        xy = np.stack((xy.real, xy.imag), axis=2)
        return xy

    def _h_arrows(self, length):
        """
        Length is in arrow width units.
        Draw a horizontal arrow
        """
        minsh = self.minshaft * self.headlength
        N = len(length)
        length = length.reshape(N, 1)
        x = np.array([0, -self.headaxislength,
                      -self.headlength, 0],
                      np.float64)
        x = x + np.array([0, 1, 1, 1]) * length
        y = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        y = np.repeat(y[np.newaxis, :], N, axis=0)
        # x0, y0: arrow without shaft, for short vectors
        x0 = np.array([0, minsh - self.headaxislength,
                       minsh - self.headlength, minsh], np.float64)
        y0 = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)

        # formalate a mirror
        ii = [0, 1, 2, 3, 2, 1, 0, 0]
        X = x[:, ii]
        Y = y[:, ii]
        Y[:, 3:-1] *= -1

        X0 = x0[ii]
        Y0 = y0[ii]
        Y0[3:-1] *= -1

        shrink = length / minsh if minsh != 0. else 0.
        X0 = shrink * X0[np.newaxis, :]
        Y0 = shrink * Y0[np.newaxis, :]
        short = np.repeat(length < minsh, 8, axis=1)
        # Now select X0, Y0 if short, otherwise X, Y
        np.copyto(X, X0, where=short)
        np.copyto(Y, Y0, where=short)
        if self.pivot == 'middle':
            X -= 0.5 * X[:, 3, np.newaxis]
        elif self.pivot == 'tip':
            # numpy bug? using -= does not work here unless we multiply by a
            # float first, as with 'mid'.
            X = X - X[:, 3, np.newaxis]
        elif self.pivot != 'tail':
            _api.check_in_list(["middle", "tip", "tail"], pivot=self.pivot)

        tooshort = length < self.minlength
        if tooshort.any():
            # Use a heptagonal dot:
            th = np.arange(0, 8, 1, np.float64) * (np.pi / 3.0)
            x1 = np.cos(th) * self.minlength * 0.5
            y1 = np.sin(th) * self.minlength * 0.5
            X1 = np.repeat(x1[np.newaxis, :], N, axis=0)
            Y1 = np.repeat(y1[np.newaxis, :], N, axis=0)
            tooshort = np.repeat(tooshort, 8, 1)
            np.copyto(X, X1, where=tooshort)
            np.copyto(Y, Y1, where=tooshort)
        return X, Y

    def update_uvc(self, u, v, colors):
        verts = self._make_verts(u, v) # 所有箭头均在原点
        verts = verts + np.repeat(self.xy[:, np.newaxis, :], 8, axis=1) # 平移
        self.set_verts(verts, closed=False)
        self.set_facecolors(colors)



        



if __name__ == '__main__':
    filepath = "./scripts/results_test_case/miyun_test_map.nc"
    # filepath = "G:\FlowFM_map.nc"
    dambreak = ncResultViewer(filepath)
    # print(test.mesh2d_s1.shape)
    fig = plt.figure(figsize=(24/2.54, 12/2.54))
    ax = fig.add_subplot(111)
    # m, sm = dambreak.plot(ax, 'ucmag', -1)
    # obj, sm = dambreak.quiver(ax, width=None, scale=2, time_step=50, delta_t=0.1)
    # ax1.vlines(2.4, 0, 3.95, linewidth=3, colors='k')
    # ax1.vlines(2.4, 4.35, 8.3, linewidth=3, colors='k')
    # ax1.set_xlim(0, 31)
    # ax1.set_ylim(0, 8.3)
    ax.set_xlabel('x(km)')
    ax.set_ylabel('y(km)')
    ax.set_aspect(1)

    # cbar = fig.colorbar(mappable=sm, ax=ax, orientation='horizontal', location='bottom', pad=0.12)
    # cbar.set_label('umag(m/s)')
    test = ncNetParser(filepath).createFaces().createPatchCollection(linewidths=0.1, edgecolors='#239AEA', facecolors='w')
    # fig = plt.figure(figsize=(12/2.54, 9/2.54))
    # ax = fig.add_axes([0.1,0.1, 0.85, 0.85])
    ax.add_collection(test)
    ax.autoscale()

    # from boundary import Boundary

    # line1 = Boundary("./data_pre/ldb/Boundary.ldb", linewidth=0.5, facecolor=None, color='k')
    # line2 = Boundary("./data_pre/ldb/Dams.ldb", linewidth=0.5, facecolor=None, color='k')
    # ax.add_collection(line1)
    # ax.add_collection(line2)
    # ax.autoscale_view()
    # ax.set_aspect(1)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # # fig.savefig('./netplot.pdf')
    plt.show()