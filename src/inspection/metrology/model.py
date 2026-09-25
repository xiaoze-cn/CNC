"""Load and triangulate STEP reference models."""

from pathlib import Path

import numpy as np
import open3d as o3d
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location


def load_mesh(path: Path, tolerance_mm: float) -> o3d.geometry.TriangleMesh:
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if str(status) not in {"IFSelect_ReturnStatus.IFSelect_RetDone", "IFSelect_RetDone"}:
        raise ValueError(f"STEP 读取失败: {path} ({status})")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(
        shape, max(0.005, tolerance_mm / 20.0), True, 0.2, True
    ).Perform()
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation:
            offset = len(vertices)
            transformation = location.Transformation()
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(index)
                point.Transform(transformation)
                vertices.append((point.X(), point.Y(), point.Z()))
            for index in range(1, triangulation.NbTriangles() + 1):
                first, second, third = triangulation.Triangle(index).Get()
                if reversed_face:
                    second, third = third, second
                triangles.append(
                    (offset + first - 1, offset + second - 1, offset + third - 1)
                )
        explorer.Next()
    if len(vertices) < 3 or not triangles:
        raise ValueError(f"STEP 三角化结果为空: {path}")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh
