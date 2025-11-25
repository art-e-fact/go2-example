from pxr import Usd, UsdGeom, Gf
import rerun as rr
import numpy as np


class UsdRerunLogger:
    """Visualize USD stages in Rerun."""

    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        rr.init("isaac_rerun_logger", spawn=True)
        self._logged_meshes = set()  # Track which meshes we've already logged

    def log_stage(self, frame_idx: int = None):
        """
        Log the entire USD stage to Rerun.

        Args:
            frame_idx: Optional frame index for this log. If None, uses static data.
        """
        # Clear the set of logged meshes for this frame
        # (we want to log meshes once per log_stage call)
        logged_this_frame = set()

        # Set frame index if provided
        if frame_idx is not None:
            rr.set_time("frame_idx", sequence=frame_idx)

        # Traverse all prims in the stage
        # Use Usd.TraverseInstanceProxies to traverse into instanceable prims (references)
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        for prim in self.stage.Traverse(predicate):
            entity_path = str(prim.GetPath())

            # Log transforms for all Xformable prims
            if prim.IsA(UsdGeom.Xformable):
                self._log_transform(prim, entity_path)

            # Log mesh geometry (only once per unique mesh)
            if prim.IsA(UsdGeom.Mesh):
                mesh_path = str(prim.GetPath())
                if mesh_path not in self._logged_meshes:
                    self._log_mesh(prim, entity_path)
                    self._logged_meshes.add(mesh_path)
                    logged_this_frame.add(mesh_path)

    def _log_transform(self, prim: Usd.Prim, entity_path: str):
        """Log the transform of an Xformable prim."""
        xformable = UsdGeom.Xformable(prim)

        # Get the local transformation matrix
        local_xform: Gf.Matrix4d = xformable.GetLocalTransformation()
        translation: Gf.Vec3d = local_xform.ExtractTranslation()
        quaternion: Gf.Quatd = local_xform.ExtractRotationQuat()

        # Log the transform to Rerun
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=(translation[0], translation[1], translation[2]),
                quaternion=(
                    quaternion.GetImaginary()[0],
                    quaternion.GetImaginary()[1],
                    quaternion.GetImaginary()[2],
                    quaternion.GetReal(),
                ),
                # TODO: set scale
            ),
        )

    def _log_mesh(self, prim: Usd.Prim, entity_path: str):
        """Log mesh geometry to Rerun."""
        mesh = UsdGeom.Mesh(prim)

        # Get vertex positions
        points_attr = mesh.GetPointsAttr()
        if not points_attr:
            return

        points = points_attr.Get()
        if not points:
            return

        # Convert to numpy array
        vertices = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float32)

        # Get face vertex indices
        face_vertex_indices_attr = mesh.GetFaceVertexIndicesAttr()
        face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()

        if not face_vertex_indices_attr or not face_vertex_counts_attr:
            # If no faces, log as point cloud
            rr.log(entity_path, rr.Points3D(positions=vertices))
            return

        face_vertex_indices = face_vertex_indices_attr.Get()
        face_vertex_counts = face_vertex_counts_attr.Get()

        if not face_vertex_indices or not face_vertex_counts:
            rr.log(entity_path, rr.Points3D(positions=vertices))
            return

        # Convert face data to triangle indices
        # USD supports arbitrary polygons, but Rerun prefers triangles
        indices = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:
                # Already a triangle
                indices.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 1],
                        face_vertex_indices[idx + 2],
                    ]
                )
            elif count == 4:
                # Quad - split into two triangles
                indices.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 1],
                        face_vertex_indices[idx + 2],
                    ]
                )
                indices.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 2],
                        face_vertex_indices[idx + 3],
                    ]
                )
            else:
                # For polygons with more vertices, use simple fan triangulation
                for i in range(1, count - 1):
                    indices.extend(
                        [
                            face_vertex_indices[idx],
                            face_vertex_indices[idx + i],
                            face_vertex_indices[idx + i + 1],
                        ]
                    )
            idx += count

        # Convert to numpy array and reshape for Rerun
        indices_array = np.array(indices, dtype=np.uint32).reshape(-1, 3)

        # Get normals if available
        normals_attr = mesh.GetNormalsAttr()
        normals = None
        if normals_attr:
            normals_data = normals_attr.Get()
            if normals_data:
                normals = np.array(
                    [(n[0], n[1], n[2]) for n in normals_data], dtype=np.float32
                )

        print(
            f"Logging mesh {entity_path} with {len(vertices)} vertices and {len(indices_array)} triangles"
        )

        # Log the mesh to Rerun
        if normals is not None and len(normals) == len(vertices):
            rr.log(
                entity_path,
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=indices_array,
                    vertex_normals=normals,
                ),
            )
        else:
            rr.log(
                entity_path,
                rr.Mesh3D(vertex_positions=vertices, triangle_indices=indices_array),
            )

    def clear_logged_meshes(self):
        """Clear the cache of logged meshes, allowing them to be logged again."""
        self._logged_meshes.clear()
