from pxr import Usd, UsdGeom, Gf, UsdShade
import rerun as rr
import numpy as np
from PIL import Image
import os


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
        print(
            f"Mesh {entity_path} has {len(vertices)} vertices and {len(indices_array)} triangles"
        )

        # Get normals if available
        normals_attr = mesh.GetNormalsAttr()
        normals = None
        if normals_attr:
            normals_data = normals_attr.Get()
            if normals_data:
                normals = np.array(
                    [(n[0], n[1], n[2]) for n in normals_data], dtype=np.float32
                )

        # --- Material and Texture Handling ---
        texcoords = None
        texture_buffer = None
        # texture_format = None
        albedo_factor = None

        # Get UVs
        uvs = self._get_uvs(mesh)
        if uvs is not None:
            texcoords = uvs

        # Get Material Info
        color, texture_path = self._get_material_info(prim)
        print(
            f"Material info for {entity_path}: color={color}, texture_path={texture_path}"
        )
        if color:
            albedo_factor = color
        if texture_path:
            data, format = self._load_texture(texture_path)
            if data is not None:
                texture_buffer = data
                # texture_format = format

        print(
            f"Logging mesh {entity_path} with {len(vertices)} vertices and {len(indices_array)} triangles"
        )

        # Log the mesh to Rerun
        mesh_args = {
            "vertex_positions": vertices,
            "triangle_indices": indices_array,
        }
        if normals is not None and len(normals) == len(vertices):
            mesh_args["vertex_normals"] = normals
        if texcoords is not None and len(texcoords) == len(vertices):
            mesh_args["vertex_texcoords"] = texcoords

        if texture_buffer is not None:
            mesh_args["albedo_texture"] = texture_buffer

        if albedo_factor is not None:
            mesh_args["albedo_factor"] = albedo_factor

        rr.log(entity_path, rr.Mesh3D(**mesh_args))

    def clear_logged_meshes(self):
        """Clear the cache of logged meshes, allowing them to be logged again."""
        self._logged_meshes.clear()

    def _get_uvs(self, mesh: UsdGeom.Mesh):
        """Get UV coordinates from the mesh."""
        primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        for name in ["st", "uv", "texcoord", "texture_coordinates"]:
            primvar = primvars_api.GetPrimvar(name)
            if primvar and primvar.IsDefined():
                uvs = primvar.Get()
                if uvs:
                    print(f"Found UVs with primvar '{name}'")
                    print(f"Sample UVs: {uvs[:5]}")
                    print(f"UVs length: {len(uvs)}")
                    indices = primvar.GetIndices()
                    if indices:
                        uvs = [uvs[i] for i in indices]
                    return np.array([(u[0], u[1]) for u in uvs], dtype=np.float32)
        return None

    def _get_material_info(self, prim: Usd.Prim):
        """
        Get material color or texture path.
        Returns: (color_tuple, texture_path)
        """
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material: UsdShade.Material = binding_api.ComputeBoundMaterial()[0]
        if not material:
            return None, None

        shader = material.ComputeSurfaceSource()[0]
        if not shader:
            return None, None

        # List of inputs to check for color/texture
        input_names = [
            "diffuseColor",
            "albedo",
            "color",
            "base_color",
            "diffuse_texture",
            "albedo_texture",
            "base_color_texture",
        ]

        for name in input_names:
            input_attr = shader.GetInput(name)
            if not input_attr:
                continue

            # 1. Check for connections (Texture)
            if input_attr.HasConnectedSource():
                source, source_name, _ = input_attr.GetConnectedSource()
                if source:
                    source_prim = source.GetPrim()

                    # Case A: Connected to UsdUVTexture (common in UsdPreviewSurface)
                    if source_prim.GetTypeName() == "UsdUVTexture":
                        file_input = source.GetInput("file")
                        if file_input:
                            file_path = file_input.Get()
                            if file_path:
                                path = (
                                    file_path.path
                                    if hasattr(file_path, "path")
                                    else str(file_path)
                                )
                                return None, path

                    # Case B: Connected to Material input (common in OmniPBR / MDL)
                    elif source_prim.IsA(UsdShade.Material):
                        material_input = source.GetInput(source_name)
                        if material_input:
                            val = material_input.Get()
                            if val:
                                path = val.path if hasattr(val, "path") else str(val)
                                if path:
                                    return None, path

            # 2. Check for direct value (Color)
            value = input_attr.Get()
            if value:
                if isinstance(value, Gf.Vec3f):
                    return (
                        int(value[0] * 255),
                        int(value[1] * 255),
                        int(value[2] * 255),
                        255,
                    ), None

        return None, None

    def _load_texture(self, texture_path):
        """Load texture from path."""
        try:
            # Resolve path relative to stage
            if not os.path.isabs(texture_path):
                stage_path = self.stage.GetRootLayer().realPath
                if stage_path:
                    texture_path = os.path.join(
                        os.path.dirname(stage_path), texture_path
                    )

            if not os.path.exists(texture_path):
                return None, None

            image = Image.open(texture_path)
            image = image.convert("RGBA")
            width, height = image.size
            data = np.array(image)

            return data, rr.datatypes.ImageFormat(
                width=width, height=height, color_model="RGBA", channel_datatype="U8"
            )
        except Exception as e:
            print(f"Failed to load texture {texture_path}: {e}")
            return None, None


if __name__ == "__main__":
    test_usds = [
        # "/home/azazdeaz/repos/art/go2-example/assets/rail_blocks/rail_blocks.usd",
        # "/home/azazdeaz/repos/art/go2-example/assets/excavator_scan/excavator.usd",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/block.usd",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/dex_cube_instanceable.usd",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/colored_cube.usda",
        "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_block_letter/block_letter.usda",
    ]
    for usd_path in test_usds:
        print(f"\n\n\n>> Logging USD stage: {usd_path}")
        stage = Usd.Stage.Open(usd_path)
        logger = UsdRerunLogger(stage)
        logger.log_stage(frame_idx=0)
