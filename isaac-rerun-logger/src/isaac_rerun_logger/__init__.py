import os

import numpy as np
import rerun as rr
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


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
        vertices = np.array(mesh.GetPointsAttr().Get())

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
        triangles = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:
                # Already a triangle
                triangles.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 1],
                        face_vertex_indices[idx + 2],
                    ]
                )
            elif count == 4:
                # Quad - split into two triangles
                triangles.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 1],
                        face_vertex_indices[idx + 2],
                    ]
                )
                triangles.extend(
                    [
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + 2],
                        face_vertex_indices[idx + 3],
                    ]
                )
            else:
                # For polygons with more vertices, use simple fan triangulation
                for i in range(1, count - 1):
                    triangles.extend(
                        [
                            face_vertex_indices[idx],
                            face_vertex_indices[idx + i],
                            face_vertex_indices[idx + i + 1],
                        ]
                    )
            idx += count

        # Convert to numpy array and reshape for Rerun
        triangles_list = np.array(triangles, dtype=np.uint32).reshape(-1, 3)

        # Get normals if available
        normals_attr = mesh.GetNormalsAttr()
        normals = np.array(normals_attr.Get())
        normals_interpolation = normals_attr.GetMetadata("interpolation")
        if normals_interpolation == "faceVarying":
            # Convert face-varying normals to vertex normals
            vertex_normals = np.zeros_like(vertices)
            indices = np.array(face_vertex_indices)
            np.add.at(vertex_normals, indices, normals)

            # Normalize
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vertex_normals = vertex_normals / norms
            normals = vertex_normals

        # Get UVs if available
        texcoords = np.array(mesh.GetPrim().GetAttribute("primvars:st").Get())

        # --- Material and Texture Handling ---
        texture_buffer = None

        subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        if subsets:
            for subset in subsets:
                if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
                    print(
                        "Warning: Unsupported subset element type:",
                        subset.GetElementTypeAttr().Get(),
                    )
                    continue

                # Rearrange the mesh data to only include the subset
                included_triangles = subset.GetIndicesAttr().Get()
                if not included_triangles:
                    continue

                # Filter triangles to only include those in the subset
                print(" Total triangles:", len(triangles_list))
                subset_triangles = triangles_list[included_triangles]
                print(" Subset triangles:", len(subset_triangles))

                # TODO: Remove unused vertices

                texture_path = self._get_image_texture_path(subset.GetPrim())
                texture_buffer = self._load_texture(texture_path)

                self._log_mesh_data(
                    str(subset.GetPath()),
                    vertices,
                    np.array(subset_triangles),
                    normals,
                    texcoords,
                    texture_buffer,
                )

        else:
            texture_path = self._get_image_texture_path(prim)
            texture_buffer = self._load_texture(texture_path)

            self._log_mesh_data(
                entity_path,
                vertices,
                triangles_list,
                normals,
                texcoords,
                texture_buffer,
            )

    def _log_mesh_data(
        self,
        entity_path: str,
        vertices: np.ndarray,
        triangles_list: np.ndarray,
        normals: np.ndarray = None,
        texcoords: np.ndarray = None,
        texture_buffer: np.ndarray = None,
        albedo_factor: tuple = None,
    ):
        rr.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles_list,
                vertex_normals=normals,
                vertex_texcoords=texcoords,
                albedo_texture=texture_buffer,
                albedo_factor=albedo_factor,
            ),
        )

    def clear_logged_meshes(self):
        """Clear the cache of logged meshes, allowing them to be logged again."""
        self._logged_meshes.clear()

    def _get_image_texture_path(self, prim: Usd.Prim):
        """
        Get material color or texture path.
        Returns: texture_path or None
        """
        # return "/home/azazdeaz/repos/art/go2-example/assets/stone_stairs/stonestairs_c/SubUSDs/textures/Stairs stone tile_Bake1_PBR_Diffuse.png"
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material: UsdShade.Material = binding_api.ComputeBoundMaterial()[0]
        if not material:
            print(f"No material found for prim {prim.GetPath()}.")
            return None

        direct_binding = binding_api.GetDirectBinding()
        print(f"Direct binding: {direct_binding}")
        material = direct_binding.GetMaterial()
        print(f"Material after direct binding: {material}")

        print(f"\n\n\nFound material: {material.GetPath()}")

        shader: UsdShade.Shader = material.ComputeSurfaceSource()[0]
        if not shader:
            print("No surface shader found.")
            return None

        implementation_source = shader.GetImplementationSource()

        if (
            implementation_source == "id"
            and shader.GetIdAttr().Get() == "UsdPreviewSurface"
        ):
            diffuse_color = shader.GetInput("diffuseColor")
            # diffuse_color = shader.GetInput("diffuse_texture")
            print(f"Diffuse Color Input: {diffuse_color}")
            print(
                f" - Connected attributes: {diffuse_color.GetValueProducingAttribute()}"
            )

            diffuse_color_source: UsdShade.ConnectableAPI = (
                diffuse_color.GetConnectedSource()[0]
            )
            print(f"Diffuse Color Connected Source: {diffuse_color_source}")

            diffuse_color_source_file = diffuse_color_source.GetInput("file")
            diffuse_color_source_file_path = diffuse_color_source_file.Get()

            if not diffuse_color_source_file_path or not isinstance(
                diffuse_color_source_file_path, Sdf.AssetPath
            ):
                print("Diffuse color source is not a valid texture file path.")
                return None

            diffuse_color_source_st = diffuse_color_source.GetInput("st")

            print(f"Shader input: {diffuse_color_source_st.GetFullName()}")
            print(" - Get", diffuse_color_source_st.Get())
            print(" - GetAttr", diffuse_color_source_st.GetAttr())
            print(" - GetBaseName", diffuse_color_source_st.GetBaseName())
            print(" - GetConnectability", diffuse_color_source_st.GetConnectability())
            print(" - GetConnectedSource", diffuse_color_source_st.GetConnectedSource())
            print(
                " - GetConnectedSources", diffuse_color_source_st.GetConnectedSources()
            )
            print(" - GetDisplayGroup", diffuse_color_source_st.GetDisplayGroup())
            print(" - GetDocumentation", diffuse_color_source_st.GetDocumentation())
            print(" - GetFullName", diffuse_color_source_st.GetFullName())
            print(" - GetPrim", diffuse_color_source_st.GetPrim())
            print(
                " - GetRawConnectedSourcePaths",
                diffuse_color_source_st.GetRawConnectedSourcePaths(),
            )
            print(" - GetRenderType", diffuse_color_source_st.GetRenderType())
            print(" - GetSdrMetadata", diffuse_color_source_st.GetSdrMetadata())
            print(" - GetTypeName", diffuse_color_source_st.GetTypeName())
            print(
                " - GetValueProducingAttribute",
                diffuse_color_source_st.GetValueProducingAttribute(),
            )
            print(
                " - GetValueProducingAttributes",
                diffuse_color_source_st.GetValueProducingAttributes(),
            )

            st_source = diffuse_color_source_st.GetConnectedSource()[0]
            print(f"ST Connected Source: {st_source}")

            # print(" - Get", diffuse_color_source.Get())
            # print(" - GetConnectedSource", diffuse_color_source.GetConnectedSource())
            # print(" - GetConnectedSources", diffuse_color_source.GetConnectedSources())
            # print(" - GetInput", diffuse_color_source.GetInput())
            print(" - GetInputs", [i.GetFullName() for i in st_source.GetInputs()])
            # print(" - GetOutput", diffuse_color_source.GetOutput())
            print(" - GetOutputs", [o.GetFullName() for o in st_source.GetOutputs()])
            # print(" - GetRawConnectedSourcePaths", diffuse_color_source.GetRawConnectedSourcePaths())
            print(" - GetSchemaAttributeNames", st_source.GetSchemaAttributeNames())

            st_source_varname = st_source.GetInput("varname")
            print(f"Shader input: {st_source_varname.GetFullName()}")
            print(" - Get", st_source_varname.Get())
            print(" - GetAttr", st_source_varname.GetAttr())
            print(" - GetBaseName", st_source_varname.GetBaseName())
            print(" - GetConnectability", st_source_varname.GetConnectability())
            print(" - GetConnectedSource", st_source_varname.GetConnectedSource())
            print(" - GetConnectedSources", st_source_varname.GetConnectedSources())
            print(" - GetDisplayGroup", st_source_varname.GetDisplayGroup())
            print(" - GetDocumentation", st_source_varname.GetDocumentation())
            print(" - GetFullName", st_source_varname.GetFullName())
            print(" - GetPrim", st_source_varname.GetPrim())
            print(
                " - GetRawConnectedSourcePaths",
                st_source_varname.GetRawConnectedSourcePaths(),
            )
            print(" - GetRenderType", st_source_varname.GetRenderType())
            print(" - GetSdrMetadata", st_source_varname.GetSdrMetadata())
            print(" - GetTypeName", st_source_varname.GetTypeName())
            print(
                " - GetValueProducingAttribute",
                st_source_varname.GetValueProducingAttribute(),
            )
            print(
                " - GetValueProducingAttributes",
                st_source_varname.GetValueProducingAttributes(),
            )

            value_producing_attributes = UsdShade.Utils.GetValueProducingAttributes(
                st_source_varname
            )[0].Get()
            print(" - Value Producing Attributes:", (type(value_producing_attributes)))

            return None, diffuse_color_source_file_path.resolvedPath

        elif (
            implementation_source == UsdShade.Tokens.sourceAsset
            and shader.GetPrim()
            .GetAttribute("info:mdl:sourceAsset:subIdentifier")
            .Get()
            == "OmniPBR"
        ):
            print("OmniPBR shader detected")
            diffuse_texture = shader.GetInput("diffuse_texture")
            print(diffuse_texture.GetConnectedSource())
            diffuse_texture_source, input_name, _ = diffuse_texture.GetConnectedSource()
            diffuse_texture_source_file = diffuse_texture_source.GetInput(
                input_name
            ).Get()
            if not diffuse_texture_source_file or not isinstance(
                diffuse_texture_source_file, Sdf.AssetPath
            ):
                print("Diffuse texture source is not a valid texture file path.")
                return None
            return diffuse_texture_source_file.resolvedPath
        else:
            print(f"Unsupported shader type: {shader.GetIdAttr().Get()}")
            return None

    def _load_texture(self, texture_path):
        """Load texture from path."""
        if not texture_path:
            return None
        print(f"Loading texture from: {texture_path}")
        try:
            # Resolve path relative to stage
            if not os.path.isabs(texture_path):
                stage_path = self.stage.GetRootLayer().realPath
                if stage_path:
                    texture_path = os.path.join(
                        os.path.dirname(stage_path), texture_path
                    )

            if not os.path.exists(texture_path):
                print(f"Warning: Texture file does not exist: {texture_path}")
                return None

            with Image.open(texture_path) as img:
                img = img.convert("RGB")  # Ensure 3 channels
                # mirror the image vertically and horizontally
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                # img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_data = np.array(img)
                print(f" Texture size: {img_data.shape}")
                return img_data

        except Exception as e:
            print(f"Failed to load texture {texture_path}: {e}")
            return None


if __name__ == "__main__":
    test_usds = [
        # "/home/azazdeaz/repos/art/go2-example/assets/rail_blocks/rail_blocks.usd",
        # "/home/azazdeaz/repos/art/go2-example/assets/excavator_scan/excavator.usd",
        # "/home/azazdeaz/repos/art/go2-example/assets/stone_stairs/stone_stairs_f.usd",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/dex_cube_instanceable.usd",
        "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_dex_cube_instanceable/dex_cube_instanceable.usda",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/simpleShading.usda",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_block_letter/block_letter_flat.usda",
    ]
    for usd_path in test_usds:
        print(f"\n\n\n>> Logging USD stage: {usd_path}")
        stage = Usd.Stage.Open(usd_path)
        logger = UsdRerunLogger(stage)
        logger.log_stage(frame_idx=0)
