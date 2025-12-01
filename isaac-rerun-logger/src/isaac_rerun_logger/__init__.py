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
        points_attr = mesh.GetPointsAttr()
        if not points_attr:
            return
        vertices = np.array(points_attr.Get())

        # Get face vertex indices
        face_vertex_indices_attr = mesh.GetFaceVertexIndicesAttr()
        face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()

        if not face_vertex_indices_attr or not face_vertex_counts_attr:
            rr.log(entity_path, rr.Points3D(positions=vertices))
            return

        face_vertex_indices = np.array(face_vertex_indices_attr.Get())
        face_vertex_counts = np.array(face_vertex_counts_attr.Get())

        if face_vertex_indices is None or face_vertex_counts is None:
            rr.log(entity_path, rr.Points3D(positions=vertices))
            return

        # --- Handle UVs ---
        # Use UsdGeom.PrimvarsAPI to handle indexed vs non-indexed primvars correctly
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        st_primvar = primvars_api.GetPrimvar("st")

        texcoords = None
        st_interpolation = "constant"

        if st_primvar:
            st_interpolation = st_primvar.GetInterpolation()

            # Get the data, resolving indices if present
            st_data = st_primvar.Get()
            st_indices = st_primvar.GetIndices()

            if st_data is not None:
                st_data = np.array(st_data)
                if st_indices:
                    st_indices = np.array(st_indices)
                    texcoords = st_data[st_indices]
                else:
                    texcoords = st_data

            print(
                f"Texcoords shape: {texcoords.shape if texcoords is not None else 'None'}"
            )
            print(f"Vertices shape: {vertices.shape}")
            print(f"ST Interpolation: {st_interpolation}")

        # --- Handle Normals ---
        normals_attr = mesh.GetNormalsAttr()
        normals = None
        normals_interpolation = "constant"
        if normals_attr.HasValue():
            normals = np.array(normals_attr.Get())
            normals_interpolation = normals_attr.GetMetadata("interpolation")

        # --- Flattening Logic ---
        # If UVs or Normals are face-varying, we must flatten the mesh to a triangle soup
        should_flatten = (st_interpolation == "faceVarying") or (
            normals_interpolation == "faceVarying"
        )

        # Fallback: if texcoords length matches face_vertex_indices length, treat as face-varying
        # (This handles cases where metadata might be missing or ambiguous but data shape is clear)
        if (
            texcoords is not None
            and len(texcoords) == len(face_vertex_indices)
            and len(texcoords) != len(vertices)
        ):
            should_flatten = True

        triangles_list = None

        # Map for subsets: face_index -> list of triangle_indices
        face_to_triangle_indices = [[] for _ in range(len(face_vertex_counts))]
        current_triangle_index = 0

        if should_flatten:
            print("Expanding vertices for face-varying data...")
            # Flatten positions: Create a new vertex for every face corner
            vertices = vertices[face_vertex_indices]

            # Flatten normals if they are vertex-interpolated
            if normals is not None:
                if normals_interpolation == "vertex":
                    normals = normals[face_vertex_indices]
                # if faceVarying, normals should already match face_vertex_indices length

            # Flatten UVs if they are vertex-interpolated
            if texcoords is not None:
                if st_interpolation == "vertex":
                    texcoords = texcoords[face_vertex_indices]
                # if faceVarying, texcoords should already match face_vertex_indices length

            # Generate trivial triangles (0,1,2), (3,4,5)...
            # But we must respect the polygon counts (3, 4, etc.)
            triangles = []
            idx = 0
            for face_idx, count in enumerate(face_vertex_counts):
                # The vertices for this face are at indices [idx, idx+1, ... idx+count-1] in our new arrays
                if count == 3:
                    triangles.extend([idx, idx + 1, idx + 2])
                    face_to_triangle_indices[face_idx].append(current_triangle_index)
                    current_triangle_index += 1
                elif count == 4:
                    triangles.extend([idx, idx + 1, idx + 2])
                    face_to_triangle_indices[face_idx].append(current_triangle_index)
                    current_triangle_index += 1

                    triangles.extend([idx, idx + 2, idx + 3])
                    face_to_triangle_indices[face_idx].append(current_triangle_index)
                    current_triangle_index += 1
                else:
                    # Fan triangulation
                    for i in range(1, count - 1):
                        triangles.extend([idx, idx + i, idx + i + 1])
                        face_to_triangle_indices[face_idx].append(
                            current_triangle_index
                        )
                        current_triangle_index += 1
                idx += count

            triangles_list = np.array(triangles, dtype=np.uint32).reshape(-1, 3)

        else:
            # Standard indexed mesh path (shared vertices)
            triangles = []
            idx = 0
            for face_idx, count in enumerate(face_vertex_counts):
                if count == 3:
                    triangles.extend(
                        [
                            face_vertex_indices[idx],
                            face_vertex_indices[idx + 1],
                            face_vertex_indices[idx + 2],
                        ]
                    )
                    face_to_triangle_indices[face_idx].append(current_triangle_index)
                    current_triangle_index += 1
                elif count == 4:
                    triangles.extend(
                        [
                            face_vertex_indices[idx],
                            face_vertex_indices[idx + 1],
                            face_vertex_indices[idx + 2],
                        ]
                    )
                    face_to_triangle_indices[face_idx].append(current_triangle_index)
                    current_triangle_index += 1

                    triangles.extend(
                        [
                            face_vertex_indices[idx],
                            face_vertex_indices[idx + 2],
                            face_vertex_indices[idx + 3],
                        ]
                    )
                    face_to_triangle_indices[face_idx].append(current_triangle_index)
                    current_triangle_index += 1
                else:
                    for i in range(1, count - 1):
                        triangles.extend(
                            [
                                face_vertex_indices[idx],
                                face_vertex_indices[idx + i],
                                face_vertex_indices[idx + i + 1],
                            ]
                        )
                        face_to_triangle_indices[face_idx].append(
                            current_triangle_index
                        )
                        current_triangle_index += 1
                idx += count

            triangles_list = np.array(triangles, dtype=np.uint32).reshape(-1, 3)

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
                included_faces = subset.GetIndicesAttr().Get()
                if not included_faces:
                    continue

                # Collect all triangles for these faces
                subset_triangle_indices = []
                for face_idx in included_faces:
                    if face_idx < len(face_to_triangle_indices):
                        subset_triangle_indices.extend(
                            face_to_triangle_indices[face_idx]
                        )

                if not subset_triangle_indices:
                    continue

                print(" Total triangles:", len(triangles_list))
                subset_triangles = triangles_list[subset_triangle_indices]
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
            mdl_surface = material.GetOutput("mdl:surface")
            if mdl_surface and mdl_surface.HasConnectedSource():
                source, sourceName, sourceType = mdl_surface.GetConnectedSource()
                print(f"Connected source: {source.GetPath()}")
                print(f"Source Name: {sourceName}")
                print(f"Source Type: {sourceType}")
                shader = UsdShade.Shader(source)
            else:
                return None

        implementation_source = shader.GetImplementationSource()
        print(f"Shader Implementation Source: {implementation_source}")

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
            if not diffuse_texture:
                print("No diffuse_texture input found in OmniPBR shader.")
                print(
                    "Shader inputs:", [inp.GetBaseName() for inp in shader.GetInputs()]
                )
                return None
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

        elif (
            implementation_source == UsdShade.Tokens.sourceAsset
            and shader.GetPrim()
            .GetAttribute("info:mdl:sourceAsset:subIdentifier")
            .Get()
            == "gltf_material"
        ):
            print("gltf_material shader detected")
            diffuse_texture = shader.GetInput("base_color_texture")
            print(diffuse_texture.GetConnectedSource())
            diffuse_texture_source = diffuse_texture.GetConnectedSource()[0]
            diffuse_texture_source_file: Sdf.AssetPath = (
                diffuse_texture_source.GetInput("texture").Get()
            )
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
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_dex_cube_instanceable/dex_cube_instanceable.usda",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/simpleShading.usda",
        # "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_block_letter/block_letter_flat.usda",
        "/home/azazdeaz/repos/art/go2-example/isaac-rerun-logger/assets/Collected_go2-piamid/go2-piamid.usda",
    ]
    for usd_path in test_usds:
        print(f"\n\n\n>> Logging USD stage: {usd_path}")
        stage = Usd.Stage.Open(usd_path)
        logger = UsdRerunLogger(stage)
        logger.log_stage(frame_idx=0)
