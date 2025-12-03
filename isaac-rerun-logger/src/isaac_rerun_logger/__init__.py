from datetime import datetime, timedelta
import os
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


class UsdRerunLogger:
    """Visualize USD stages in Rerun."""

    def __init__(self, stage: Usd.Stage):
        self.stage = None
        self._logger_id = None
        self._logged_meshes = set()  # Track which meshes we've already logged
        self._last_transforms = {}  # Track last logged transforms for change detection

    def initialize(
        self,
        stage: Usd.Stage,
        logger_id: str = "isaac_rerun_logger",
        spawn=True,
        save_path: Path | None = None,
    ):
        """Initialize the Rerun logger with a USD stage."""
        self.stop()
        self.stage = stage

        # Add random postfix to logger ID to avoid conflicts
        self._logger_id = f"{logger_id}_{np.random.randint(10000)}"
        rr.init(self._logger_id, spawn=spawn)
        if save_path is not None:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rr.save(save_path)
        self._logged_meshes = set()  # Track which meshes we've already logged
        self._last_transforms: dict[
            str, Gf.Matrix4d
        ] = {}  # Track last logged transforms for change detection

    def stop(self):
        """Stop the Rerun logger."""
        if self._logger_id is not None:
            rr.disconnect()
            self._logger_id = None
            self.stage = None
            self._logged_meshes.clear()
            self._last_transforms.clear()

    def set_time(
        self,
        sequence: int | None = None,
        duration: int | float | timedelta | np.timedelta64 | None = None,
        timestamp: int | float | datetime | np.datetime64 | None = None,
    ) -> None:
        """
        Set the current time of the Rerun logger.

        Used for all subsequent logging on the same thread, until the next call to
        [`rerun.set_time`][], [`rerun.reset_time`][] or [`rerun.disable_timeline`][].

        For example: `set_time("frame_nr", sequence=frame_nr)`.

        There is no requirement of monotonicity. You can move the time backwards if you like.

        You are expected to set exactly ONE of the arguments `sequence`, `duration`, or `timestamp`.
        You may NOT change the type of a timeline, so if you use `duration` for a specific timeline,
        you must only use `duration` for that timeline going forward.

        Parameters
        ----------
        sequence:
            Used for sequential indices, like `frame_nr`.
            Must be an integer.
        duration:
            Used for relative times, like `time_since_start`.
            Must either be in seconds, a [`datetime.timedelta`][], or [`numpy.timedelta64`][].
            For nanosecond precision, use `numpy.timedelta64(nanoseconds, 'ns')`.
        timestamp:
            Used for absolute time indices, like `capture_time`.
            Must either be in seconds since Unix epoch, a [`datetime.datetime`][], or [`numpy.datetime64`][].
            For nanosecond precision, use `numpy.datetime64(nanoseconds, 'ns')`.

        """
        rr.set_time(
            "clock",
            sequence=sequence,
            duration=duration,
            timestamp=timestamp,
        )

    def log_stage(self):
        """
        Log the entire USD stage to Rerun.
        """
        if self.stage is None:
            print("Warning: USD stage is not initialized.")
            return

        # Traverse all prims in the stage
        current_paths = set()
        # Using Usd.TraverseInstanceProxies to traverse into instanceable prims (references)
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        for prim in self.stage.Traverse(predicate):
            # Skip guides
            if prim.GetAttribute("purpose").Get() == UsdGeom.Tokens.guide:
                continue

            entity_path = str(prim.GetPath())
            current_paths.add(entity_path)

            # Log transforms for all Xformable prims
            if prim.IsA(UsdGeom.Xformable):
                self._log_transform(prim, entity_path)

            # Log mesh geometry (only once per unique mesh)
            if prim.IsA(UsdGeom.Mesh):
                mesh_path = str(prim.GetPath())
                if mesh_path not in self._logged_meshes:
                    self._log_mesh(prim, entity_path)
                    self._logged_meshes.add(mesh_path)

            if prim.IsA(UsdGeom.Cube):
                cube_path = str(prim.GetPath())
                if cube_path not in self._logged_meshes:
                    self._log_cube(prim, entity_path)
                    self._logged_meshes.add(cube_path)

        # Clear the logged paths that are no longer present in the stage
        for path in list(self._last_transforms.keys()):
            if path not in current_paths:
                rr.log(path, rr.Clear.flat())
                del self._last_transforms[path]

    def _log_transform(self, prim: Usd.Prim, entity_path: str):
        """Log the transform of an Xformable prim."""
        if not prim.IsA(UsdGeom.Xformable):
            return

        # Get the local transformation
        xformable = UsdGeom.Xformable(prim)
        transform_matrix: Gf.Matrix4d = xformable.GetLocalTransformation()

        if (
            entity_path in self._last_transforms
            and self._last_transforms[entity_path] == transform_matrix
        ):
            return

        self._last_transforms[entity_path] = transform_matrix

        transform = Gf.Transform(transform_matrix)

        quaternion = transform.GetRotation().GetQuat()

        # Log the transform to Rerun
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=transform.GetTranslation(),
                quaternion=(*quaternion.GetImaginary(), quaternion.GetReal()),
                scale=transform.GetScale(),
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
            rr.log(entity_path, rr.Points3D(positions=vertices), static=True)
            return

        face_vertex_indices = np.array(face_vertex_indices_attr.Get())
        face_vertex_counts = np.array(face_vertex_counts_attr.Get())

        if face_vertex_indices is None or face_vertex_counts is None:
            rr.log(entity_path, rr.Points3D(positions=vertices), static=True)
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

                subset_triangles = triangles_list[subset_triangle_indices]

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
            static=True,
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

        shader: UsdShade.Shader = material.ComputeSurfaceSource()[0]
        if not shader:
            print("No surface shader found.")
            mdl_surface = material.GetOutput("mdl:surface")
            if mdl_surface and mdl_surface.HasConnectedSource():
                source, sourceName, sourceType = mdl_surface.GetConnectedSource()
                shader = UsdShade.Shader(source)
            else:
                return None

        implementation_source = shader.GetImplementationSource()

        if (
            implementation_source == "id"
            and shader.GetIdAttr().Get() == "UsdPreviewSurface"
        ):
            diffuse_color = shader.GetInput("diffuseColor")

            diffuse_color_source: UsdShade.ConnectableAPI = (
                diffuse_color.GetConnectedSource()[0]
            )

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
            diffuse_texture = shader.GetInput("diffuse_texture")
            if not diffuse_texture:
                print("No diffuse_texture input found in OmniPBR shader.")
                print(
                    "Shader inputs:", [inp.GetBaseName() for inp in shader.GetInputs()]
                )
                return None
            print(diffuse_texture.GetConnectedSource())
            diffuse_texture_source = diffuse_texture.GetConnectedSource()
            if not diffuse_texture_source:
                print("No connected source for diffuse_texture.")
                return None
            diffuse_texture_source, input_name, _ = diffuse_texture_source
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
                return img_data

        except Exception as e:
            print(f"Failed to load texture {texture_path}: {e}")
            return None

    def _log_cube(self, prim: Usd.Prim, entity_path: str):
        """Log a cube prim as a Rerun box."""
        cube = UsdGeom.Cube(prim)
        size_attr = cube.GetSizeAttr()
        size = size_attr.Get() if size_attr else 2.0
        half_size = size / 2.0

        color = None
        display_color_attr = cube.GetDisplayColorAttr()
        if display_color_attr and display_color_attr.HasValue():
            colors = display_color_attr.Get()
            if colors and len(colors) > 0:
                c = colors[0]
                color = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

        rr.log(
            entity_path,
            rr.Boxes3D(
                half_sizes=[half_size, half_size, half_size],
                colors=color,
                fill_mode="solid",
            ),
            static=True,
        )


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
