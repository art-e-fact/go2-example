from isaacsim.core.utils.semantics import add_update_semantics
from pxr import Gf, UsdGeom, PhysxSchema, UsdPhysics
import omni.usd
from isaacsim.core.utils.prims import define_prim


def create_stepped_pyramid(
    prim_path="/World/SteppedPyramid",
    num_steps=4,
    step_height=0.08,
    step_width=0.4,
    position=(0.0, 0.0, 0.0),
    add_physics=True,
):
    """
    Creates a stepped pyramid using scaled cubes, often used in RL for quadrupeds.

    The pyramid is built from the bottom up. Each level is a single scaled cube.

    Args:
        path (str): The prim path for the root of the pyramid.
        num_steps (int): The number of levels in the pyramid, including the base.
        step_height (float): The height of each individual step.
        step_width (float): The width and depth of the smallest step (at the top).
        position (tuple): The (x, y, z) base position of the pyramid.
        add_physics (bool): Whether to add physics properties to the steps.
    """
    stage = omni.usd.get_context().get_stage()
    pyramid_root = UsdGeom.Xform.Define(stage, prim_path)
    UsdGeom.Xformable(pyramid_root).AddTranslateOp().Set(Gf.Vec3f(position))

    for level in range(num_steps):
        # The size of the step at this level (e.g., bottom level is largest)
        level_size_multiplier = num_steps - level
        step_size_xy = level_size_multiplier * step_width

        # The vertical position of the center of the cube for this step
        level_z = (level * step_height) + (step_height / 2.0)

        cube_path = pyramid_root.GetPath().AppendChild(f"Step_{level}")
        prim = stage.DefinePrim(cube_path, "Cube")

        # Set the position of the step
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3f(0, 0, level_z))

        # Set the scale for the cube to form the step
        UsdGeom.Xformable(prim).AddScaleOp().Set(
            Gf.Vec3f(step_size_xy, step_size_xy, step_height)
        )

        cube = UsdGeom.Cube(prim)
        cube.CreateSizeAttr(1.0)  # Base size is 1, scale does the work

        if add_physics:
            # Make the steps static colliders
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigidBodyAPI.CreateKinematicEnabledAttr(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)

        add_update_semantics(prim, "step", "class")

    # Add Waypoints
    wp_path = pyramid_root.GetPath().AppendChild("Waypoint_01")
    UsdGeom.Xform.Define(stage, wp_path).AddTranslateOp().Set(Gf.Vec3f(0, 2, 0))
    wp_path = pyramid_root.GetPath().AppendChild("Waypoint_02")
    UsdGeom.Xform.Define(stage, wp_path).AddTranslateOp().Set(Gf.Vec3f(0, 0, level_z))
    wp_path = pyramid_root.GetPath().AppendChild("Waypoint_03")
    UsdGeom.Xform.Define(stage, wp_path).AddTranslateOp().Set(Gf.Vec3f(0, -2, 0))
    wp_path = pyramid_root.GetPath().AppendChild("Waypoint_04")
    UsdGeom.Xform.Define(stage, wp_path).AddTranslateOp().Set(Gf.Vec3f(-1, 0, 0))
