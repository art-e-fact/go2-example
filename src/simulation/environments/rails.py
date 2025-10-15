from isaacsim.core.utils.semantics import add_update_semantics
from pxr import  Sdf, UsdGeom, PhysxSchema, UsdPhysics
import omni.usd


def create_rails(prim_path="/World/Rail"):
    stage = omni.usd.get_context().get_stage()

    segments = 10
    segment_length = -0.8
    start_x = -1.2
    track_gauge = 1.435
    rail_width = 0.067
    rail_height = 0.172 * 0.6
    sleeper_height = 0.08
    sleeper_length = track_gauge * 1.2
    sleeper_width = 0.2

    root_path = Sdf.Path(prim_path)

    for i in range(segments):
        # create sleepers
        prim = stage.DefinePrim(root_path.AppendChild(f"Sleeper_{i}"), "Cube")
        UsdGeom.Xformable(prim).AddTranslateOp().Set(
            (i * segment_length + start_x, 0.0, -0.5 + sleeper_height / 2)
        )
        UsdGeom.Xformable(prim).AddScaleOp().Set(
            (sleeper_width, sleeper_length, sleeper_height)
        )
        cube = UsdGeom.Cube(prim)
        cube.CreateSizeAttr(1.0)
        rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigidBodyAPI.CreateKinematicEnabledAttr(True, False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        add_update_semantics(prim, "sleeper", "class")

        # create rails
        for side in [-1, 1]:
            side_name = "left" if side < 0 else "right"
            prim = stage.DefinePrim(root_path.AppendChild(f"Rail_{i}_{side_name}"), "Cube")
            cube = UsdGeom.Cube(prim)
            cube.CreateSizeAttr(1.0)
            UsdGeom.Xformable(prim).AddTranslateOp().Set(
                (
                    i * segment_length + start_x,
                    side * track_gauge / 2,
                    -0.5 + rail_height / 2 + sleeper_height,
                )
            )
            UsdGeom.Xformable(prim).AddScaleOp().Set(
                (abs(segment_length), rail_width, rail_height)
            )
            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigidBodyAPI.CreateKinematicEnabledAttr(True, False)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            add_update_semantics(prim, "rail", "class")
