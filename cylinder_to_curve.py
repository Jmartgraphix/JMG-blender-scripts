bl_info = {
    "name": "Cylinder to Curve",
    "author": "Jason K. Martin",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D Viewport > F3 Search > Cylinder to Curve",
    "description": "Convert selected cylinders to curves aligned to cylinder axis; optional bevel and keep-original",
    "category": "Object",
    "release_date": "2025-10-29",
}

"""
Blender Python Script: Cylinder to Curve
Creates a two-point curve centered within a cylinder, aligned to the cylinder's axis,
and removes the original cylinder geometry.
"""

import bpy
from mathutils import Vector
import numpy as np

def apply_rotation_scale(obj):
    """
    Apply rotation and scale to the given object so geometry matches its world-space orientation.
    """
    # Ensure we're in Object mode
    active_obj = bpy.context.view_layer.objects.active
    if active_obj and hasattr(active_obj, "mode") and active_obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Select only this object and make it active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Apply rotation and scale
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

def analyze_cylinder_geometry(obj):
    """
    Robustly compute the cylinder's main axis, length, center, and radius using PCA.
    Returns: (axis_world: Vector, length: float, center_world: Vector, radius: float)
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    try:
        if len(eval_mesh.vertices) == 0:
            return Vector((0, 0, 1)), 0.0, eval_obj.matrix_world.translation.copy(), 0.0

        # Collect vertex coordinates in world space
        world_coords = np.array([
            (eval_obj.matrix_world @ v.co)[:] for v in eval_mesh.vertices
        ], dtype=np.float64)

        # PCA: center the points
        mean = world_coords.mean(axis=0)
        centered = world_coords - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_idx = int(np.argmax(eigvals))
        axis_world_np = eigvecs[:, principal_idx]

        # Ensure deterministic direction (optional)
        axis_world_np = axis_world_np / np.linalg.norm(axis_world_np)
        axis_world = Vector(axis_world_np.tolist()).normalized()

        # Project points onto axis to get length and center
        t = centered @ axis_world_np  # scalar projection relative to mean
        t_min = float(t.min())
        t_max = float(t.max())
        length = t_max - t_min
        center_world = Vector((mean + axis_world_np * ((t_min + t_max) * 0.5)).tolist())

        # Radius: median distance of points to axis
        # Compute perpendicular distances: |(p - center) - dot(p - center, a) * a|
        vecs = world_coords - center_world[:]
        axial = (vecs @ axis_world_np)[:, None] * axis_world_np[None, :]
        perp = vecs - axial
        dists = np.linalg.norm(perp, axis=1)
        radius = float(np.median(dists))

        return axis_world, float(length), center_world, radius
    finally:
        eval_obj.to_mesh_clear()

def compute_cylinder_radius(cylinder_obj):
    _, _, _, radius = analyze_cylinder_geometry(cylinder_obj)
    return radius

def ensure_collection(name):
    """
    Ensure a collection with the given name exists and is linked to the scene.
    Returns the collection.
    """
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    else:
        # Make sure it's linked to the scene
        if col.name not in {c.name for c in bpy.context.scene.collection.children}:
            bpy.context.scene.collection.children.link(col)
    return col

def get_cylinder_axis_and_length(obj):
    """
    Determine the cylinder's main axis and length using PCA for robustness.
    Returns: (axis_vector, length, center)
    """
    axis_world, length, center_world, _ = analyze_cylinder_geometry(obj)
    return axis_world, length, center_world

def create_cylinder_curve(cylinder_obj, bevel_radius=None):
    """
    Create a two-point curve centered within the cylinder and aligned to its axis.
    """
    # Get cylinder information via PCA
    axis, length, center = get_cylinder_axis_and_length(cylinder_obj)
    
    # Calculate curve endpoints (centered along the axis)
    half_length = length / 2
    point1 = center - axis * half_length
    point2 = center + axis * half_length
    
    # Create a new curve data block
    curve_data = bpy.data.curves.new(name=cylinder_obj.name + "_curve", type='CURVE')
    curve_data.dimensions = '3D'
    if bevel_radius is not None and bevel_radius > 0:
        curve_data.bevel_depth = bevel_radius
        curve_data.bevel_resolution = 8
    
    # Create a spline in the curve
    spline = curve_data.splines.new('NURBS')
    spline.points.add(1)  # Add one more point (total 2 points)
    spline.points[0].co = (*point1, 1.0)  # NURBS points use 4D coordinates
    spline.points[1].co = (*point2, 1.0)
    spline.use_endpoint_u = True  # Make the curve go through the endpoints
    
    # Create a new object with the curve
    curve_obj = bpy.data.objects.new(cylinder_obj.name + "_curve", curve_data)
    
    # Link to scene
    bpy.context.collection.objects.link(curve_obj)
    
    return curve_obj

def convert_single_cylinder(cylinder_obj, bevel=False, keep_original=False):
    """
    Convert a single cylinder to a curve and remove the cylinder.
    Returns the created curve object, or None if conversion failed.
    """
    cylinder_name = cylinder_obj.name
    
    # Check if it's a mesh
    if cylinder_obj.type != 'MESH':
        print(f"Warning: '{cylinder_name}' is not a mesh, skipping")
        return None
    
    try:
        source_obj = cylinder_obj
        temp_obj = None

        if keep_original:
            # Duplicate object and mesh so we can safely apply transforms
            temp_obj = cylinder_obj.copy()
            temp_obj.data = cylinder_obj.data.copy()
            temp_obj.matrix_world = cylinder_obj.matrix_world.copy()
            bpy.context.collection.objects.link(temp_obj)
            source_obj = temp_obj

        # Apply rotation and scale so the geometry reflects current orientation
        apply_rotation_scale(source_obj)

        # Compute bevel radius before removing/cleaning
        bevel_radius = compute_cylinder_radius(source_obj) if bevel else None

        # Create the curve from the transformed source
        curve_obj = create_cylinder_curve(source_obj, bevel_radius=bevel_radius)

        # Cleanup or move original
        if keep_original:
            # Remove the temporary duplicate if created
            if temp_obj is not None:
                bpy.data.objects.remove(temp_obj, do_unlink=True)

            # Move the original to the 'original_meshes' collection
            target_col = ensure_collection('original_meshes')
            if target_col not in cylinder_obj.users_collection:
                target_col.objects.link(cylinder_obj)
            # Unlink from other collections to effectively move it
            for col in list(cylinder_obj.users_collection):
                if col != target_col:
                    try:
                        col.objects.unlink(cylinder_obj)
                    except Exception:
                        pass
        else:
            # Remove the original cylinder
            bpy.data.objects.remove(cylinder_obj, do_unlink=True)
        
        print(f"Successfully converted '{cylinder_name}' to a curve")
        return curve_obj
    except Exception as e:
        print(f"Error converting '{cylinder_name}': {e}")
        return None

def cylinder_to_curve(bevel=False, keep_original=False):
    """
    Main function: Convert selected cylinders to curves and remove the cylinders.
    Works with single or multiple selected objects.
    """
    # Get all selected objects
    selected_objects = [obj for obj in bpy.context.selected_objects]
    
    if not selected_objects:
        print("Error: No objects selected")
        return
    
    # Filter for mesh objects only
    mesh_objects = [obj for obj in selected_objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("Error: No mesh objects selected")
        return
    
    # Process each cylinder and collect created curves
    created_curves = []
    for cylinder_obj in mesh_objects:
        curve_obj = convert_single_cylinder(cylinder_obj, bevel=bevel, keep_original=keep_original)
        if curve_obj is not None:
            created_curves.append(curve_obj)
    
    # Select all created curves and make the last one active
    if created_curves:
        bpy.ops.object.select_all(action='DESELECT')
        for curve_obj in created_curves:
            curve_obj.select_set(True)
        bpy.context.view_layer.objects.active = created_curves[-1]
        
        print(f"\nCompleted: Converted {len(created_curves)} cylinder(s) to curve(s)")
    else:
        print("Warning: No curves were created")

class OBJECT_OT_cylinder_to_curve(bpy.types.Operator):
    bl_idname = "object.cylinder_to_curve"
    bl_label = "Cylinder to Curve"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Ensure it's callable in Object Mode from the 3D Viewport and similar contexts
        return context is not None and getattr(context, "mode", "OBJECT") == 'OBJECT'

    bevel: bpy.props.BoolProperty(
        name="Bevel curve to match cylinder",
        description="Give the created curve a round bevel matching the source cylinder diameter",
        default=False,
    )

    keep_original: bpy.props.BoolProperty(
        name="Keep original mesh (move to 'original_meshes')",
        description="Preserve the source mesh and move it into the 'original_meshes' collection",
        default=False,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        cylinder_to_curve(bevel=self.bevel, keep_original=self.keep_original)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(OBJECT_OT_cylinder_to_curve)
    # Add to Object menu for discoverability (also helps F3 indexing)
    try:
        bpy.types.VIEW3D_MT_object.append(menu_func_cylinder_to_curve)
    except Exception:
        pass
    # Add default hotkey: Ctrl+Alt+C in Object Mode
    try:
        wm = bpy.context.window_manager
        kc = wm.keyconfigs.addon
        if kc is not None:
            km = kc.keymaps.new(name='Object Mode', space_type='EMPTY')
            kmi = km.keymap_items.new(OBJECT_OT_cylinder_to_curve.bl_idname, 'C', 'PRESS', ctrl=True, alt=True)
            _addon_keymaps.append((km, kmi))
    except Exception:
        pass


def unregister():
    try:
        bpy.types.VIEW3D_MT_object.remove(menu_func_cylinder_to_curve)
    except Exception:
        pass
    bpy.utils.unregister_class(OBJECT_OT_cylinder_to_curve)
    # Remove hotkey(s)
    try:
        for km, kmi in _addon_keymaps:
            km.keymap_items.remove(kmi)
        _addon_keymaps.clear()
    except Exception:
        pass


def menu_func_cylinder_to_curve(self, context):
    self.layout.operator(OBJECT_OT_cylinder_to_curve.bl_idname, text="Cylinder to Curve")


# Storage for add-on keymaps so we can unregister cleanly
_addon_keymaps = []


# Run the script
if __name__ == "__main__":
    register()
    try:
        bpy.ops.object.cylinder_to_curve('INVOKE_DEFAULT')
    except Exception:
        # Fallback to direct execution if UI isn't available
        cylinder_to_curve(bevel=False)
