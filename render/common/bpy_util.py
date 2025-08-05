import bpy
import math

def look_at(obj_camera, point):
    """
    orient the given camera with a fixed position to loot at a given point in space
    """
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()


def clean_scene(start_with_strings=["Camera", "procedural", "Light"]):
    """
    delete all object of which the name's prefix is matching any of the given strings
    """
    scene = bpy.context.scene
    bpy.ops.object.select_all(action='DESELECT')
    for obj in scene.objects:
        if any([obj.name.startswith(starts_with_string) for starts_with_string in start_with_strings]):
            # select the object
            if obj.visible_get():
                obj.select_set(True)
    bpy.ops.object.delete()


def del_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()

def normalize_scale(obj, scale=1.0):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # set origin to the center of the bounding box
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    obj.location.x = 0
    obj.location.y = 0
    obj.location.z = 0

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    max_vert_dist = max([math.sqrt(v.co.dot(v.co)) for v in obj.data.vertices])

    for v in obj.data.vertices:
        v.co /= (max_vert_dist / scale)

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
