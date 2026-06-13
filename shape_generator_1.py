import bpy
import random
import os

#NOTE TO USERS: PASTE THIS SCRIPT IN BLENDER NOT IN PYTHON. IT WILL GET YOU NOWHERE.

# This puts the images right on your computer's Desktop in a folder called 'Dataset'
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
SAVE_FOLDER = os.path.join(DESKTOP_PATH, "Dataset")

# Creates the folder automatically if it isn't there
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Clear existing objects in the scene center
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

shapes = ['CUBE', 'CUBOID', 'CONE', 'SPHERE']

for i in range(60):
    shape = random.choice(shapes)

    if shape == 'CUBE':
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object
        scale = random.uniform(0.5, 2.0)
        obj.scale = (scale, scale, scale)
    elif shape == 'CUBOID':
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object
        obj.scale = (random.uniform(0.5, 2.5), random.uniform(0.5, 2.5), random.uniform(0.5, 2.5))
    elif shape == 'CONE':
        bpy.ops.mesh.primitive_cone_add()
        obj = bpy.context.active_object
        scale = random.uniform(0.5, 2.0)
        obj.scale = (scale, scale, scale)
    elif shape == 'SPHERE':
        bpy.ops.mesh.primitive_uv_sphere_add()
        obj = bpy.context.active_object
        scale = random.uniform(0.5, 2.0)
        obj.scale = (scale, scale, scale)

    # Randomize Orientation
    obj.rotation_euler = (random.uniform(0, 6.28), random.uniform(0, 6.28), random.uniform(0, 6.28))

    # Create and assign a random color material
    mat = bpy.data.materials.new(name=f"Material_{i}")
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get('Principled BSDF')
    if principled:
        # FIXED: Explicitly targeting 'Base Color' to prevent the attribute error
        principled.inputs["Base Color"].default_value = (random.random(), random.random(), random.random(), 1.0)
    obj.data.materials.append(mat)

    # Randomize Light Intensity
    light = bpy.data.lights.get("Light")
    if light:
        light.energy = random.uniform(500, 3000)

    # Render and save the image
    bpy.context.scene.render.filepath = os.path.join(SAVE_FOLDER, f"shape_{i}.png")
    bpy.ops.render.render(write_still=True)

    # Clean up object for next iteration
    bpy.data.objects.remove(obj, do_unlink=True)

print("FINISHED!")
