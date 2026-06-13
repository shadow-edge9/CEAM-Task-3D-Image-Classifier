import bpy
import random
import os

#NOTE TO USERS: PASTE THIS SCRIPT IN BLENDER. IT WILL NOT RUN IN PYTHON.

# Destination targets
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
SAVE_FOLDER = os.path.join(DESKTOP_PATH, "Dataset")

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

SHAPES_TO_GENERATE = ['CUBE', 'CONE', 'SPHERE']
IMAGES_PER_CLASS = 5  # Change this value if you want different number for each type


def clean_blender_cache():
    """Flushes empty data-blocks from volatile cache to protect system memory."""
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)


# --- THE GRADIENT ENGINE ---
for shape_type in SHAPES_TO_GENERATE:
    print(f"🚀 Launching gradient generation for: {shape_type}")

    for i in range(IMAGES_PER_CLASS):
        # 1. Clear out active scene items
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()

        # 2. Add Geometry with Randomized Scalar Sizes
        scale = random.uniform(0.6, 1.8)
        if shape_type == 'CUBE':
            bpy.ops.mesh.primitive_cube_add()
        elif shape_type == 'CONE':
            bpy.ops.mesh.primitive_cone_add()
        elif shape_type == 'SPHERE':
            bpy.ops.mesh.primitive_uv_sphere_add()

        obj = bpy.context.active_object
        obj.scale = (scale, scale, scale)

        # 3. Randomize Orientation (3D Rotations)
        obj.rotation_euler = (random.uniform(0, 6.28), random.uniform(0, 6.28), random.uniform(0, 6.28))

        # 4. BUILD THE ADVANCED DYNAMIC GRADIENT MATERIAL NODES
        mat = bpy.data.materials.new(name=f"Gradient_Mat_{shape_type}_{i}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Locate the standard Principled BSDF node
        principled = nodes.get('Principled BSDF')

        if principled:
            # Create new procedural material generator nodes
            tex_coord = nodes.new(type='ShaderNodeTexCoord')
            gradient_tex = nodes.new(type='ShaderNodeTexGradient')

            # FIXED: Internal API type name changed to 'ShaderNodeValToRGB'
            color_ramp = nodes.new(type='ShaderNodeValToRGB')

            # Configure Gradient Type randomly
            gradient_tex.gradient_type = random.choice(['LINEAR', 'DIAGONAL', 'SPHERICAL'])

            # Pick TWO entirely random colors for the gradient sweep
            color_1 = (random.random(), random.random(), random.random(), 1.0)
            color_2 = (random.random(), random.random(), random.random(), 1.0)

            # Inject colors into the Color Ramp boundaries (Pos 0.0 and Pos 1.0)
            color_ramp.color_ramp.elements[0].color = color_1
            color_ramp.color_ramp.elements[1].color = color_2

            # LINK THE SYSTEM TOGETHER:
            links.new(tex_coord.outputs['Generated'], gradient_tex.inputs['Vector'])
            links.new(gradient_tex.outputs['Factor'], color_ramp.inputs['Fac'])
            links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])

        obj.data.materials.append(mat)

        # 5. Randomize Light Intensity
        light = bpy.data.lights.get("Light")
        if light:
            light.energy = random.uniform(800, 3500)

        # 6. Render and Save
        bpy.context.scene.render.filepath = os.path.join(SAVE_FOLDER, f"{shape_type.lower()}_{i}.png")
        bpy.ops.render.render(write_still=True)

        # 7. Flush memory
        bpy.data.objects.remove(obj, do_unlink=True)
        clean_blender_cache()

print("\n🏁 --- DATASET COMPILED WITH RANDOMIZED DYNAMIC GRADIENTS --- 🏁")