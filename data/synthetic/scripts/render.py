import subprocess
import sys
import os

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, installing...")
    python_exe = sys.executable
    subprocess.check_call([python_exe, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm
    print("tqdm installed successfully!")

try:
    import argparse
except ImportError:
    print("argparse not found, installing...")
    python_exe = sys.executable
    subprocess.check_call([python_exe, "-m", "pip", "install", "argparse"])
    import argparse
    print("argparse installed successfully!")

import subprocess

import glob

import bpy
import json
import math
import mathutils

import contextlib
from io import StringIO
import logging

@contextlib.contextmanager
def suppress_blender_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    class FilteredOutput:
        def __init__(self, original_stream, allow_print=True):
            self.original_stream = original_stream
            self.allow_print = allow_print
            self.buffer = ""
            
        def write(self, text):
            if self.allow_print and any(keyword in text for keyword in [
                "Object motion boundary", "Rendering complete", "Rendering progress", "Creating infinite desktop", 
                "tqdm", "Progress", "Current frame", "Remaining"
            ]):
                self.original_stream.write(text)
                self.original_stream.flush()
            elif any(blender_keyword in text for blender_keyword in [
                "Fra:", "Mem:", "Time:", "Remaining:", "Sample", "Tiles", 
                "Scene", "ViewLayer", "Rendered", "Loading denoising", 
                "Finished", "Denoising", "Saving:", "Reading full buffer"
            ]):
                pass 
            else:
                self.original_stream.write(text)
                self.original_stream.flush()
                
        def flush(self):
            self.original_stream.flush()
            
        def isatty(self):
            return self.original_stream.isatty()
    
    try:
        sys.stdout = FilteredOutput(old_stdout, allow_print=True)
        sys.stderr = FilteredOutput(old_stderr, allow_print=True)
        
        if hasattr(bpy.app, 'debug'):
            bpy.app.debug = False
        
        if hasattr(bpy.context.scene.cycles, 'debug_use_spatial_splits'):
            bpy.context.scene.cycles.debug_use_spatial_splits = False
        
        yield
        
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

@contextlib.contextmanager
def suppress_stdout():
    with suppress_blender_output():
        yield


def create_material(name, color, is_metal):
    if name in materials:
        return materials[name]

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    principled_bsdf = nodes.get('Principled BSDF')
    rgba = tuple(color[:3]) + (1.0,)
    principled_bsdf.inputs['Base Color'].default_value = rgba

    if is_metal:
        principled_bsdf.inputs['Metallic'].default_value = 0.9
        principled_bsdf.inputs['Roughness'].default_value = 0.2
    else: 
        principled_bsdf.inputs['Metallic'].default_value = 0.0
        principled_bsdf.inputs['Roughness'].default_value = 0.7

    materials[name] = mat
    return mat

def create_object(obj_data):
    shape = obj_data['shape']
    material = obj_data['material']
    color_name = obj_data['color']

    if shape == 'cube':
        scale = [0.167, 0.167, 0.167]  
    elif shape == 'sphere':
        scale = [0.167] 
    elif shape == 'cylinder':
        scale = [0.167, 0.167]  
    else:
        scale = [0.167, 0.167, 0.167]  
    
    if len(sim_data['motion_trajectory']) > 0 and len(sim_data['motion_trajectory'][0]['objects']) > obj_data['object_id']:
        position = sim_data['motion_trajectory'][0]['objects'][obj_data['object_id']]['location']
    else:
        position = [0, 0, 0]  

    color_map = {
        'gray': [0.5, 0.5, 0.5, 1],
        'red': [0.8, 0.1, 0.1, 1],
        'blue': [0.1, 0.1, 0.8, 1],
        'green': [0.1, 0.8, 0.1, 1],
        'brown': [0.6, 0.4, 0.2, 1],
        'cyan': [0.1, 0.8, 0.8, 1],
        'purple': [0.8, 0.1, 0.8, 1],
        'yellow': [0.8, 0.8, 0.1, 1]
    }
    color = color_map.get(color_name, [0.8, 0.8, 0.8, 1])


    if shape == 'cube':
        bpy.ops.mesh.primitive_cube_add(location=position)
        obj = bpy.context.active_object
        obj.scale = scale
        bevel = obj.modifiers.new('Bevel', 'BEVEL')
        bevel.width = 0.03
        bevel.segments = 3
        bevel.limit_method = 'ANGLE'
        bevel.angle_limit = 0.7 
    elif shape == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=scale[0], location=position, segments=48, ring_count=24)
        obj = bpy.context.active_object
        subsurf = obj.modifiers.new('Subdivision', 'SUBSURF')
        subsurf.levels = 1
        subsurf.render_levels = 2
    elif shape == 'cylinder':
        # More vertices for smoother cylinder
        bpy.ops.mesh.primitive_cylinder_add(radius=scale[0], depth=scale[1]*2, location=position, vertices=48)
        obj = bpy.context.active_object
        # Add bevel to cylinder caps
        bevel = obj.modifiers.new('Bevel', 'BEVEL')
        bevel.width = 0.02
        bevel.segments = 3
        bevel.limit_method = 'ANGLE'
        bevel.angle_limit = 1.0  

    # Apply material
    is_metal = (material == 'metal')
    mat_name = f"{color_name}_{material}"
    mat = create_material(mat_name, color, is_metal)

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Enable shadow casting
    obj.cycles.cast_shadow = True

    return obj


def is_in_camera_view(obj, scene, camera):
    """Check if object is visible in camera's field of view"""
    from bpy_extras.object_utils import world_to_camera_view
    
    # Get the bounding box corners in world space
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    
    # Check if any corner is in the camera's view
    for corner in bbox_corners:
        # Convert corner to camera view coordinates (NDC)
        co_ndc = world_to_camera_view(scene, camera, corner)
        
        # Check if point is inside frustum (0 to 1 range for visible area)
        if (0.0 <= co_ndc.x <= 1.0 and 
            0.0 <= co_ndc.y <= 1.0 and 
            co_ndc.z > 0):  # Point is in front of camera
            return True
            
    return False

def calculate_motion_bounds(motion_trajectory):
    """Calculate the boundary of all objects"""
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    
    for frame_data in motion_trajectory:
        for obj in frame_data['objects']:
            pos = obj['location']
            min_x = min(min_x, pos[0])
            min_y = min(min_y, pos[1])
            min_z = min(min_z, pos[2])
            max_x = max(max_x, pos[0])
            max_y = max(max_y, pos[1])
            max_z = max(max_z, pos[2])
    
    if min_x == float('inf'):
        return {'center': [0, 0, 0], 'size': [2, 2, 2]}
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    # size_x = max(max_x - min_x, 1.0)  
    # size_y = max(max_y - min_y, 1.0)
    # size_z = max(max_z - min_z, 1.0)
    size_x = 5.0
    size_y = 5.0
    size_z = 1.0

    return {
        'center': [center_x, center_y, center_z],
        'size': [size_x, size_y, size_z],
        'min': [min_x, min_y, min_z],
        'max': [max_x, max_y, max_z]
    }

def parse_args():
    import sys
    import argparse
    
    script_args = []
    if '--' in sys.argv:
        separator_index = sys.argv.index('--')
        script_args = sys.argv[separator_index + 1:]
    else:
        skip_next = False
        for i, arg in enumerate(sys.argv):
            if skip_next:
                skip_next = False
                continue
            

            if (arg == '--python' or 
                arg == '--background' or 
                'blender' in arg.lower() or 
                arg.endswith('.py')):
                if arg == '--python':
                    skip_next = True  
                continue
            
            if arg.startswith('--'):
                script_args.append(arg)
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                    script_args.append(sys.argv[i + 1])
                    skip_next = True
    
    parser = argparse.ArgumentParser(description='Video renderer')
    parser.add_argument('--scene', type=str, default='overdetermination', help='render scene')
    parser.add_argument('--input_dir', type=str, default='simulation_output', help='input file')
    parser.add_argument('--begin', type=int, default=0, help='begin index')
    parser.add_argument('--end', type=int, default=99, help='end index')
    parser.add_argument('--output_dir', type=str, default='render_output', help='output file')
    parser.add_argument('--camera_distance_factor', type=float, default=1.5, help='control the distance between camera and center')
    parser.add_argument('--camera_height_factor', type=float, default=0.8, help='control the height of the camera')
    parser.add_argument('--camera_angle', type=float, default=30.0, help='control overhead angle')
    
    args = parser.parse_args(script_args)
    return args


if __name__ == "__main__":
    args = parse_args()
    scene = args.scene  # Change this to the desired setting
    begin = args.begin
    end = args.end
    indices = range(begin, end + 1)
    # scene_list = ['overdetermination', 'switch', 'early', 'late', 'double', 'bogus', 'all']
    if scene == 'all':
        scene_list = ['overdetermination', 'switch', 'early', 'late', 'double', 'bogus']
    else:
        scene_list = [scene]
    for scene in scene_list:
            
        for index in indices:

                
            input_dir = os.path.join(args.input_dir, scene)
            data_path = os.path.join(input_dir, f"annotation_{index:05d}.json")
            output_dir = os.path.join(args.output_dir, scene)
            # Clear existing objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()

            with open(data_path, 'r') as f:
                sim_data = json.load(f)

            # Scene setup
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.scene.cycles.device = 'GPU'
            ## New code
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = 'OPTIX'
            for device in prefs.devices:
                device.use = True

            # Increase samples for better GPU usage
            bpy.context.scene.cycles.samples = 128

            # Enable persistent data for better rendering efficiency
            bpy.context.scene.render.use_persistent_data = True

            # Set tile rendering for better GPU utilization
            bpy.context.scene.cycles.tile_size = 256  
            bpy.context.scene.cycles.progressive = 'PATH'  

            # Set memory management
            bpy.context.scene.cycles.use_auto_tile = True
            bpy.context.scene.cycles.auto_tile_size = 2048  

            # Set frame rate and animation length
            fps = 25  # Default CLEVRER fps
            duration = 5  # Default CLEVRER duration
            total_frames = fps * duration

            # Ensure Cycles plugin is enabled
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" 
            bpy.context.scene.cycles.device = 'GPU'

            # Enable all available GPU devices
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.get_devices()  # Refresh device list
            for device in prefs.devices:
                device.use = True

            bpy.context.scene.render.fps = fps
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = total_frames - 1

            # Create table surface (instead of floor)
            motion_bounds = calculate_motion_bounds(sim_data['motion_trajectory'])
            
            center = motion_bounds['center']
            size = motion_bounds['size']
            max_dimension = max(size[0], size[1])
            camera_distance = max_dimension * args.camera_distance_factor
            
            table_size = camera_distance * 10
            table_size = max(table_size, 200)  
            
            bpy.ops.mesh.primitive_plane_add(size=table_size, location=(center[0], center[1], 0))
            table = bpy.context.active_object
            table.name = 'Table'
            
            table_mat = bpy.data.materials.new(name="InfiniteTableMaterial")
            table_mat.use_nodes = True
            
            nodes = table_mat.node_tree.nodes
            links = table_mat.node_tree.links
            
            for node in nodes:
                nodes.remove(node)
            
            output = nodes.new('ShaderNodeOutputMaterial')
            principaled = nodes.new('ShaderNodeBsdfPrincipled')
            coord = nodes.new('ShaderNodeTexCoord')
            mapping = nodes.new('ShaderNodeMapping')
            noise = nodes.new('ShaderNodeTexNoise')
            colorRamp = nodes.new('ShaderNodeValToRGB')
            
            principaled.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1.0)
            principaled.inputs['Metallic'].default_value = 0.0
            principaled.inputs['Roughness'].default_value = 0.8
            
            noise.inputs['Scale'].default_value = 50.0
            noise.inputs['Detail'].default_value = 2.0
            noise.inputs['Roughness'].default_value = 0.5
            
            colorRamp.color_ramp.elements[0].color = (0.01, 0.01, 0.01, 1.0)
            colorRamp.color_ramp.elements[1].color = (0.03, 0.03, 0.03, 1.0)
            
            links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
            links.new(noise.outputs['Fac'], colorRamp.inputs['Fac'])
            links.new(colorRamp.outputs['Color'], principaled.inputs['Base Color'])
            links.new(principaled.outputs['BSDF'], output.inputs['Surface'])
            
            if len(table.data.materials) > 0:
                table.data.materials[0] = table_mat
            else:
                table.data.materials.append(table_mat)
            
            bpy.ops.mesh.primitive_plane_add(size=table_size * 2, location=(center[0], center[1], -0.01))
            background_plane = bpy.context.active_object
            background_plane.name = 'BackgroundPlane'
            
            bg_mat = bpy.data.materials.new(name="BackgroundMaterial")
            bg_mat.use_nodes = True
            bg_nodes = bg_mat.node_tree.nodes
            bg_links = bg_mat.node_tree.links
            
            for node in bg_nodes:
                bg_nodes.remove(node)
            
            bg_output = bg_nodes.new('ShaderNodeOutputMaterial')
            bg_principled = bg_nodes.new('ShaderNodeBsdfPrincipled')
            
            bg_principled.inputs['Base Color'].default_value = (0.005, 0.005, 0.005, 1.0)  
            bg_principled.inputs['Roughness'].default_value = 1.0
            bg_principled.inputs['Metallic'].default_value = 0.0
            
            bg_links.new(bg_principled.outputs['BSDF'], bg_output.inputs['Surface'])
            
            if len(background_plane.data.materials) > 0:
                background_plane.data.materials[0] = bg_mat
            else:
                background_plane.data.materials.append(bg_mat)

            nodes = table_mat.node_tree.nodes
            links = table_mat.node_tree.links

            for node in nodes:
                nodes.remove(node)

            output = nodes.new('ShaderNodeOutputMaterial')
            principled = nodes.new('ShaderNodeBsdfPrincipled')

            principled.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1.0)  # Near black
            principled.inputs['Metallic'].default_value = 0.0
            principled.inputs['Roughness'].default_value = 0.8
            # principled.inputs['Specular'].default_value = 0.1

            links.new(principled.outputs['BSDF'], output.inputs['Surface'])

            if len(table.data.materials) > 0:
                table.data.materials[0] = table_mat
            else:
                table.data.materials.append(table_mat)

            # Create materials dictionary to reuse materials
            materials = {}

            # Create a list to store the created objects
            blender_objects = []

            # Create objects from simulation data
            for obj_data in sim_data['object_property']:
                blender_obj = create_object(obj_data)
                blender_objects.append(blender_obj)

            # Set keyframes for object motion
            for frame_data in sim_data['motion_trajectory']:
                frame_idx = frame_data['frame_id']
                bpy.context.scene.frame_set(frame_idx)

                for obj_transform in frame_data['objects']:
                    obj_idx = obj_transform['object_id']
                    if obj_idx < len(blender_objects):
                        obj = blender_objects[obj_idx]
                        
                        # Set position
                        position = obj_transform['location']
                        obj.location = position
                        obj.keyframe_insert(data_path="location", frame=frame_idx)
                        
                        # Set orientation (convert quaternion to Blender format)
                        orientation = obj_transform['orientation']
                        # Blender uses WXYZ format whereas many physics engines use XYZW
                        quat = mathutils.Quaternion([orientation[3], orientation[0], orientation[1], orientation[2]])
                        obj.rotation_mode = 'QUATERNION'
                        obj.rotation_quaternion = quat
                        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)


            # Remove default lights if any
            for obj in bpy.data.objects:
                if obj.type == 'LIGHT':
                    bpy.data.objects.remove(obj)

            # Key light (main directional light)
            bpy.ops.object.light_add(type='SUN', location=(6, -6, 8))
            key_light = bpy.context.active_object
            key_light.name = 'Key Light'
            key_light.data.energy = 2.0  
            key_light.data.angle = 0.15  
            key_light.rotation_euler = (
                math.radians(45),  
                math.radians(-15), 
                math.radians(35)   
            )
            key_light.data.use_contact_shadow = True  
            key_light.data.contact_shadow_distance = 0.3
            key_light.data.contact_shadow_thickness = 0.01

            # Fill light (softer light to fill shadows)
            bpy.ops.object.light_add(type='AREA', location=(-5, -3, 4))
            fill_light = bpy.context.active_object
            fill_light.name = 'Fill Light'
            fill_light.data.energy = 20.0  
            fill_light.data.size = 5
            fill_light.data.shadow_soft_size = 1.2  
            fill_light.rotation_euler = (
                math.radians(25),
                math.radians(10),
                math.radians(-45)
            )

            # Back light (rim light for separation)
            bpy.ops.object.light_add(type='AREA', location=(0, 7, 5))
            back_light = bpy.context.active_object
            back_light.name = 'Back Light'
            back_light.data.energy = 40.0  
            back_light.data.size = 4
            back_light.data.shadow_soft_size = 0.5  
            back_light.rotation_euler = (
                math.radians(-35),
                math.radians(5),
                math.radians(165)
            )

            # Add a low fill light to enhance table shadows
            bpy.ops.object.light_add(type='AREA', location=(0, 0, 0.5))
            table_light = bpy.context.active_object
            table_light.name = 'Table Light'
            table_light.data.energy = 10.0 
            table_light.data.size = 15
            table_light.data.shadow_soft_size = 1.0
            table_light.rotation_euler = (
                math.radians(150),
                math.radians(0),
                math.radians(20)
            )

            motion_bounds = calculate_motion_bounds(sim_data['motion_trajectory'])
            print(f"Motion boundary: {motion_bounds}")

            center = motion_bounds['center']
            size = motion_bounds['size']

            max_dimension = max(size[0], size[1])
            camera_distance = max_dimension * args.camera_distance_factor * 1.4 

            camera_angle_rad = math.radians(args.camera_angle)
            camera_x = center[0] - camera_distance * math.cos(camera_angle_rad)
            camera_y = center[1] - camera_distance * math.sin(camera_angle_rad)
            camera_z = center[2] + size[2] * args.camera_height_factor + camera_distance * 0.5

            bpy.ops.object.camera_add(location=(camera_x, camera_y, camera_z))
            camera = bpy.context.active_object
            camera.name = 'Camera'

            camera_constraint = camera.constraints.new(type='TRACK_TO')
            camera_constraint.target = bpy.data.objects.new("Empty", None)
            bpy.context.collection.objects.link(camera_constraint.target)

            camera_constraint.target.location = (center[0], center[1], center[2] + size[2] * 0.2)
            camera_constraint.track_axis = 'TRACK_NEGATIVE_Z'
            camera_constraint.up_axis = 'UP_Y'

            bpy.context.scene.camera = camera

            # Setup render settings
            bpy.context.scene.render.resolution_x = 1920
            bpy.context.scene.render.resolution_y = 1080
            bpy.context.scene.render.resolution_percentage = 100

            # Set output to image sequence instead of video
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            frames_dir = os.path.join(output_dir, f"{index:02d}")

            if os.path.exists(frames_dir):
                continue

            os.makedirs(frames_dir)
            bpy.context.scene.render.filepath = os.path.join(frames_dir, "frame_")

            # Add world environment with darker settings
            world = bpy.context.scene.world
            if world is None:
                world = bpy.data.worlds.new("World")
                bpy.context.scene.world = world

            world.use_nodes = True
            bg_node = world.node_tree.nodes['Background']
            bg_node.inputs[0].default_value = (0.02, 0.02, 0.02, 1.0)  
            bg_node.inputs[1].default_value = 0.1  

            # Improved shadow and global illumination settings
            bpy.context.scene.eevee.use_soft_shadows = True
            bpy.context.scene.eevee.shadow_cube_size = '2048'
            bpy.context.scene.eevee.shadow_cascade_size = '2048'
            bpy.context.scene.eevee.use_shadow_high_bitdepth = True
            bpy.context.scene.eevee.use_gtao = True
            bpy.context.scene.eevee.gtao_distance = 0.5
            bpy.context.scene.eevee.gtao_factor = 1.0

            # Set render performance options optimized for shadows
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.preview_denoiser = 'OPTIX'
            bpy.context.scene.cycles.denoiser = 'OPTIX'

            # Set GPU memory optimization
            bpy.context.scene.cycles.use_fast_gi = True
            bpy.context.scene.cycles.ao_bounces = 4  

            # Improve shadow quality in Cycles
            bpy.context.scene.cycles.shadow_samples = 8  
            bpy.context.scene.cycles.use_adaptive_sampling = True
            bpy.context.scene.cycles.adaptive_threshold = 0.005  
            bpy.context.scene.cycles.adaptive_min_samples = 16

            # Set render boundary box to improve performance
            for obj in blender_objects:
                obj.cycles.use_adaptive_subdivision = True

            # Optimize shadow settings for Cycles
            bpy.context.scene.cycles.caustics_reflective = True
            bpy.context.scene.cycles.caustics_refractive = True

            bpy.context.scene.cycles.max_bounces = 4
            bpy.context.scene.cycles.diffuse_bounces = 2
            bpy.context.scene.cycles.glossy_bounces = 2
            bpy.context.scene.cycles.transmission_bounces = 2
            bpy.context.scene.cycles.volume_bounces = 0
            bpy.context.scene.cycles.transparent_max_bounces = 4

            bpy.context.scene.cycles.sample_clamp_direct = 5.0  
            bpy.context.scene.cycles.sample_clamp_indirect = 3.0  

            # Increase sampling for shadow detail
            # bpy.context.scene.cycles.samples = 256  
            bpy.context.scene.cycles.samples = 64
            bpy.context.scene.cycles.use_adaptive_sampling = True
            bpy.context.scene.cycles.adaptive_threshold = 0.02

            # Add ambient occlusion to table for better shadow contact
            if 'Table' in bpy.data.objects:
                table = bpy.data.objects['Table']
                # Create a new material slot if needed
                if not table.data.materials:
                    table_mat = bpy.data.materials.new(name="TableMaterial")
                    table.data.materials.append(table_mat)
                else:
                    table_mat = table.data.materials[0]
                
                # Enable nodes for the material
                table_mat.use_nodes = True
                nodes = table_mat.node_tree.nodes
                links = table_mat.node_tree.links
                
                # Create AO node for the table material
                ao_node = nodes.new(type='ShaderNodeAmbientOcclusion')
                ao_node.inputs['Distance'].default_value = 0.8  
                ao_node.inputs['Color'].default_value = (0.1, 0.1, 0.1, 1.0)  
                
                # Connect AO to BSDF
                bsdf = nodes.get('Principled BSDF')
                links.new(ao_node.outputs['AO'], bsdf.inputs['Base Color'])

            # Setup complete. Ready to render animation with enhanced shadows.
            # Output frames will be saved to: {bpy.context.scene.render.filepath}


            # For each frame, check if objects are in camera view
            for frame_data in sim_data['motion_trajectory']:
                frame_idx = frame_data['frame_id']
                bpy.context.scene.frame_set(frame_idx)
                
                # Update inside_camera_view for each object
                for obj_transform in frame_data['objects']:
                    obj_idx = obj_transform['object_id']
                    if obj_idx < len(blender_objects):
                        # Get the object's current state at this frame
                        bpy.context.view_layer.update()
                        
                        # Check if object is in camera view
                        in_camera_view = is_in_camera_view(blender_objects[obj_idx], bpy.context.scene, camera)
                        
                        # Update the simulation data
                        obj_transform['inside_camera_view'] = in_camera_view

            # Save the updated simulation data back to the file
            with open(data_path, 'w') as f:
                json.dump(sim_data, f, indent=2)
            
            # Updated simulation data with inside_camera_view information: {data_path}


            # Render animation
            # bpy.ops.render.render(animation=True)
            # Render animation with progress bar
            total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1


            with tqdm(total=total_frames, desc="progress", unit="frame") as pbar:

                bpy.context.scene.render.filepath = os.path.join(frames_dir, "frame_")
                bpy.ops.render.render(animation=True)

            print("The program is completed!")
            


       
