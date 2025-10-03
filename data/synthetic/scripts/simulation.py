# simulation
import pybullet as p
import pybullet_data
import numpy as np
import random
import os
import json
import argparse
from collections import namedtuple

Shape = namedtuple('Shape', ['name', 'function', 'scale'])
SHAPES = [
    Shape('cube', p.GEOM_BOX, [0.167, 0.167, 0.167]),
    Shape('sphere', p.GEOM_SPHERE, [0.167]),
    Shape('cylinder', p.GEOM_CYLINDER, [0.167, 0.167])
]

MATERIALS = ['metal', 'rubber']
COLORS = {
    'gray': [0.5, 0.5, 0.5, 1],
    'red': [0.8, 0.1, 0.1, 1],
    'blue': [0.1, 0.1, 0.8, 1],
    'green': [0.1, 0.8, 0.1, 1],
    'brown': [0.6, 0.4, 0.2, 1],
    'cyan': [0.1, 0.8, 0.8, 1],
    'purple': [0.8, 0.1, 0.8, 1],
    'yellow': [0.8, 0.8, 0.1, 1]
}

def get_visible_area():
    return {
        'left': -5,
        'right': 5,
        'front': -3,
        'back': 7
    }

def get_initial_height(shape, scale):
    if shape.name == 'cube':
        return scale[2]
    elif shape.name == 'sphere':
        return scale[0]
    elif shape.name == 'cylinder':
        return scale[1]
    return 0.5

class Object:
    def __init__(self, shape_idx, material_idx, color_name, position, velocity, object_id):
        self.shape = SHAPES[shape_idx]
        self.material = MATERIALS[material_idx]
        self.color_name = color_name
        self.color = COLORS[color_name]
        height = get_initial_height(self.shape, self.shape.scale)
        self.position = [position[0], position[1], height]
        self.velocity = velocity
        self.object_id = object_id
        self.id = None
        self.create()

    def create(self):
        if self.shape.name == 'cube':
            visual_shape_id = p.createVisualShape(
                shapeType=self.shape.function,
                halfExtents=self.shape.scale,
                rgbaColor=self.color
            )
            collision_shape_id = p.createCollisionShape(
                shapeType=self.shape.function,
                halfExtents=self.shape.scale
            )
        elif self.shape.name == 'sphere':
            visual_shape_id = p.createVisualShape(
                shapeType=self.shape.function,
                radius=self.shape.scale[0],
                rgbaColor=self.color
            )
            collision_shape_id = p.createCollisionShape(
                shapeType=self.shape.function,
                radius=self.shape.scale[0]
            )
        else:  # cylinder
            visual_shape_id = p.createVisualShape(
                shapeType=self.shape.function,
                radius=self.shape.scale[0],
                length=self.shape.scale[1] * 2,
                rgbaColor=self.color
            )
            collision_shape_id = p.createCollisionShape(
                shapeType=self.shape.function,
                radius=self.shape.scale[0],
                height=self.shape.scale[1] * 2
            )
        
        mass = 1.0
        restitution = 0.7 if self.material == 'rubber' else 0.5
        friction = 0.8 if self.material == 'rubber' else 0.4
        
        self.id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.position
        )
        
        p.changeDynamics(
            self.id, -1,
            restitution=restitution,
            lateralFriction=friction,
            linearDamping=0.1,
            angularDamping=0.1,
            spinningFriction=0.05,
            rollingFriction=0.05
        )
        
        p.resetBaseVelocity(
            self.id,
            linearVelocity=self.velocity
        )

    def to_dict(self):
        return {
            'object_id': self.object_id,
            'color': self.color_name,
            'material': self.material,
            'shape': self.shape.name
        }

def generate_unique_objects(num_objects, extra_static_objects=0, extra_moving_objects=0, without_cube=False, no_rubber_position=None):    
    """Generate unique object attribute conbinations
    """
    objects_props = []
    no_rubber_position = no_rubber_position or -1
    
    rubber_idx = MATERIALS.index('rubber')
    
    for i in range(num_objects):
        while True:
            if without_cube:
                shape_idx = random.randint(1, len(SHAPES) - 1)
            else:
                shape_idx = random.randint(0, len(SHAPES) - 1)
            
            if i == no_rubber_position:
                available_materials = [idx for idx in range(len(MATERIALS)) if idx != rubber_idx]
                material_idx = random.choice(available_materials)
            else:
                material_idx = random.randint(0, len(MATERIALS) - 1)
            
            color_name = random.choice(list(COLORS.keys()))
            props = (shape_idx, material_idx, color_name)
            
            if props not in objects_props:
                objects_props.append(props)
                break
    
    while len(objects_props) < num_objects + extra_static_objects + extra_moving_objects:
        shape_idx = random.randint(0, len(SHAPES) - 1)
        material_idx = random.randint(0, len(MATERIALS) - 1)
        color_name = random.choice(list(COLORS.keys()))
        props = (shape_idx, material_idx, color_name)
        if props not in objects_props:
            objects_props.append(props)
    
    return objects_props


def generate_non_overlapping_position(objects, visible_area, min_distance=1.0, max_attempts=1000):
    """Generate a position that doesn't overlap with existing objects."""
    for attempt in range(max_attempts):
        pos = [
            random.uniform(visible_area['left'], visible_area['right']),
            random.uniform(visible_area['front'], visible_area['back']),
            0
        ]
        
        # Check if position overlaps with any existing object
        overlaps = False
        for obj in objects:
            distance_sq = (pos[0] - obj.position[0])**2 + (pos[1] - obj.position[1])**2
            if distance_sq <= min_distance**2:
                overlaps = True
                break
        
        if not overlaps:
            return pos
    
    # If we can't find a non-overlapping position after max_attempts
    raise RuntimeError(f"Could not find non-overlapping position after {max_attempts} attempts")

def generate_overdetermination(num_objects=3, extra_static_objects=0, extra_moving_objects=0):
    objects = []
    visible_area = get_visible_area()
    props = generate_unique_objects(num_objects,extra_static_objects, extra_moving_objects, without_cube=True)

    # Static object (S)
    margin = 2.0
    static_pos = [
        random.uniform(visible_area['left'] + margin, visible_area['right'] - margin),
        random.uniform(visible_area['front'] + margin, visible_area['back'] - margin),
        0
    ]
    static_vel = [0, 0, 0]
    
    objects.append(Object(props[0][0], props[0][1], props[0][2], static_pos, static_vel, 0))

    # Moving objects (M1, M2)
    radius = random.uniform(4.0, 8.0)
    speed = random.uniform(12.0, 16.0)
    base_angle = random.uniform(0, 2 * np.pi)
    angle_between = np.radians(random.uniform(65, 85))
    
    angles = [base_angle, base_angle + angle_between]
    
    for i in range(2):
        moving_pos = [
            static_pos[0] + radius * np.cos(angles[i]),
            static_pos[1] + radius * np.sin(angles[i]),
            0
        ]       
        direction = [
            static_pos[0] - moving_pos[0],
            static_pos[1] - moving_pos[1],
            0
        ]
        norm = np.sqrt(direction[0]**2 + direction[1]**2)
        moving_vel = [
            direction[0]/norm * speed,
            direction[1]/norm * speed,
            0
        ]
        
        shape_idx, material_idx, color_name = props[i+1]
        objects.append(Object(shape_idx, material_idx, color_name, 
                              moving_pos, moving_vel, i+1))

    # Add extra static objects based on the count
    if extra_static_objects > 0:
        num_extra = min(extra_static_objects, 2)
        for i in range(num_extra):
            try:
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                extra_vel = [0, 0, 0]  

                prop_idx = num_objects + i 
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, prop_idx))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break
    
    # Add extra moving objects 
    if extra_moving_objects > 0:
        num_extra_moving = min(extra_moving_objects, 3)  
        for i in range(num_extra_moving):
            try:
                extra_moving_pos = generate_non_overlapping_position(objects, visible_area, min_distance=2.0)
                
                random_angle = random.uniform(0, 2 * np.pi)
                distractor_speed = random.uniform(8.0, 14.0) 
                
                extra_moving_vel = [
                    distractor_speed * np.cos(random_angle),
                    distractor_speed * np.sin(random_angle),
                    0
                ]
                
                prop_idx = num_objects + extra_static_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    object_id = num_objects + extra_static_objects + i
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_moving_pos, extra_moving_vel, object_id))
            except RuntimeError as e:
                print(f"Warning: Could not place extra moving object {i}: {e}")
                break 
    
    return objects

def generate_switch(num_objects=3, extra_static_objects=0, extra_moving_objects=0):
    """
    static object: S
    moving objects: M1(center, low speed), M2(high speed)
    """
    objects = []
    visible_area = get_visible_area()
    props = generate_unique_objects(num_objects, extra_static_objects, extra_moving_objects, without_cube = True, no_rubber_position=1)
    
    # creating static object (S)- near the center 
    margin = 2.0
    static_pos = [
        random.uniform(visible_area['left'] + margin, visible_area['right'] - margin),
        random.uniform(visible_area['front'] + margin, visible_area['back'] - margin),
        0
    ]
    static_vel = [0, 0, 0]
    objects.append(Object(props[0][0], props[0][1], props[0][2], static_pos, static_vel, 0))

    while True:
        direction_angle = random.uniform(0, 2 * np.pi)
        if (abs(direction_angle - np.pi/2) > np.pi/8 and 
            abs(direction_angle - 3*np.pi/2) > np.pi/8):
            break
    direction_unit = [np.cos(direction_angle), np.sin(direction_angle), 0]
    
    m2_distance = random.uniform(5.0, 6.0)  
    m1_distance = random.uniform(3.0, 4.0)  
    
    m1_speed = random.uniform(6.0, 9.0)  
    m2_speed = random.uniform(21.0, 24.0)   
    
    # crateing moving object M1
    m1_pos = [
        static_pos[0] - direction_unit[0] * m1_distance,
        static_pos[1] - direction_unit[1] * m1_distance,
        0
    ]
    m1_vel = [
        direction_unit[0] * m1_speed,
        direction_unit[1] * m1_speed,
        0
    ]
    objects.append(Object(props[1][0], props[1][1], props[1][2], m1_pos, m1_vel, 1))

    # creating moving object M2
    m2_pos = [
        m1_pos[0] - direction_unit[0] * m2_distance,
        m1_pos[1] - direction_unit[1] * m2_distance,
        0
    ]
    m2_vel = [
        direction_unit[0] * m2_speed,
        direction_unit[1] * m2_speed,
        0
    ]
    objects.append(Object(props[2][0], props[2][1], props[2][2], m2_pos, m2_vel, 2))
    
    # Add extra static objects based on the count
    if extra_static_objects > 0:
        num_extra = 1 if extra_static_objects == 1 else 2
        for i in range(num_extra):
            try:
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                extra_vel = [0, 0, 0]
                prop_idx = num_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, prop_idx))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break  # Stop trying to add more objects if we can't find space

    # Add extra moving objects
    if extra_moving_objects > 0:
        num_extra_moving = 1 if extra_moving_objects == 1 else 2
        for i in range(num_extra_moving):
            try:
                extra_direction_angle = random.uniform(0, 2 * np.pi)
                extra_direction_unit = [np.cos(extra_direction_angle), np.sin(extra_direction_angle), 0]
                extra_speed = random.uniform(5.0, 25.0)
                
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                
                extra_vel = [
                    extra_direction_unit[0] * extra_speed,
                    extra_direction_unit[1] * extra_speed,
                    0
                ]
                
                prop_idx = num_objects + extra_static_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    object_id = num_objects + extra_static_objects + i
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, object_id))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break 

    return objects


def generate_late(num_objects=3, extra_static_objects=0, extra_moving_objects=0):
    """
    static object:S
    moving objects:M1,M2
    collision:M1 with S 
    """
    objects = []
    visible_area = get_visible_area()
    props = generate_unique_objects(num_objects, extra_static_objects, extra_moving_objects)

    # creating static object (S)- near the center
    margin = 2.0
    static_pos = [
        random.uniform(visible_area['left'] + margin, visible_area['right'] - margin),
        random.uniform(visible_area['front'] + margin, visible_area['back'] - margin),
        0
    ]
    static_vel = [0, 0, 0]
    objects.append(Object(props[0][0], props[0][1], props[0][2], static_pos, static_vel, 0))

    # Moving object M1
    base_radius_M1 = random.uniform(4.0, 12.0)
    speed_M1 = random.uniform(12.0, 24.0)
    base_angle_M1 = random.uniform(0, 2 * np.pi)

    #Moving ovjext M2
    base_radius_M2 = random.uniform(4.0, 12.0)
    spped_M2 = random.uniform(12.0, 24.0)
    base_angle_M2 = base_angle_M1+np.radians(random.uniform(45, 135))

    
    radiuses = [base_radius_M1, base_radius_M2]
    speed = [speed_M1, spped_M2]
    angles = [base_angle_M1, base_angle_M2]
    
    for i in range(2):
        moving_pos = [
            static_pos[0] + radiuses[i] * np.cos(angles[i]),
            static_pos[1] + radiuses[i] * np.sin(angles[i]),
            0
        ]
        
        direction = [
            static_pos[0] - moving_pos[0],
            static_pos[1] - moving_pos[1],
            0
        ]
        norm = np.sqrt(direction[0]**2 + direction[1]**2)
        moving_vel = [
            direction[0]/norm * speed[i],
            direction[1]/norm * speed[i],
            0
        ]
        
        objects.append(Object(props[i+1][0], props[i+1][1], props[i+1][2], 
                              moving_pos, moving_vel, i+1))

    # Add extra static objects based on the count
    if extra_static_objects > 0:
        num_extra = 1 if extra_static_objects == 1 else 2
        for i in range(num_extra):
            try:
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                extra_vel = [0, 0, 0]
                prop_idx = num_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, prop_idx))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break
    
    # Add extra moving objects
    if extra_moving_objects > 0:
        num_extra_moving = 1 if extra_moving_objects == 1 else 2
        for i in range(num_extra_moving):
            try:
                extra_direction_angle = random.uniform(0, 2 * np.pi)
                extra_direction_unit = [np.cos(extra_direction_angle), np.sin(extra_direction_angle), 0]
                extra_speed = random.uniform(5.0, 25.0)
                
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                
                extra_vel = [
                    extra_direction_unit[0] * extra_speed,
                    extra_direction_unit[1] * extra_speed,
                    0
                ]
                
                prop_idx = num_objects + extra_static_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    object_id = num_objects + extra_static_objects + i
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, object_id))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break
    
    return objects

def generate_early(num_objects=4, extra_static_objects=0, extra_moving_objects=0):
    """
    static objects:S1, S2(center)
    moving objects: M1, M2
    collision: {0,2} {1,3} 
    """
    objects = []
    visible_area = get_visible_area()
    props = generate_unique_objects(num_objects, extra_static_objects, extra_moving_objects)
    
    # Static objects
    center_x = (visible_area['left'] + visible_area['right']) / 2
    center_y = (visible_area['front'] + visible_area['back']) / 2
    
    s1_s2_distance = random.uniform(2.0, 4.0)
    alignment_angle = random.uniform(0, 2 * np.pi)
    
    offset_range = 0.3  
    s2_pos = [
        center_x + random.uniform(-offset_range, offset_range),
        center_y + random.uniform(-offset_range, offset_range),
        0
    ]
    s2_vel = [0, 0, 0]
    s1_pos = [
        s2_pos[0] + s1_s2_distance * np.cos(alignment_angle),
        s2_pos[1] + s1_s2_distance * np.sin(alignment_angle),
        0
    ]
    s1_vel = [0, 0, 0]
    
    # create static objects S1 S2
    objects.append(Object(props[0][0], props[0][1], props[0][2], s1_pos, s1_vel, 0))
    objects.append(Object(props[1][0], props[1][1], props[1][2], s2_pos, s2_vel, 1))
    
    m1_angle_offset = random.uniform(np.pi/3, 2*np.pi/3) * random.choice([-1, 1]) 
    m1_angle = alignment_angle + m1_angle_offset

    m1_direction = [np.cos(m1_angle), np.sin(m1_angle)]
    m1_distance = random.uniform(2.0, 5.0) 
    m1_speed = random.uniform(10.0, 25.0)   

    m1_pos = [
        s1_pos[0] + m1_distance * m1_direction[0],
        s1_pos[1] + m1_distance * m1_direction[1],
        0
    ]
    m1_vel = [
        -m1_direction[0] * m1_speed,
        -m1_direction[1] * m1_speed,
        0
    ]

    m2_direction = [-np.cos(alignment_angle), -np.sin(alignment_angle)]
    m2_distance = random.uniform(6.0, 12.0)  

    m2_initial_speed = random.uniform(16.0, 30.0)  
       
    m2_pos = [
        s2_pos[0] + m2_distance * m2_direction[0],
        s2_pos[1] + m2_distance * m2_direction[1],
        0
    ]
    m2_vel = [
        -m2_direction[0] * m2_initial_speed,
        -m2_direction[1] * m2_initial_speed,
        0
    ]
    
    # create moving ovjects M1 M2
    objects.append(Object(props[2][0], props[2][1], props[2][2], m1_pos, m1_vel, 2))
    objects.append(Object(props[3][0], props[3][1], props[3][2], m2_pos, m2_vel, 3))
    
    # Add extra static objects based on the count
    if extra_static_objects > 0:
        num_extra = 1 if extra_static_objects == 1 else 2
        for i in range(num_extra):
            try:
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                extra_vel = [0, 0, 0]
                prop_idx = num_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, prop_idx))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break
    
    # Add extra moving objects
    if extra_moving_objects > 0:
        num_extra_moving = 1 if extra_moving_objects == 1 else 2
        for i in range(num_extra_moving):
            try:
                extra_direction_angle = random.uniform(0, 2 * np.pi)
                extra_direction_unit = [np.cos(extra_direction_angle), np.sin(extra_direction_angle), 0]
                extra_speed = random.uniform(5.0, 25.0)
                
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                
                extra_vel = [
                    extra_direction_unit[0] * extra_speed,
                    extra_direction_unit[1] * extra_speed,
                    0
                ]
                
                prop_idx = num_objects + extra_static_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    object_id = num_objects + extra_static_objects + i
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, object_id))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break 
    
    return objects

def generate_double(num_objects=4, extra_static_objects=0, extra_moving_objects=0):  # [s, m1, m2, m3]
    """
    static objects: S
    moving objects: M1 M2 M3
    M1 towards S, M2 hinder M1, M3 hinder M2
    collisions: {2,3}, {0,1}
    """
    objects = []
    visible_area = get_visible_area()
    props = generate_unique_objects(num_objects, extra_static_objects, extra_moving_objects)
    
    # Static object S 
    center_x = (visible_area['left'] + visible_area['right']) / 2
    center_y = (visible_area['front'] + visible_area['back']) / 2
    offset_range = 0.5
    s_pos = [
        center_x + random.uniform(-offset_range, offset_range),
        center_y + random.uniform(-offset_range, offset_range),
        0
    ]
    s_vel = [0, 0, 0]
    
    objects.append(Object(props[0][0], props[0][1], props[0][2], s_pos, s_vel, 0))
    
    m1_distance = random.uniform(7.0, 11.0)  
    m1_speed = random.uniform(12.0, 16.0)   
    
    m1_angle = random.uniform(0, 2 * np.pi)
    m1_direction = np.array([np.cos(m1_angle), np.sin(m1_angle)])
    
    m1_pos = s_pos[:2] - m1_distance * m1_direction
    m1_pos = np.append(m1_pos, 0)
    m1_vel = np.append(m1_direction * m1_speed, 0)
    
    friction_factor = 0.85 
    m1_to_s_time = m1_distance / (m1_speed * friction_factor)
    
    m2_intercept_fraction_m1 = 0.6
    m2_intercept_time = m1_to_s_time * m2_intercept_fraction_m1
    m2_intercept_point_m1 = m1_pos[:2] + (m1_vel[:2] * m1_to_s_time * m2_intercept_fraction_m1)

    m2_angle = m1_angle + np.pi / 2
    m2_direction = np.array([np.cos(m2_angle), np.sin(m2_angle)])
    
    m2_distance = random.uniform(6.0, 10.0)

    m2_speed = m2_distance / (m2_intercept_time * 0.8) 
    m2_speed = np.clip(m2_speed, 12.0, 18.0)  

    m2_pos = m2_intercept_point_m1 - m2_distance * m2_direction
    m2_pos = np.append(m2_pos, 0)
    m2_vel = np.append(m2_direction * m2_speed, 0)

    m3_intercept_fraction_m2 = random.uniform(0.5, 0.7)
    m2_time_to_intercept = (m2_distance * m3_intercept_fraction_m2) / (m2_speed * friction_factor)
    m3_intercept_point_m2 = m2_pos[:2] + m2_direction * (m2_distance * m3_intercept_fraction_m2)

    m3_angle = m2_angle + np.radians(random.uniform(110, 160))
    m3_direction = np.array([np.cos(m3_angle), np.sin(m3_angle)])
    m3_distance = random.uniform(6.0, 10.0)  
    m3_speed = m3_distance / (m2_time_to_intercept * 0.8)
    m3_speed = np.clip(m3_speed, 10.0, 25.0)  
    m3_pos = m3_intercept_point_m2 - m3_distance * m3_direction
    m3_pos = np.append(m3_pos, 0)
    m3_vel = np.append(m3_direction * m3_speed, 0)

    # creating moving objects M1 M2 M3
    objects.append(Object(props[1][0], props[1][1], props[1][2], m1_pos.tolist(), m1_vel.tolist(), 1))
    objects.append(Object(props[2][0], props[2][1], props[2][2], m2_pos.tolist(), m2_vel.tolist(), 2))
    objects.append(Object(props[3][0], props[3][1], props[3][2], m3_pos.tolist(), m3_vel.tolist(), 3))
    
    # Add extra static objects based on the count
    if extra_static_objects > 0:
        num_extra = 1 if extra_static_objects == 1 else 2
        for i in range(num_extra):
            try:
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                # Create extra static object with zero velocity
                extra_vel = [0, 0, 0]
                # Use the last props for extra objects (assuming props has enough elements)
                prop_idx = num_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, prop_idx))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break  # Stop trying to add more objects if we can't find space
    
    # Add extra moving objects
    if extra_moving_objects > 0:
        num_extra_moving = 1 if extra_moving_objects == 1 else 2
        for i in range(num_extra_moving):
            try:
                # Generate random movement direction and speed for extra moving objects
                extra_direction_angle = random.uniform(0, 2 * np.pi)
                extra_direction_unit = [np.cos(extra_direction_angle), np.sin(extra_direction_angle), 0]
                extra_speed = random.uniform(5.0, 25.0)
                
                # Generate position that doesn't overlap with existing objects
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                
                # Set velocity for extra moving object
                extra_vel = [
                    extra_direction_unit[0] * extra_speed,
                    extra_direction_unit[1] * extra_speed,
                    0
                ]
                
                # Use props for extra moving objects
                prop_idx = num_objects + extra_static_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    object_id = num_objects + extra_static_objects + i
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, object_id))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break  # Stop trying to add more objects if we can't find space
    
    return objects

def generate_bogus(num_objects=4, extra_static_objects=0, extra_moving_objects=0):  
    objects = []
    visible_area = get_visible_area()
    props = generate_unique_objects(num_objects, extra_static_objects, extra_moving_objects)
    
    # Static objects
    center_x = (visible_area['left'] + visible_area['right']) / 2
    center_y = (visible_area['front'] + visible_area['back']) / 2
    
    s1_s2_distance = random.uniform(2.0, 3.0) 
    alignment_angle = random.uniform(0, 2 * np.pi)
    
    offset_range = 0.3  
    s2_pos = [
        center_x + random.uniform(-offset_range, offset_range),
        center_y + random.uniform(-offset_range, offset_range),
        0
    ]
    s2_vel = [0, 0, 0]
    
    s1_pos = [
        s2_pos[0] + s1_s2_distance * np.cos(alignment_angle),
        s2_pos[1] + s1_s2_distance * np.sin(alignment_angle),
        0
    ]
    s1_vel = [0, 0, 0]
    
    # creating static objects S1 S2
    objects.append(Object(props[0][0], props[0][1], props[0][2], s1_pos, s1_vel, 0)) # S1
    objects.append(Object(props[1][0], props[1][1], props[1][2], s2_pos, s2_vel, 1)) #S2
    
    m2_direction = [np.cos(alignment_angle), np.sin(alignment_angle)]
    
    m2_distance = random.uniform(4.0, 8.0)  
    m2_initial_speed = random.uniform(2.0, 10.0)  
    
    m2_pos = [
        s2_pos[0] - m2_distance * m2_direction[0],
        s2_pos[1] - m2_distance * m2_direction[1],
        0
    ]
    m2_vel = [
        m2_direction[0] * m2_initial_speed,
        m2_direction[1] * m2_initial_speed,
        0
    ]
    
    m1_angle_offset = random.uniform(np.pi/3, 2*np.pi/3) 
    m1_angle = alignment_angle + m1_angle_offset
    m1_direction = [-np.cos(m1_angle), -np.sin(m1_angle)]
    
    m1_distance = random.uniform(4.0, 10.0)  
    m1_speed = random.uniform(20.0, 30.0) 
    
    m1_pos = [
        s2_pos[0] - m1_distance * m1_direction[0],
        s2_pos[1] - m1_distance * m1_direction[1],
        0
    ]
    m1_vel = [
        m1_direction[0] * m1_speed,
        m1_direction[1] * m1_speed,
        0
    ]
    
    objects.append(Object(props[2][0], props[2][1], props[2][2], m1_pos, m1_vel, 2))
    objects.append(Object(props[3][0], props[3][1], props[3][2], m2_pos, m2_vel, 3))
    # Add extra static objects based on the count
    if extra_static_objects > 0:
        num_extra = 1 if extra_static_objects == 1 else 2
        for i in range(num_extra):
            try:
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                # Create extra static object with zero velocity
                extra_vel = [0, 0, 0]
                # Use the last props for extra objects (assuming props has enough elements)
                prop_idx = num_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, prop_idx))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break  # Stop trying to add more objects if we can't find space
    
    # Add extra moving objects
    if extra_moving_objects > 0:
        num_extra_moving = 1 if extra_moving_objects == 1 else 2
        for i in range(num_extra_moving):
            try:
                # Generate random movement direction and speed for extra moving objects
                extra_direction_angle = random.uniform(0, 2 * np.pi)
                extra_direction_unit = [np.cos(extra_direction_angle), np.sin(extra_direction_angle), 0]
                extra_speed = random.uniform(5.0, 25.0)
                
                # Generate position that doesn't overlap with existing objects
                extra_pos = generate_non_overlapping_position(objects, visible_area)
                
                # Set velocity for extra moving object
                extra_vel = [
                    extra_direction_unit[0] * extra_speed,
                    extra_direction_unit[1] * extra_speed,
                    0
                ]
                
                # Use props for extra moving objects
                prop_idx = num_objects + extra_static_objects + i
                if prop_idx < len(props):
                    shape_idx, material_idx, color_name = props[prop_idx]
                    object_id = num_objects + extra_static_objects + i
                    objects.append(Object(shape_idx, material_idx, color_name, 
                                        extra_pos, extra_vel, object_id))
            except RuntimeError as e:
                print(f"Warning: {e}")
                break 
    
    return objects

def generate_short(num_objects=4):  # [s1, s2, m1, m2]
    objects = []
    visible_area = get_visible_area()
    props = generate_unique_objects(4)
    
    # Static objects
    center_x = (visible_area['left'] + visible_area['right']) / 2
    center_y = (visible_area['front'] + visible_area['back']) / 2
    
    offset_range = 0.3
    s2_pos = [
        center_x + random.uniform(-offset_range, offset_range),
        center_y + random.uniform(-offset_range, offset_range),
        0
    ]
    s2_vel = [0, 0, 0]
    
    alignment_angle = random.uniform(0, 2 * np.pi)
    direction_vector = [np.cos(alignment_angle), np.sin(alignment_angle)]
    
    s1_distance = random.uniform(3.0, 5.0)
    s1_pos = [
        s2_pos[0] + s1_distance * direction_vector[0],
        s2_pos[1] + s1_distance * direction_vector[1],
        0
    ]
    s1_vel = [0, 0, 0]
    
    m2_distance = random.uniform(8.0, 12.0)
    m2_speed = random.uniform(15.0, 25.0)
    m2_pos = [
        s2_pos[0] - m2_distance * direction_vector[0],
        s2_pos[1] - m2_distance * direction_vector[1],
        0
    ]
    m2_vel = [
        direction_vector[0] * m2_speed,
        direction_vector[1] * m2_speed,
        0
    ]
    
    m1_angle_offset = random.uniform(np.pi/4 - np.pi/12, np.pi/4 + np.pi/12)  
    m1_approach_angle = alignment_angle + m1_angle_offset
    m1_direction = [np.cos(m1_approach_angle), np.sin(m1_approach_angle)]
    
    s1_s2_direction = direction_vector  
    
    target_offset_distance = random.uniform(0.2, 0.5)  
    target_point = [
        s2_pos[0] + target_offset_distance * s1_s2_direction[0],
        s2_pos[1] + target_offset_distance * s1_s2_direction[1]
    ]
    
    m1_distance = random.uniform(4.0, 7.0)
    m1_speed = random.uniform(20.0, 30.0)
    
    m1_pos = [
        target_point[0] - m1_distance * m1_direction[0],
        target_point[1] - m1_distance * m1_direction[1],
        0
    ]
    m1_vel = [
        m1_direction[0] * m1_speed,
        m1_direction[1] * m1_speed,
        0
    ]
    
    objects.append(Object(props[0][0], props[0][1], props[0][2], s1_pos, s1_vel, 0))  # S1
    objects.append(Object(props[1][0], props[1][1], props[1][2], s2_pos, s2_vel, 1))  # S2
    objects.append(Object(props[2][0], props[2][1], props[2][2], m1_pos, m1_vel, 2))  # M1
    objects.append(Object(props[3][0], props[3][1], props[3][2], m2_pos, m2_vel, 3))  # M2
    
    return objects

def detect_collisions(frame, objects, collision_pairs):
    collisions = []
    points = p.getContactPoints()
    
    # Create object ID mapping for CLEVRER format
    id_to_object_id = {obj.id: obj.object_id for obj in objects}
    
    current_frame_pairs = set()

    for point in points:
        obj1_id, obj2_id = point[1], point[2]
        
        if obj1_id == planeId or obj2_id == planeId:
            continue
        
        if obj1_id not in id_to_object_id or obj2_id not in id_to_object_id:
            continue
            
        if obj1_id > obj2_id:
            obj1_id, obj2_id = obj2_id, obj1_id
        
        pair = (obj1_id, obj2_id)
        
        if pair in current_frame_pairs:
            continue
        current_frame_pairs.add(pair)
        
        min_interval = 5  
        can_record = True
        
        for recorded_pair, recorded_frame in collision_pairs:
            if recorded_pair == pair and abs(frame - recorded_frame) < min_interval:
                can_record = False
                break
        
        if can_record:
            collision_pairs.append((pair, frame))
            
            contact_point = list(point[5])  
            
            object_ids = sorted([id_to_object_id[obj1_id], id_to_object_id[obj2_id]])
            
            collisions.append({
                'object_ids': object_ids,
                'frame_id': frame,
                'location': contact_point
            })
    
    return collisions


def parse_args():
    parser = argparse.ArgumentParser(description='Scene generator')
    parser.add_argument('--nsave', type=int, default=200,
                      help='Number of scenes to generate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--output_dir', type=str, default='simulation_output_v4',
                      help='Output file')
    parser.add_argument('--scene', type=str, default='all', help='Logical scene')
    #[overdetermination, switch, early, late, bogus, double, short, all]
    parser.add_argument('--extra_static_objects', type=int, default=0, help='Number of static distractors')
    parser.add_argument('--extra_moving_objects', type=int, default=0, help='Number of moving distractors')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    extra_static_objects = args.extra_static_objects
    extra_moving_objects = args.extra_moving_objects
    print("Program started")
    if args.scene == 'all':
        scene_list = ['overdetermination', 'early', 'late', 'bogus', 'double', 'switch']
    else:
        scene_list = [args.scene]

    for scene in scene_list:
        print(f"scene:{scene}")
        if extra_static_objects != 0:
            base_output_dir = f"{args.output_dir}_{extra_static_objects}_static_objects"
            output_dir = os.path.join(base_output_dir, scene)
        elif extra_moving_objects != 0:
            base_output_dir = f"{args.output_dir}_{extra_moving_objects}_moving_objects"
            output_dir = os.path.join(base_output_dir, scene)
        else:
            output_dir = os.path.join(args.output_dir, scene)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cnt = 1
        while cnt <= args.nsave:

            physicsClient = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            planeId = p.loadURDF("plane.urdf")
            p.changeDynamics(planeId, -1,
                restitution=0.3,
                lateralFriction=0.8,
                spinningFriction=0.1,
                rollingFriction=0.1
            )

            if scene == 'overdetermination':
                objects = generate_overdetermination(extra_static_objects=extra_static_objects, extra_moving_objects=extra_moving_objects)
            elif scene == 'switch':
                objects = generate_switch(extra_static_objects=extra_static_objects, extra_moving_objects=extra_moving_objects)
            elif scene == 'early':
                objects = generate_early(extra_static_objects=extra_static_objects, extra_moving_objects=extra_moving_objects)
            elif scene == 'late':
                objects = generate_late(extra_static_objects=extra_static_objects, extra_moving_objects=extra_moving_objects)
            elif scene == 'bogus':
                objects = generate_bogus(extra_static_objects=extra_static_objects, extra_moving_objects=extra_moving_objects)
            elif scene == 'double':
                objects = generate_double(extra_static_objects=extra_static_objects, extra_moving_objects=extra_moving_objects)
            else:
                raise ValueError(f"Unknown scene: {scene}")

            fps = 25
            duration = 5
            total_frames = fps * duration
            collision_pairs =[] 
            all_collisions = []
            motion_trajectory = []
            
            for _ in range(50):
                p.stepSimulation()

            for frame in range(total_frames):
                for _ in range(5):
                    p.stepSimulation() 
                frame_collisions = detect_collisions(frame, objects, collision_pairs)
                all_collisions.extend(frame_collisions)

                frame_data = {'frame_id': frame, 'objects': []}
                for obj in objects:
                    pos, orn = p.getBasePositionAndOrientation(obj.id)
                    lin_vel, ang_vel = p.getBaseVelocity(obj.id)
                    
                    frame_data['objects'].append({
                        'object_id': obj.object_id,
                        'location': list(pos),
                        'orientation': list(orn),
                        'velocity': list(lin_vel),
                        'angular_velocity': list(ang_vel)
                    })
                
                motion_trajectory.append(frame_data)
            p.disconnect()

            #check if we have the correct number of collisions(2)
            num_collisions = len(all_collisions)
            if scene == 'overdetermination':
                if num_collisions < 2:  
                    continue
                
                collision_objects = [set(c["object_ids"]) for c in all_collisions] 
                collision_frames = [c["frame_id"] for c in all_collisions]
                
                main_objects = {0, 1, 2}  
                has_invalid_collision = False
                
                for collision_set in collision_objects:
                    extra_objects_in_collision = collision_set - main_objects
                    main_objects_in_collision = collision_set & main_objects
                    
                    if extra_objects_in_collision and main_objects_in_collision:
                        has_invalid_collision = True
                        break
                
                if has_invalid_collision:
                    continue
                
                main_collisions = []
                main_collision_frames = []
                for i, collision_set in enumerate(collision_objects):
                    if collision_set.issubset(main_objects):
                        main_collisions.append(collision_set)
                        main_collision_frames.append(collision_frames[i])
                
                if len(main_collisions) != 2:
                    continue
                    
                if not((main_collisions[0] == {0, 1} and main_collisions[1] == {0, 2}) and len(set(main_collision_frames)) == 1):
                    continue
            elif scene == 'switch':
                if num_collisions != 2:
                    continue
                
                collision_pairs = [c["object_ids"] for c in all_collisions]
                collision_frames = [c["frame_id"] for c in all_collisions]
                    
                if not (collision_pairs[0] == [1, 2] and collision_pairs[1] == [0, 1] and collision_frames[1] - collision_frames[0] > 5):
                    continue
            elif scene == 'late':
                if num_collisions < 1: 
                    continue
                
                collision_objects = [set(c['object_ids']) for c in all_collisions]
                collision_frames = [c["frame_id"] for c in all_collisions]
                
                main_objects = {0, 1, 2}
                has_invalid_collision = False
                
                for collision_set in collision_objects:
                    extra_objects_in_collision = collision_set - main_objects
                    main_objects_in_collision = collision_set & main_objects
                    
                    if extra_objects_in_collision and main_objects_in_collision:
                        has_invalid_collision = True
                        break
                
                if has_invalid_collision:
                    continue
                
                main_collisions = []
                main_collision_frames = []
                for i, collision_set in enumerate(collision_objects):
                    if collision_set.issubset(main_objects):
                        main_collisions.append(collision_set)
                        main_collision_frames.append(collision_frames[i])
                
                if len(main_collisions) != 1:
                    continue
                    
                if not ({0, 1} in main_collisions or {0, 2} in main_collisions):
                    continue
                
                s_initial_pos = None
                for obj in objects:
                    if obj.object_id == 0:  
                        s_initial_pos = obj.position
                        break

                m1_can_reach = False
                m2_can_reach = False
                reach_threshold = 0.5  
                
                for frame_data in motion_trajectory:
                    for obj_data in frame_data['objects']:
                        if obj_data['object_id'] == 1:  # M1
                            m1_pos = obj_data['location']
                            distance_m1_to_s = ((m1_pos[0] - s_initial_pos[0])**2 + 
                                              (m1_pos[1] - s_initial_pos[1])**2 + 
                                              (m1_pos[2] - s_initial_pos[2])**2) ** 0.5
                            if distance_m1_to_s <= reach_threshold:
                                m1_can_reach = True
                        
                        elif obj_data['object_id'] == 2:  # M2
                            m2_pos = obj_data['location']
                            distance_m2_to_s = ((m2_pos[0] - s_initial_pos[0])**2 + 
                                              (m2_pos[1] - s_initial_pos[1])**2 + 
                                              (m2_pos[2] - s_initial_pos[2])**2) ** 0.5
                            if distance_m2_to_s <= reach_threshold:
                                m2_can_reach = True
                    
                    if m1_can_reach and m2_can_reach:
                        break
                
                if not (m1_can_reach and m2_can_reach):
                    continue
            elif scene == 'early':
                if num_collisions < 2: 
                    continue
                
                collision_objects = [set(c['object_ids']) for c in all_collisions]
                collision_frames = [c["frame_id"] for c in all_collisions]
                print(f"collision_objects:{collision_objects }")
                print(f"collision_frames:{collision_frames}")

                main_objects = {0, 1, 2, 3}  
                has_invalid_collision = False
                
                for collision_set in collision_objects:
                    extra_objects_in_collision = collision_set - main_objects
                    main_objects_in_collision = collision_set & main_objects
                    
                    if extra_objects_in_collision and main_objects_in_collision:
                        has_invalid_collision = True
                        break
                
                if has_invalid_collision:
                    continue
                
                main_collisions = []
                main_collision_frames = []
                for i, collision_set in enumerate(collision_objects):
                    if collision_set.issubset(main_objects):
                        main_collisions.append(collision_set)
                        main_collision_frames.append(collision_frames[i])
                
                if len(main_collisions) != 2:
                    continue
                    
                if not ({0, 2} in main_collisions and {1, 3} in main_collisions):
                    continue
                
                collision_02_frame = None
                collision_13_frame = None
                for i, collision_set in enumerate(main_collisions):
                    if collision_set == {0, 2}:
                        collision_02_frame = main_collision_frames[i]
                    elif collision_set == {1, 3}:
                        collision_13_frame = main_collision_frames[i]
                
                if not (collision_02_frame is not None and collision_13_frame is not None and 
                        collision_13_frame - collision_02_frame >= 5):
                    continue
                
                s1_initial_pos = None
                s2_initial_pos = None
                for obj in objects:
                    if obj.object_id == 0:  
                        s1_initial_pos = obj.position
                    elif obj.object_id == 1: 
                        s2_initial_pos = obj.position
                
                if s1_initial_pos is None or s2_initial_pos is None:
                    continue
                
                initial_s1_s2_distance = ((s1_initial_pos[0] - s2_initial_pos[0])**2 + 
                                         (s1_initial_pos[1] - s2_initial_pos[1])**2 + 
                                         (s1_initial_pos[2] - s2_initial_pos[2])**2) ** 0.5
                
                m2_s2_collision_frame = None
                for i, collision_set in enumerate(collision_objects):
                    if collision_set == {1, 3}:  
                        m2_s2_collision_frame = collision_frames[i]
                        break
                
                if m2_s2_collision_frame is None:
                    continue
                
                s2_final_pos = None
                final_frame_data = motion_trajectory[-1]  
                for obj_data in final_frame_data['objects']:
                    if obj_data['object_id'] == 1:  # S2
                        s2_final_pos = obj_data['location']
                        break
                
                if s2_final_pos is None:
                    continue
                
                s2_movement_distance = ((s2_final_pos[0] - s2_initial_pos[0])**2 + 
                                       (s2_final_pos[1] - s2_initial_pos[1])**2 + 
                                       (s2_final_pos[2] - s2_initial_pos[2])**2) ** 0.5
                
                if s2_movement_distance <= initial_s1_s2_distance:
                    print("S2 can't reach S1")
                    continue
            elif scene == 'double':
                if num_collisions < 2: 
                    continue
                
                collision_objects = [set(c['object_ids']) for c in all_collisions]
                collision_frames = [c["frame_id"] for c in all_collisions]
                
                main_objects = {0, 1, 2, 3}  
                has_invalid_collision = False
                
                for collision_set in collision_objects:
                    extra_objects_in_collision = collision_set - main_objects
                    main_objects_in_collision = collision_set & main_objects
                    
                    if extra_objects_in_collision and main_objects_in_collision:
                        has_invalid_collision = True
                        break
                
                if has_invalid_collision:
                    continue
                
                main_collisions = []
                main_collision_frames = []
                for i, collision_set in enumerate(collision_objects):
                    if collision_set.issubset(main_objects):
                        main_collisions.append(collision_set)
                        main_collision_frames.append(collision_frames[i])
                
                if len(main_collisions) != 2:
                    continue
                    
                print(f"collision_objects:{main_collisions}")
                print(f"collision_frames:{main_collision_frames}")
                if not (main_collisions[0] == {2, 3} and main_collisions[1] == {0, 1} and main_collision_frames[1] - main_collision_frames[0] > 30):
                    continue
            elif scene == 'bogus':
                if num_collisions < 1: 
                    continue
                
                collision_objects = [set(c['object_ids']) for c in all_collisions]
                collision_frames = [c["frame_id"] for c in all_collisions]
                
                main_objects = {0, 1, 2, 3} 
                has_invalid_collision = False
                
                for collision_set in collision_objects:
                    extra_objects_in_collision = collision_set - main_objects
                    main_objects_in_collision = collision_set & main_objects
                    
                    if extra_objects_in_collision and main_objects_in_collision:
                        has_invalid_collision = True
                        break
                
                if has_invalid_collision:
                    continue
                
                main_collisions = []
                main_collision_frames = []
                for i, collision_set in enumerate(collision_objects):
                    if collision_set.issubset(main_objects):
                        main_collisions.append(collision_set)
                        main_collision_frames.append(collision_frames[i])
                
                if len(main_collisions) != 1:
                    continue

                m2_final_pos = None
                s2_initial_pos = None
                
                for obj_data in motion_trajectory[-1]['objects']:
                    if obj_data['object_id'] == 3: 
                        m2_final_pos = obj_data['location']
                        break
                
                for obj in objects:
                    if obj.object_id == 1:  
                        s2_initial_pos = obj.position
                        break
                
                distance_to_s2 = None
                if m2_final_pos is not None and s2_initial_pos is not None:
                    distance_to_s2 = ((m2_final_pos[0] - s2_initial_pos[0])**2 + 
                                    (m2_final_pos[1] - s2_initial_pos[1])**2 + 
                                    (m2_final_pos[2] - s2_initial_pos[2])**2) ** 0.5
                print(f"collision_objects:{main_collisions}")
                print(f"collision_frames:{main_collision_frames}")
                if m2_final_pos is not None and s2_initial_pos is not None:
                    print(f"Distance of fi_M2 & In_S2: {distance_to_s2}")
                
                if not (main_collisions[0] == {1, 2} and main_collision_frames[0] > 20 and 
                        distance_to_s2 is not None and distance_to_s2 < 2.5): 
                    continue
            elif scene == 'short':
                if num_collisions != 2:
                    continue
                
                collision_objects = [set(c['object_ids']) for c in all_collisions]
                collision_frames = [c["frame_id"] for c in all_collisions]
                
                if not (collision_objects[0] == {1, 2}):
                    continue 
                s1_initial_pos = None
                s2_initial_pos = None
                s2_final_pos = None
                
                for obj in objects:
                    if obj.object_id == 0:  
                        s1_initial_pos = obj.position
                    elif obj.object_id == 1: 
                        s2_initial_pos = obj.position
                
                for obj_data in motion_trajectory[-1]['objects']:
                    if obj_data['object_id'] == 1: 
                        s2_final_pos = obj_data['location']
                        break
                
                distance_s2_to_s1 = None
                if s2_final_pos is not None and s1_initial_pos is not None:
                    distance_s2_to_s1 = ((s2_final_pos[0] - s1_initial_pos[0])**2 + 
                                        (s2_final_pos[1] - s1_initial_pos[1])**2 + 
                                        (s2_final_pos[2] - s1_initial_pos[2])**2) ** 0.5
                
                has_m1_s2_collision = any({1, 2} == collision_set for collision_set in collision_objects)
                has_m2_s2_collision = any({1, 3} == collision_set for collision_set in collision_objects)
                
                m1_s2_frame = None
                m2_s2_frame = None
                for i, collision_set in enumerate(collision_objects):
                    if collision_set == {1, 2}:
                        m1_s2_frame = collision_frames[i]
                    elif collision_set == {1, 3}:
                        m2_s2_frame = collision_frames[i]
                
                s2_velocity_after_m1_collision = None
                if m1_s2_frame is not None and m1_s2_frame < len(motion_trajectory) - 1:
                    for obj_data in motion_trajectory[m1_s2_frame + 1]['objects']:
                        if obj_data['object_id'] == 1:  
                            s2_velocity_after_m1_collision = obj_data['velocity']
                            break
                
                s1_s2_m2_direction = None
                if s1_initial_pos is not None and s2_initial_pos is not None:
                    s2_to_s1_direction = [
                        s1_initial_pos[0] - s2_initial_pos[0],
                        s1_initial_pos[1] - s2_initial_pos[1]
                    ]
                    s1_s2_m2_direction = [-s2_to_s1_direction[0], -s2_to_s1_direction[1]]
                    
                    magnitude = (s1_s2_m2_direction[0]**2 + s1_s2_m2_direction[1]**2) ** 0.5
                    if magnitude > 0:
                        s1_s2_m2_direction = [
                            s1_s2_m2_direction[0] / magnitude,
                            s1_s2_m2_direction[1] / magnitude
                        ]
                
                velocity_direction_valid = False
                if (s2_velocity_after_m1_collision is not None and 
                    s1_s2_m2_direction is not None and
                    len(s2_velocity_after_m1_collision) >= 2):
                    
                    s2_vel_magnitude = (s2_velocity_after_m1_collision[0]**2 + 
                                    s2_velocity_after_m1_collision[1]**2) ** 0.5
                    
                    if s2_vel_magnitude > 0.1: 
                        s2_vel_direction = [
                            s2_velocity_after_m1_collision[0] / s2_vel_magnitude,
                            s2_velocity_after_m1_collision[1] / s2_vel_magnitude
                        ]
                        
                        dot_product = (s2_vel_direction[0] * s1_s2_m2_direction[0] + 
                                    s2_vel_direction[1] * s1_s2_m2_direction[1])
                        
                        velocity_direction_valid = dot_product > 0.92
                
                print(f"collision_objects:{collision_objects}")
                print(f"collision_frames:{collision_frames}")
                if distance_s2_to_s1 is not None:
                    print(f"Distance of fi_S2 & In_S1: {distance_s2_to_s1}")
                print(f"S2 velocity after M1 collision: {s2_velocity_after_m1_collision}")
                print(f"S1-S2-M2 direction: {s1_s2_m2_direction}")
                print(f"Velocity direction valid: {velocity_direction_valid}")
                
                if not (has_m1_s2_collision and has_m2_s2_collision and
                        m1_s2_frame is not None and m2_s2_frame is not None and
                        m1_s2_frame < m2_s2_frame and  
                        m1_s2_frame > 20 and m2_s2_frame > 20 and  
                        distance_s2_to_s1 is not None and distance_s2_to_s1 > 2.0 and  
                        velocity_direction_valid):  
                    continue
            else:
                raise ValueError(f"Unknown scene: {scene}")
            scene_index = cnt - 1
            simulation_data = {
                'scene_index': scene_index,
                'video_filename': f'video_{scene_index:05d}.mp4',
                'object_property': [obj.to_dict() for obj in objects],
                'motion_trajectory': motion_trajectory,
                'collision': all_collisions
            }
            
            output_file = os.path.join(output_dir, f"annotation_{scene_index:05d}.json")
            with open(output_file, "w") as f:
                json.dump(simulation_data, f, indent=2)
                
            print(f"Generate simulation {cnt} / {args.nsave}")
            cnt += 1

    print("All scenes generation completed")