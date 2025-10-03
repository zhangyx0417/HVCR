import random
import numpy as np
from typing import List, Dict, Any
from base_generator import BaseScenarioQAGenerator, SettingType


# Double scenario QA generator
class DoubleScenarioQAGenerator(BaseScenarioQAGenerator):
    """
    QA generator for the Double scenario.
    M1 moves toward S, M2 moves toward M1 (with expected collision point), M3 moves toward M2.
    M3 hits M2, causing M2 to deviate from its trajectory, M1 finally hits S causing S to move.
    """
    
    def __init__(self):
        super().__init__('double')
    
    def analyze_scenario(self, data: Dict, setting: SettingType = None) -> Dict[str, Any]:
        """Analyze important information and events in the double scenario."""
        objects = data['object_property']
        trajectory = data['motion_trajectory']
        collisions = data['collision']
        
        analysis = {
            'M1': None,  # Object that finally hits S
            'M2': None,  # Object that was supposed to hit M1 but was deviated by M3
            'M3': None,  # Object that hits M2 causing it to deviate
            'S': None,   # Stationary target object
            'events': [],
            'collision_info': collisions,
            'collision_sequence': [],  # Collision sequence
            'trajectory_deviation': False,  # Whether M2 deviated from original trajectory
            'added_static': [],      # Added static objects
            'added_moving': []       # Added moving objects
        }
        
        # Identify objects based on setting type
        if setting:
            object_info = self._identify_objects_by_setting(trajectory, setting)
            analysis['added_static'] = object_info['added_static']
            analysis['added_moving'] = object_info['added_moving']
            
            # Use identified main objects
            if len(object_info['static_objects']) >= 1:
                analysis['S'] = object_info['static_objects'][0]
            if len(object_info['moving_objects']) >= 3:
                analysis['M1'] = object_info['moving_objects'][0]
                analysis['M2'] = object_info['moving_objects'][1]
                analysis['M3'] = object_info['moving_objects'][2]
        else:
            # Original object identification logic (backward compatible)
            # Find initial stationary object as S
            first_frame = trajectory[0]['objects']
            static_objects = []
            moving_objects = []
            
            for obj in first_frame:
                velocity_mag = self._calculate_velocity_magnitude(obj['velocity'])
                if velocity_mag < 0.01:  # Considered stationary
                    static_objects.append(obj['object_id'])
                else:
                    moving_objects.append(obj['object_id'])
            
            if len(static_objects) < 1 or len(moving_objects) < 3:
                return analysis
            
            analysis['S'] = static_objects[0]  # Assume only one stationary object
            
            # Analyze collision sequence to determine object roles
            collision_sequence = sorted(collisions, key=lambda x: x['frame_id'])
            analysis['collision_sequence'] = collision_sequence
            
            if len(collision_sequence) >= 2:
                # First collision: M3 hits M2
                first_collision = collision_sequence[0]
                # Second collision: M1 hits S
                second_collision = collision_sequence[1]
                
                # Determine M1 from second collision (object that collides with S)
                for obj_id in second_collision['object_ids']:
                    if obj_id != analysis['S']:
                        analysis['M1'] = obj_id
                        break
                
                # Determine M2 and M3 from first collision
                collision_objects = first_collision['object_ids']
                remaining_objects = [obj for obj in moving_objects if obj != analysis['M1']]
                
                # M2 and M3 are both in the first collision
                if len(collision_objects) == 2:
                    # Determine which is M2 (originally moving toward M1) and which is M3 through trajectory analysis
                    # Need to check which object was originally moving toward M1
                    m2_candidate = None
                    m3_candidate = None
                    
                    for obj_id in collision_objects:
                        if obj_id in remaining_objects:
                            # Check if this object was originally moving toward M1
                            if self._check_moving_towards_before_collision(trajectory, obj_id, analysis['M1'], first_collision['frame_id']):
                                m2_candidate = obj_id
                            else:
                                m3_candidate = obj_id
                    
                    # If found object moving toward M1, that's M2
                    if m2_candidate is not None:
                        analysis['M2'] = m2_candidate
                        analysis['M3'] = m3_candidate if m3_candidate is not None else collision_objects[1] if collision_objects[0] == m2_candidate else collision_objects[0]
                    else:
                        # If cannot determine, fall back to simple assignment
                        analysis['M2'] = collision_objects[0] if collision_objects[0] in remaining_objects else collision_objects[1]
                        analysis['M3'] = collision_objects[1] if collision_objects[0] == analysis['M2'] else collision_objects[0]
        
        # Analyze key events
        self._analyze_double_events(data, analysis)
        
        return analysis
    
    def _analyze_double_events(self, data: Dict, analysis: Dict):
        """Analyze key events in the double scenario."""
        trajectory = data['motion_trajectory']
        M1, M2, M3, S = analysis['M1'], analysis['M2'], analysis['M3'], analysis['S']
        
        events = []
        m2_trajectory_changed = False
        
        # Analyze each frame
        for i, frame in enumerate(trajectory):
            frame_objects = {obj['object_id']: obj for obj in frame['objects']}
            
            # Check M1 moving toward S
            if M1 is not None and S is not None:
                if M1 in frame_objects and S in frame_objects:
                    m1_obj = frame_objects[M1]
                    s_obj = frame_objects[S]
                    
                    if self._is_moving_towards(m1_obj['location'], m1_obj['velocity'],
                                             s_obj['location'], s_obj['velocity']):
                        events.append({
                            'type': 'moving_towards',
                            'frame': i,
                            'subject': M1,
                            'target': S,
                            'description': f"Object {M1} is moving towards object {S}"
                        })
            
            # Check M2 moving toward M1 (before being hit by M3)
            if M2 is not None and M1 is not None and not m2_trajectory_changed:
                if M2 in frame_objects and M1 in frame_objects:
                    m2_obj = frame_objects[M2]
                    m1_obj = frame_objects[M1]
                    
                    if self._is_moving_towards(m2_obj['location'], m2_obj['velocity'],
                                             m1_obj['location'], m1_obj['velocity']):
                        events.append({
                            'type': 'moving_towards',
                            'frame': i,
                            'subject': M2,
                            'target': M1,
                            'description': f"Object {M2} is moving towards object {M1}"
                        })
            
            # Check M3 moving toward M2
            if M3 is not None and M2 is not None:
                if M3 in frame_objects and M2 in frame_objects:
                    m3_obj = frame_objects[M3]
                    m2_obj = frame_objects[M2]
                    
                    if self._is_moving_towards(m3_obj['location'], m3_obj['velocity'],
                                             m2_obj['location'], m2_obj['velocity']):
                        events.append({
                            'type': 'moving_towards',
                            'frame': i,
                            'subject': M3,
                            'target': M2,
                            'description': f"Object {M3} is moving towards object {M2}"
                        })
            
            # Check M2 trajectory change (after being hit by M3)
            if M2 is not None and M2 in frame_objects and i > 0:
                m2_obj = frame_objects[M2]
                
                prev_frame_objects = {obj['object_id']: obj for obj in trajectory[i-1]['objects']}
                if M2 in prev_frame_objects:
                    prev_m2_obj = prev_frame_objects[M2]
                    
                    # Check for significant velocity direction change
                    current_vel = np.array(m2_obj['velocity'])
                    prev_vel = np.array(prev_m2_obj['velocity'])
                    
                    # Calculate dot product of velocity directions
                    if np.linalg.norm(current_vel) > 0.1 and np.linalg.norm(prev_vel) > 0.1:
                        dot_product = np.dot(current_vel, prev_vel) / (np.linalg.norm(current_vel) * np.linalg.norm(prev_vel))
                        
                        # If dot product is below threshold, significant direction change occurred
                        if dot_product < 0.5 and not m2_trajectory_changed:
                            m2_trajectory_changed = True
                            analysis['trajectory_deviation'] = True
                            events.append({
                                'type': 'trajectory_deviation',
                                'frame': i,
                                'subject': M2,
                                'description': f"Object {M2} deviates from its original trajectory",
                                'direction_change': float(np.arccos(np.clip(dot_product, -1, 1)))
                            })
            
            # Check S starts moving
            if S is not None and S in frame_objects:
                s_obj = frame_objects[S]
                velocity_mag = self._calculate_velocity_magnitude(s_obj['velocity'])
                
                if velocity_mag > 0.1 and i > 0:
                    prev_frame_objects = {obj['object_id']: obj for obj in trajectory[i-1]['objects']}
                    if S in prev_frame_objects:
                        prev_s_obj = prev_frame_objects[S]
                        prev_velocity_mag = self._calculate_velocity_magnitude(prev_s_obj['velocity'])
                        
                        if prev_velocity_mag < 0.01:
                            events.append({
                                'type': 'start_moving',
                                'frame': i,
                                'subject': S,
                                'description': f"Object {S} starts moving"
                            })
            
            # This project does not rely on camera view field (inside_camera_view), skip related events
        
        # Add collision events
        for collision in data['collision']:
            events.append({
                'type': 'collision',
                'frame': collision['frame_id'],
                'subjects': collision['object_ids'],
                'location': collision['location'],
                'description': f"Collision between objects {collision['object_ids']}"
            })
        
        analysis['events'] = events
    
    def _check_moving_towards_before_collision(self, trajectory: List[Dict], obj_id: int, target_id: int, collision_frame: int) -> bool:
        """Check if object was moving toward target before collision."""
        # Check trajectory for several frames before collision
        frames_to_check = min(5, collision_frame)  # Check at most 5 frames before collision
        
        for i in range(max(0, collision_frame - frames_to_check), collision_frame):
            frame_objects = {obj['object_id']: obj for obj in trajectory[i]['objects']}
            
            if obj_id in frame_objects and target_id in frame_objects:
                obj_data = frame_objects[obj_id]
                target_data = frame_objects[target_id]
                
                # Check if moving toward target
                if self._is_moving_towards(obj_data['location'], obj_data['velocity'],
                                         target_data['location'], target_data['velocity']):
                    return True
        
        return False
    
    def generate_qa_templates(self, data: Dict, analysis: Dict) -> List[Dict]:
        """Generate QA pairs for the double scenario."""
        if any(x is None for x in [analysis['M1'], analysis['M2'], analysis['M3'], analysis['S']]):
            return []
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects)
        m3_desc = self._get_object_description(analysis['M3'], objects)
        s_desc = self._get_object_description(analysis['S'], objects)
        
        # Add distractor object descriptions based on setting type
        as_desc = None
        as1_desc = None
        as2_desc = None
        am_desc = None
        am1_desc = None
        am2_desc = None
        
        if hasattr(self, 'scenario_name'):
            if hasattr(self, 'setting') and self.setting:
                # Assign distractor object descriptions based on setting type
                if self.setting == SettingType.ADD_ONE_STATIC and len(analysis.get('added_static', [])) >= 1:
                    as_desc = self._get_object_description(analysis['added_static'][0], objects)
                elif self.setting == SettingType.ADD_TWO_STATIC and len(analysis.get('added_static', [])) >= 2:
                    as1_desc = self._get_object_description(analysis['added_static'][0], objects)
                    as2_desc = self._get_object_description(analysis['added_static'][1], objects)
                elif self.setting == SettingType.ADD_ONE_MOVING and len(analysis.get('added_moving', [])) >= 1:
                    am_desc = self._get_object_description(analysis['added_moving'][0], objects)
                elif self.setting == SettingType.ADD_TWO_MOVING and len(analysis.get('added_moving', [])) >= 2:
                    am1_desc = self._get_object_description(analysis['added_moving'][0], objects)
                    am2_desc = self._get_object_description(analysis['added_moving'][1], objects)
        
        qa_pairs = []

        qa_pairs.append({
            "question": f"Does the {m1_desc}'s collision with the {s_desc} affect the {s_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Does the {m2_desc}'s collision with the {m1_desc} affect the {s_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Does the {m3_desc}'s collision with the {m2_desc} affect the {s_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"Because the {m1_desc} collided with the {s_desc}.",
                f"Because the {m2_desc} did not collide with the {m1_desc}.",
                f"Because the {m3_desc} collided with the {m2_desc}.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"Because the {m1_desc} collided with the {s_desc}.",
                f"Because the {m2_desc} did not collide with the {m1_desc}.",
                f"Because the {m3_desc} collided with the {m2_desc}.",
                f"Because the {as_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"Because the {m1_desc} collided with the {s_desc}.",
                f"Because the {m2_desc} did not collide with the {m1_desc}.",
                f"Because the {m3_desc} collided with the {m2_desc}.",
                f"Because the {as1_desc} was present.",
                f"Because the {as2_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"Because the {m1_desc} collided with the {s_desc}.",
                f"Because the {m2_desc} did not collide with the {m1_desc}.",
                f"Because the {m3_desc} collided with the {m2_desc}.",
                f"Because the {am_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"Because the {m1_desc} collided with the {s_desc}.",
                f"Because the {m2_desc} did not collide with the {m1_desc}.",
                f"Because the {m3_desc} collided with the {m2_desc}.",
                f"Because the {am1_desc} was present.",
                f"Because the {am2_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        correct_answers = [f"Because the {m1_desc} collided with the {s_desc}."]
        shuffled_options = random.sample(options, len(options))
        answer_indices = [shuffled_options.index(ans) for ans in correct_answers]
        answer_indices.sort()
        qa_pairs.append({
            "question": f"Why did the {s_desc} move?",
            "answer": answer_indices,
            "question_type": "causal_attribution",
            "question_rung": "discovery",
            "answer_type": "multi_choice",
            "options": shuffled_options
        })

        qa_pairs.append({
            "question": f"If we force the {m2_desc} to collide with the {m1_desc}, will the {m1_desc} cause the {s_desc} to move?",
            "answer": "No",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If we force the {m3_desc} not to collide with the {m2_desc}, will the {m1_desc} cause the {s_desc} to move?",
            "answer": "No",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If we force the {m3_desc} not to collide with the {m2_desc}, will the {m2_desc} cause the {s_desc} to stay stationary?",
            "answer": "Yes",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {m1_desc} had not collided with the {s_desc}, would the {s_desc} still have moved?",
            "answer": "No",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {m2_desc} had collided with the {m1_desc}, would the {s_desc} still have moved?",
            "answer": "No",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {m3_desc} had not collided with the {m2_desc}, would the {s_desc} still have moved?",
            "answer": "No",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} collided with the {s_desc} sufficient for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} did not collide with the {m1_desc} sufficient for the {s_desc} to move?",
            "answer": "No",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m3_desc} collided with the {m2_desc} sufficient for the {s_desc} to move?",
            "answer": "No",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} collided with the {s_desc} necessary for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} did not collide with the {m1_desc} necessary for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m3_desc} collided with the {m2_desc} necessary for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} moves toward the {m1_desc}.",
                f"The {m2_desc} does not collide with the {m1_desc}.",
                f"The {m3_desc} moves toward the {m2_desc}.",
                f"The {m3_desc} collides with the {m2_desc}.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} moves toward the {m1_desc}.",
                f"The {m2_desc} does not collide with the {m1_desc}.",
                f"The {m3_desc} moves toward the {m2_desc}.",
                f"The {m3_desc} collides with the {m2_desc}.",
                f"The {as_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} moves toward the {m1_desc}.",
                f"The {m2_desc} does not collide with the {m1_desc}.",
                f"The {m3_desc} moves toward the {m2_desc}.",
                f"The {m3_desc} collides with the {m2_desc}.",
                f"The {as1_desc} is present.",
                f"The {as2_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} moves toward the {m1_desc}.",
                f"The {m2_desc} does not collide with the {m1_desc}.",
                f"The {m3_desc} moves toward the {m2_desc}.",
                f"The {m3_desc} collides with the {m2_desc}.",
                f"The {am_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} moves toward the {m1_desc}.",
                f"The {m2_desc} does not collide with the {m1_desc}.",
                f"The {m3_desc} moves toward the {m2_desc}.",
                f"The {m3_desc} collides with the {m2_desc}.",
                f"The {am1_desc} is present.",
                f"The {am2_desc} is present.",
                f"None.",
            ]
        mapping = {
            "HP": [f"The {m1_desc} moves toward the {s_desc}.", f"The {m1_desc} collides with the {s_desc}.", f"The {m2_desc} does not collide with the {m1_desc}.", f"The {m3_desc} moves toward the {m2_desc}.", f"The {m3_desc} collides with the {m2_desc}."],
            "BV": [f"The {m1_desc} moves toward the {s_desc}.", f"The {m1_desc} collides with the {s_desc}.", f"The {m2_desc} does not collide with the {m1_desc}.", f"The {m3_desc} moves toward the {m2_desc}.", f"The {m3_desc} collides with the {m2_desc}."],
            "DBV": [f"The {m1_desc} moves toward the {s_desc}.", f"The {m1_desc} collides with the {s_desc}."],
            "Boc": [f"The {m1_desc} moves toward the {s_desc}.", f"The {m1_desc} collides with the {s_desc}.", f"The {m2_desc} does not collide with the {m1_desc}.", f"The {m3_desc} moves toward the {m2_desc}.", f"The {m3_desc} collides with the {m2_desc}."],
        }
        shuffled_options = random.sample(options, len(options))
        answer = {}
        for definition in mapping:
            correct_answers = mapping[definition]
            answer_indices = [shuffled_options.index(ans) for ans in correct_answers]
            answer_indices.sort()
            answer[definition] = answer_indices
        qa_pairs.append({
            "question": f"What is the actual cause of the {s_desc} moving?",
            "answer": answer,
            "question_type": "actual_cause",
            "question_rung": "counterfactual",
            "answer_type": "multi_choice",
            "options": shuffled_options
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} collided with the {s_desc} responsible for the {s_desc} moving?",
            "answer": "Yes",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} did not collide with the {m1_desc} responsible for the {s_desc} moving?",
            "answer": "No",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m3_desc} collided with the {m2_desc} responsible for the {s_desc} moving?",
            "answer": "No",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        return qa_pairs
    
    def generate_cr_templates(self, data: Dict, analysis: Dict) -> Dict[str, Any]:
        """Generate causal reasoning templates including causal graph and twin networks."""
        if any(x is None for x in [analysis['M1'], analysis['M2'], analysis['M3'], analysis['S']]):
            return {
                'causal_graph': None,
                'twin_network': None
            }
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects)
        m3_desc = self._get_object_description(analysis['M3'], objects)
        s_desc = self._get_object_description(analysis['S'], objects)
        
        # Add distractor object descriptions based on setting type
        as_desc = None
        as1_desc = None
        as2_desc = None
        am_desc = None
        am1_desc = None
        am2_desc = None
        
        if hasattr(self, 'scenario_name'):
            if hasattr(self, 'setting') and self.setting:
                # Assign distractor object descriptions based on setting type
                if self.setting == SettingType.ADD_ONE_STATIC and len(analysis.get('added_static', [])) >= 1:
                    as_desc = self._get_object_description(analysis['added_static'][0], objects)
                elif self.setting == SettingType.ADD_TWO_STATIC and len(analysis.get('added_static', [])) >= 2:
                    as1_desc = self._get_object_description(analysis['added_static'][0], objects)
                    as2_desc = self._get_object_description(analysis['added_static'][1], objects)
                elif self.setting == SettingType.ADD_ONE_MOVING and len(analysis.get('added_moving', [])) >= 1:
                    am_desc = self._get_object_description(analysis['added_moving'][0], objects)
                elif self.setting == SettingType.ADD_TWO_MOVING and len(analysis.get('added_moving', [])) >= 2:
                    am1_desc = self._get_object_description(analysis['added_moving'][0], objects)
                    am2_desc = self._get_object_description(analysis['added_moving'][1], objects)
        
        causal_graph = {
            "variables": {
                "X1": f"The {m1_desc}'s collision with the {s_desc}",
                "X2": f"The {m2_desc}'s collision with the {m1_desc}",
                "X3": f"The {m3_desc}'s collision with the {m2_desc}",
                "Z1": f"The {m1_desc}'s motion toward the {s_desc}",
                "Z2": f"The {m2_desc}'s motion toward the {m1_desc}",
                "Z3": f"The {m3_desc}'s motion toward the {m2_desc}",
                "Y": f"The {s_desc}'s motion"
            },
            "edges": [
                "Z1 -> X1",
                "Z2 -> X2",
                "Z2 -> X3",
                "Z3 -> X3",
                "X1 -> Y",
                "X2 -> X1",
                "X3 -> X2"
            ]
        }

        twin_network = {
            "factual_world": "Z1=1, Z2=1, Z3=1, X1=1, X2=0, X3=1, Y=1",
            "counterfactual_world": {
                "do(X1=0)": "Z1=1, Z2=?, Z3=?, X2=?, X3=?, Y=0",
                "do(X2=1)": "Z1=1, Z2=1, Z3=?, X1=0, X3=?, Y=0",
                "do(X3=0)": "Z1=1, Z2=1, Z3=1, X1=0, X2=1, Y=0"
            }
        }

        if self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            causal_graph["variables"]["S"] = f"The {as_desc}'s presence"
            twin_network["factual_world"] += f", S=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", S=1"
            twin_network["counterfactual_world"]["do(X2=1)"] += f", S=1"
            twin_network["counterfactual_world"]["do(X3=0)"] += f", S=1"
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            causal_graph["variables"]["S1"] = f"The {as1_desc}'s presence"
            causal_graph["variables"]["S2"] = f"The {as2_desc}'s presence"
            twin_network["factual_world"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(X2=1)"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(X3=0)"] += f", S1=1, S2=1"
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            causal_graph["variables"]["M"] = f"The {am_desc}'s presence"
            twin_network["factual_world"] += f", M=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", M=1"
            twin_network["counterfactual_world"]["do(X2=1)"] += f", M=1"
            twin_network["counterfactual_world"]["do(X3=0)"] += f", M=1"
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            causal_graph["variables"]["M1"] = f"The {am1_desc}'s presence"
            causal_graph["variables"]["M2"] = f"The {am2_desc}'s presence"
            twin_network["factual_world"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(X2=1)"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(X3=0)"] += f", M1=1, M2=1"

        return {
            'causal_graph': causal_graph,
            'twin_network': twin_network
        }