import random
import numpy as np
from typing import List, Dict, Any
from enum import Enum
from base_generator import BaseScenarioQAGenerator, SettingType


# Bogus scenario QA generator
class BogusScenarioQAGenerator(BaseScenarioQAGenerator):
    """
    QA generator for the Bogus scenario.
    M1 and M2 both move toward S2; M1 hits S2 first, S2 does not hit S1.
    If M1 were removed, M2 would not actually hit S2 due to insufficient momentum.
    This is an example of "spurious causation".
    """
    
    def __init__(self):
        super().__init__('bogus')
    
    def analyze_scenario(self, data: Dict, setting: SettingType = None) -> Dict[str, Any]:
        """Analyze important information and events in the bogus scenario."""
        objects = data['object_property']
        trajectory = data['motion_trajectory']
        collisions = data['collision']
        
        analysis = {
            'M1': None,  # Object that actually hits S2
            'M2': None,  # Object moving toward S2 but with insufficient momentum
            'S1': None,  # Stationary target 1 (collinear with S2)
            'S2': None,  # Stationary target 2 (hit by M1)
            'events': [],
            'collision_info': collisions,
            'collinear_objects': [],  # Collinear objects
            'm2_insufficient_momentum': False,  # Whether M2 has insufficient momentum
            'added_static': [],      # Added static objects
            'added_moving': []       # Added moving objects
        }
        
        # Identify objects based on setting type
        if setting:
            object_info = self._identify_objects_by_setting(trajectory, setting)
            analysis['added_static'] = object_info['added_static']
            analysis['added_moving'] = object_info['added_moving']
            
            # Use identified main objects
            if len(object_info['static_objects']) >= 2:
                analysis['S1'] = object_info['static_objects'][0]
                analysis['S2'] = object_info['static_objects'][1]
            if len(object_info['moving_objects']) >= 2:
                analysis['M1'] = object_info['moving_objects'][0]
                analysis['M2'] = object_info['moving_objects'][1]
        else:
            # Original object identification logic (backward compatible)
            # Find initial stationary objects as S1 and S2
            first_frame = trajectory[0]['objects']
            static_objects = []
            moving_objects = []
            
            for obj in first_frame:
                velocity_mag = self._calculate_velocity_magnitude(obj['velocity'])
                if velocity_mag < 0.01:  # Considered stationary
                    static_objects.append(obj['object_id'])
                else:
                    moving_objects.append(obj['object_id'])
            
            if len(static_objects) < 2 or len(moving_objects) < 2:
                return analysis
            
            # Determine S2 and M1 from collision information
            if collisions:
                collision = collisions[0]  # Assume main collision is M1 hitting S2
                collision_objects = collision['object_ids']
                
                # Determine S2 (stationary object in collision)
                for obj_id in collision_objects:
                    if obj_id in static_objects:
                        analysis['S2'] = obj_id
                        break
                
                # Determine M1 (moving object in collision)
                for obj_id in collision_objects:
                    if obj_id in moving_objects:
                        analysis['M1'] = obj_id
                        break
            
            # Determine S1 (other stationary object)
            for obj_id in static_objects:
                if obj_id != analysis['S2']:
                    analysis['S1'] = obj_id
                    break
            
            # Determine M2 (other moving object)
            for obj_id in moving_objects:
                if obj_id != analysis['M1']:
                    analysis['M2'] = obj_id
                    break
        
        # Check collinearity and M2's momentum
        self._analyze_collinearity_and_momentum(data, analysis)
        
        # Analyze key events
        self._analyze_bogus_events(data, analysis)
        
        return analysis
    
    def _analyze_collinearity_and_momentum(self, data: Dict, analysis: Dict):
        """Analyze collinearity and M2's momentum situation."""
        trajectory = data['motion_trajectory']
        M2, S1, S2 = analysis['M2'], analysis['S1'], analysis['S2']
        
        if any(x is None for x in [M2, S1, S2]):
            return
        
        # Check if M2, S2, S1 are roughly collinear
        first_frame = trajectory[0]['objects']
        frame_objects = {obj['object_id']: obj for obj in first_frame}
        
        if all(obj_id in frame_objects for obj_id in [M2, S1, S2]):
            m2_pos = np.array(frame_objects[M2]['location'][:2])  # Only consider x,y coordinates
            s1_pos = np.array(frame_objects[S1]['location'][:2])
            s2_pos = np.array(frame_objects[S2]['location'][:2])
            
            # Calculate if three points are roughly collinear
            # Use vector cross product to determine collinearity
            vec1 = s2_pos - m2_pos
            vec2 = s1_pos - s2_pos
            
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cross_product = abs(np.cross(vec1, vec2))
                # If cross product is small, the three points are roughly collinear
                if cross_product < 0.5:  # Adjustable threshold
                    analysis['collinear_objects'] = [M2, S2, S1]
        
        # Analyze M2's momentum situation
        # Judge by observing whether M2 approaches S2 throughout the trajectory
        min_distance_to_s2 = float('inf')
        m2_final_velocity = 0
        
        for frame in trajectory:
            frame_objects = {obj['object_id']: obj for obj in frame['objects']}
            if M2 in frame_objects and S2 in frame_objects:
                m2_obj = frame_objects[M2]
                s2_obj = frame_objects[S2]
                
                distance = self._calculate_distance(m2_obj['location'], s2_obj['location'])
                min_distance_to_s2 = min(min_distance_to_s2, distance)
                
                # Record M2's final velocity
                m2_final_velocity = self._calculate_velocity_magnitude(m2_obj['velocity'])
        
        # If M2 still has some distance when closest to S2 and final velocity is small, insufficient momentum
        if min_distance_to_s2 > 0.5 and m2_final_velocity < 0.1:
            analysis['m2_insufficient_momentum'] = True
    
    def _analyze_bogus_events(self, data: Dict, analysis: Dict):
        """Analyze key events in the bogus scenario."""
        trajectory = data['motion_trajectory']
        M1, M2, S1, S2 = analysis['M1'], analysis['M2'], analysis['S1'], analysis['S2']
        
        events = []
        m2_stopped = False
        
        # Analyze each frame
        for i, frame in enumerate(trajectory):
            frame_objects = {obj['object_id']: obj for obj in frame['objects']}
            
            # Check M1 moving toward S2
            if M1 is not None and S2 is not None:
                if M1 in frame_objects and S2 in frame_objects:
                    m1_obj = frame_objects[M1]
                    s2_obj = frame_objects[S2]
                    
                    if self._is_moving_towards(m1_obj['location'], m1_obj['velocity'],
                                             s2_obj['location'], s2_obj['velocity']):
                        events.append({
                            'type': 'moving_towards',
                            'frame': i,
                            'subject': M1,
                            'target': S2,
                            'description': f"Object {M1} is moving towards object {S2}"
                        })
            
            # Check M2 moving toward S2
            if M2 is not None and S2 is not None:
                if M2 in frame_objects and S2 in frame_objects:
                    m2_obj = frame_objects[M2]
                    s2_obj = frame_objects[S2]
                    
                    if self._is_moving_towards(m2_obj['location'], m2_obj['velocity'],
                                             s2_obj['location'], s2_obj['velocity']):
                        events.append({
                            'type': 'moving_towards',
                            'frame': i,
                            'subject': M2,
                            'target': S2,
                            'description': f"Object {M2} is moving towards object {S2}"
                        })
            
            # Check if M2 stops moving (insufficient momentum)
            if M2 is not None and M2 in frame_objects and not m2_stopped:
                m2_obj = frame_objects[M2]
                velocity_mag = self._calculate_velocity_magnitude(m2_obj['velocity'])
                
                if velocity_mag < 0.05:  # Considered basically stopped
                    # Check if M2 hasn't reached S2 yet
                    if S2 in frame_objects:
                        s2_obj = frame_objects[S2]
                        distance = self._calculate_distance(m2_obj['location'], s2_obj['location'])
                        
                        if distance > 0.3:  # Still some distance away when stopping
                            m2_stopped = True
                            events.append({
                                'type': 'insufficient_momentum',
                                'frame': i,
                                'subject': M2,
                                'description': f"Object {M2} stops before reaching object {S2} due to insufficient momentum",
                                'distance_remaining': distance
                            })
            
            # Check S2 starts moving
            if S2 is not None and S2 in frame_objects:
                s2_obj = frame_objects[S2]
                velocity_mag = self._calculate_velocity_magnitude(s2_obj['velocity'])
                
                if velocity_mag > 0.1 and i > 0:
                    prev_frame_objects = {obj['object_id']: obj for obj in trajectory[i-1]['objects']}
                    if S2 in prev_frame_objects:
                        prev_s2_obj = prev_frame_objects[S2]
                        prev_velocity_mag = self._calculate_velocity_magnitude(prev_s2_obj['velocity'])
                        
                        if prev_velocity_mag < 0.01:
                            events.append({
                                'type': 'start_moving',
                                'frame': i,
                                'subject': S2,
                                'description': f"Object {S2} starts moving"
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
    
    def generate_qa_templates(self, data: Dict, analysis: Dict) -> List[Dict]:
        """Generate QA pairs for the bogus scenario."""
        if any(x is None for x in [analysis['M1'], analysis['M2'], analysis['S1'], analysis['S2']]):
            return []
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects)
        s1_desc = self._get_object_description(analysis['S1'], objects)
        s2_desc = self._get_object_description(analysis['S2'], objects)
        
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
            "question": f"Does the {m1_desc}'s collision with the {s2_desc} affect the {s1_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Does the {m2_desc}'s collision with the {s2_desc} affect the {s1_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Does the {s2_desc}'s motion toward the {s1_desc} affect the {s1_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"Because the {m1_desc} collided with the {s2_desc}.",
                f"Because the {m2_desc} did not collide with the {s2_desc}.",
                f"Because the {s2_desc} did not move toward the {s1_desc}.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"Because the {m1_desc} collided with the {s2_desc}.",
                f"Because the {m2_desc} did not collide with the {s2_desc}.",
                f"Because the {s2_desc} did not move toward the {s1_desc}.",
                f"Because the {as_desc} was present.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"Because the {m1_desc} collided with the {s2_desc}.",
                f"Because the {m2_desc} did not collide with the {s2_desc}.",
                f"Because the {s2_desc} did not move toward the {s1_desc}.",
                f"Because the {as1_desc} was present.",
                f"Because the {as2_desc} was present.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"Because the {m1_desc} collided with the {s2_desc}.",
                f"Because the {m2_desc} did not collide with the {s2_desc}.",
                f"Because the {s2_desc} did not move toward the {s1_desc}.",
                f"Because the {am_desc} was present.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"Because the {m1_desc} collided with the {s2_desc}.",
                f"Because the {m2_desc} did not collide with the {s2_desc}.",
                f"Because the {s2_desc} did not move toward the {s1_desc}.",
                f"Because the {am1_desc} was present.",
                f"Because the {am2_desc} was present.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        correct_answers = [f"Because the {m2_desc} did not collide with the {s2_desc}."]
        shuffled_options = random.sample(options, len(options))
        answer_indices = [shuffled_options.index(ans) for ans in correct_answers]
        answer_indices.sort()
        qa_pairs.append({
            "question": f"Why did the {s1_desc} stay stationary?",
            "answer": answer_indices,
            "question_type": "causal_attribution",
            "question_rung": "discovery",
            "answer_type": "multi_choice",
            "options": shuffled_options
        })

        qa_pairs.append({
            "question": f"If we force the {m1_desc} not to collide with the {s2_desc} and force the {m2_desc} to collide with the {s2_desc}, will the {s2_desc} cause the {s1_desc} to move?",
            "answer": "Yes",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If we force the {m1_desc} not to collide with the {s2_desc} and force the {m2_desc} to collide with the {s2_desc}, will the {m1_desc} cause the {s1_desc} to move?",
            "answer": "No",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If we force the {m2_desc} to collide with the {s2_desc}, will the {s2_desc} cause the {s1_desc} to move?",
            "answer": "No",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {m1_desc} had not collided with the {s2_desc} and the {m2_desc} had collided with the {s2_desc}, would the {s1_desc} still have stayed stationary?",
            "answer": "No",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {m2_desc} had collided with the {s2_desc}, would the {s1_desc} still have stayed stationary?",
            "answer": "Yes",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} collided with the {s2_desc} sufficient for the {s1_desc} to stay stationary?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} did not collide with the {s2_desc} sufficient for the {s1_desc} to stay stationary?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {s2_desc} did not move toward the {s1_desc} sufficient for the {s1_desc} to stay stationary?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} collided with the {s2_desc} necessary for the {s1_desc} to stay stationary?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} did not collide with the {s2_desc} necessary for the {s1_desc} to stay stationary?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {s2_desc} did not move toward the {s1_desc} necessary for the {s1_desc} to stay stationary?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"The {m1_desc} collides with the {s2_desc}.",
                f"The {m2_desc} does not collide with the {s2_desc}.",
                f"The {m1_desc} collides with the {s2_desc} and the {m2_desc} does not collide with the {s2_desc}.",
                f"The {s2_desc} does not move toward the {s1_desc}.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"The {m1_desc} collides with the {s2_desc}.",
                f"The {m2_desc} does not collide with the {s2_desc}.",
                f"The {m1_desc} collides with the {s2_desc} and the {m2_desc} does not collide with the {s2_desc}.",
                f"The {s2_desc} does not move toward the {s1_desc}.",
                f"The {as_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"The {m1_desc} collides with the {s2_desc}.",
                f"The {m2_desc} does not collide with the {s2_desc}.",
                f"The {m1_desc} collides with the {s2_desc} and the {m2_desc} does not collide with the {s2_desc}.",
                f"The {s2_desc} does not move toward the {s1_desc}.",
                f"The {as1_desc} is present.",
                f"The {as2_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"The {m1_desc} collides with the {s2_desc}.",
                f"The {m2_desc} does not collide with the {s2_desc}.",
                f"The {m1_desc} collides with the {s2_desc} and the {m2_desc} does not collide with the {s2_desc}.",
                f"The {s2_desc} does not move toward the {s1_desc}.",
                f"The {am_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"The {m1_desc} collides with the {s2_desc}.",
                f"The {m2_desc} does not collide with the {s2_desc}.",
                f"The {m1_desc} collides with the {s2_desc} and the {m2_desc} does not collide with the {s2_desc}.",
                f"The {s2_desc} does not move toward the {s1_desc}.",
                f"The {am1_desc} is present.",
                f"The {am2_desc} is present.",
                f"None.",
            ]
        mapping = {
            "HP": [f"The {m1_desc} collides with the {s2_desc} and the {m2_desc} does not collide with the {s2_desc}.", f"The {s2_desc} does not move toward the {s1_desc}."],
            "BV": [f"The {m2_desc} does not collide with the {s2_desc}.", f"The {s2_desc} does not move toward the {s1_desc}."],
            "DBV": [f"The {m2_desc} does not collide with the {s2_desc}.", f"The {s2_desc} does not move toward the {s1_desc}."],
            "Boc": [f"The {m1_desc} collides with the {s2_desc}.", f"The {m2_desc} does not collide with the {s2_desc}.", f"The {s2_desc} does not move toward the {s1_desc}."],
        }
        shuffled_options = random.sample(options, len(options))
        answer = {}
        for definition in mapping:
            correct_answers = mapping[definition]
            answer_indices = [shuffled_options.index(ans) for ans in correct_answers]
            answer_indices.sort()
            answer[definition] = answer_indices
        qa_pairs.append({
            "question": f"What is the actual cause of the {s1_desc} staying stationary?",
            "answer": answer,
            "question_type": "actual_cause",
            "question_rung": "counterfactual",
            "answer_type": "multi_choice",
            "options": shuffled_options
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} collided with the {s2_desc} responsible for the {s1_desc} staying stationary?",
            "answer": "No",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} did not collide with the {s2_desc} responsible for the {s1_desc} staying stationary?",
            "answer": "Yes",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {s2_desc} did not move toward the {s1_desc} responsible for the {s1_desc} staying stationary?",
            "answer": "No",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        return qa_pairs
    
    def generate_cr_templates(self, data: Dict, analysis: Dict) -> Dict[str, Any]:
        """Generate causal reasoning templates including causal graph and twin networks."""
        if any(x is None for x in [analysis['M1'], analysis['M2'], analysis['S1'], analysis['S2']]):
            return {
                'causal_graph': None,
                'twin_network': None
            }
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects)
        s1_desc = self._get_object_description(analysis['S1'], objects)
        s2_desc = self._get_object_description(analysis['S2'], objects)
        
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
                "X1": f"The {m1_desc}'s collision with the {s2_desc}",
                "X2": f"The {m2_desc}'s collision with the {s2_desc}",
                "Z": f"The {s2_desc}'s motion toward the {s1_desc}",
                "W": f"The {s2_desc} reaches the {s1_desc}",
                "Y": f"The {s1_desc}'s motion"
            },
            "edges": [
                "X1 -> Z",
                "X2 -> Z",
                "Z -> Y",
                "W -> Y"
            ]
        }

        twin_network = {
            "factual_world": "X1=1, X2=0, Z=0, W=0, Y=0",
            "counterfactual_world": {
                "do(X1=0, X2=1)": "Z=1, W=?, Y=W",
                "do(X1=1, X2=1)": "Z=0, W=0, Y=0",
            }
        }

        if self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            causal_graph["variables"]["S"] = f"The {as_desc}'s presence"
            twin_network["factual_world"] += f", S=1"
            twin_network["counterfactual_world"]["do(X1=0, X2=1)"] += f", S=1"
            twin_network["counterfactual_world"]["do(X1=1, X2=1)"] += f", S=1"
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            causal_graph["variables"]["S1"] = f"The {as1_desc}'s presence"
            causal_graph["variables"]["S2"] = f"The {as2_desc}'s presence"
            twin_network["factual_world"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(X1=0, X2=1)"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(X1=1, X2=1)"] += f", S1=1, S2=1"
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            causal_graph["variables"]["M"] = f"The {am_desc}'s presence"
            twin_network["factual_world"] += f", M=1"
            twin_network["counterfactual_world"]["do(X1=0, X2=1)"] += f", M=1"
            twin_network["counterfactual_world"]["do(X1=1, X2=1)"] += f", M=1"
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            causal_graph["variables"]["M1"] = f"The {am1_desc}'s presence"
            causal_graph["variables"]["M2"] = f"The {am2_desc}'s presence"
            twin_network["factual_world"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(X1=0, X2=1)"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(X1=1, X2=1)"] += f", M1=1, M2=1"

        return {
            'causal_graph': causal_graph,
            'twin_network': twin_network
        }