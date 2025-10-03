import random
from typing import List, Dict, Any
from enum import Enum
from base_generator import BaseScenarioQAGenerator, SettingType


class SwitchScenarioQAGenerator(BaseScenarioQAGenerator):
    """
    QA generator for the Switch scenario.
    M1 moves slowly toward S, M2 moves fast toward M1; M2 hits M1, then M1 hits S.
    Although M2 is in the causal chain, M1 is the direct cause of S's motion.
    """
    
    def __init__(self):
        super().__init__('switch')
    
    def analyze_scenario(self, data: Dict, setting: SettingType = None) -> Dict[str, Any]:
        """Analyze important information and events in the switch scenario."""
        objects = data['object_property']
        trajectory = data['motion_trajectory']
        collisions = data['collision']
        
        analysis = {
            'M1': None,  # Final hitter of S (slow)
            'M2': None,  # Hits M1 (fast)
            'S': None,   # Stationary target
            'events': [],
            'collision_info': collisions,
            'collision_sequence': [],  # Collision sequence
            'added_static': [],      # Added static objects
            'added_moving': []       # Added moving objects
        }
        
        # Identify objects by setting type
        if setting:
            object_info = self._identify_objects_by_setting(trajectory, setting)
            analysis['added_static'] = object_info['added_static']
            analysis['added_moving'] = object_info['added_moving']
            
            # Use identified main objects
            if len(object_info['static_objects']) >= 1:
                analysis['S'] = object_info['static_objects'][0]
            if len(object_info['moving_objects']) >= 2:
                analysis['M1'] = object_info['moving_objects'][0]
                analysis['M2'] = object_info['moving_objects'][1]
        else:
            # Original identification logic (backward compatible)
            first_frame = trajectory[0]['objects']
            for obj in first_frame:
                velocity_mag = self._calculate_velocity_magnitude(obj['velocity'])
                if velocity_mag < 0.01:  # Considered stationary
                    analysis['S'] = obj['object_id']
                    break
            
            if analysis['S'] is None:
                return analysis
            
            # Analyze collision sequence to determine M1 and M2
            collision_sequence = sorted(collisions, key=lambda x: x['frame_id'])
            analysis['collision_sequence'] = collision_sequence
            
            if len(collision_sequence) >= 2:
                # First collision: M2 hits M1
                first_collision = collision_sequence[0]
                # Second collision: M1 hits S
                second_collision = collision_sequence[1]
                
                # Determine M1: involved in collision with S
                for obj_id in second_collision['object_ids']:
                    if obj_id != analysis['S']:
                        analysis['M1'] = obj_id
                        break
                
                # Determine M2: in first collision but not M1
                for obj_id in first_collision['object_ids']:
                    if obj_id != analysis['M1']:
                        analysis['M2'] = obj_id
                        break
        
        # Analyze key events
        self._analyze_switch_events(data, analysis)
        
        return analysis
    
    def _analyze_switch_events(self, data: Dict, analysis: Dict):
        """Analyze key events in the switch scenario."""
        trajectory = data['motion_trajectory']
        M1, M2, S = analysis['M1'], analysis['M2'], analysis['S']
        
        events = []
        m1_speed_increased = False  # Whether M1 accelerated after being hit by M2
        
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
            
            # Check M2 moving toward M1
            if M2 is not None and M1 is not None:
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
            
            # Check M1 speed increase (after being hit by M2)
            if M1 is not None and M1 in frame_objects and i > 0:
                m1_obj = frame_objects[M1]
                current_speed = self._calculate_velocity_magnitude(m1_obj['velocity'])
                
                prev_frame_objects = {obj['object_id']: obj for obj in trajectory[i-1]['objects']}
                if M1 in prev_frame_objects:
                    prev_m1_obj = prev_frame_objects[M1]
                    prev_speed = self._calculate_velocity_magnitude(prev_m1_obj['velocity'])
                    
                    # Significant increase indicates a hit
                    if current_speed > prev_speed * 1.5 and not m1_speed_increased:
                        m1_speed_increased = True
                        events.append({
                            'type': 'speed_increase',
                            'frame': i,
                            'subject': M1,
                            'description': f"Object {M1} speeds up after being hit",
                            'speed_change': current_speed - prev_speed
                        })
            
            # Check S starts moving (after being hit by M1)
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
            
        # Early scenarios do not rely on camera view (inside_camera_view)
        
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
        """Generate QA pairs for the switch scenario."""
        if analysis['M1'] is None or analysis['M2'] is None or analysis['S'] is None:
            return []
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects)
        s_desc = self._get_object_description(analysis['S'], objects)
        
        # Add distractor descriptions by setting type
        as_desc = None
        as1_desc = None
        as2_desc = None
        am_desc = None
        am1_desc = None
        am2_desc = None
        
        if hasattr(self, 'scenario_name'):
            if hasattr(self, 'setting') and self.setting:
                # Assign distractor object descriptions by setting type
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
            "question": f"Does the {m1_desc}'s motion toward the {s_desc} before its collision with the {m2_desc} affect the {s_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Does the {m2_desc}'s collision with the {m1_desc} affect the {s_desc}'s motion?",
            "answer": "No",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })  

        qa_pairs.append({
            "question": f"Does the {m1_desc}'s motion toward the {s_desc} after its collision with the {m2_desc} affect the {s_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc} before its collision with the {m2_desc}.",
                f"Because the {m2_desc} collided with the {m1_desc}.",
                f"Because the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc}.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc} before its collision with the {m2_desc}.",
                f"Because the {m2_desc} collided with the {m1_desc}.",
                f"Because the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc}.",
                f"Because the {as_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc} before its collision with the {m2_desc}.",
                f"Because the {m2_desc} collided with the {m1_desc}.",
                f"Because the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc}.",
                f"Because the {as1_desc} was present.",
                f"Because the {as2_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc} before its collision with the {m2_desc}.",
                f"Because the {m2_desc} collided with the {m1_desc}.",
                f"Because the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc}.",
                f"Because the {am_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc} before its collision with the {m2_desc}.",
                f"Because the {m2_desc} collided with the {m1_desc}.",
                f"Because the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc}.",
                f"Because the {am1_desc} was present.",
                f"Because the {am2_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        correct_answers = [f"Because the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc}."]
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
            "question": f"If we force the {m1_desc} not to move toward the {s_desc} after its collision with the {m2_desc}, will the {m1_desc} cause the {s_desc} to move?",
            "answer": "No",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })
        
        qa_pairs.append({
            "question": f"If we force the {m2_desc} not to collide with the {m1_desc}, will the {m1_desc} cause the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })
        
        qa_pairs.append({
            "question": f"If the {m1_desc} had not moved toward the {s_desc} after its collision with the {m2_desc}, would the {s_desc} still have moved?",
            "answer": "No",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })
        
        qa_pairs.append({
            "question": f"If the {m2_desc} had not collided with the {m1_desc}, would the {s_desc} still have moved?",
            "answer": "Yes",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} before its collision with the {m2_desc} sufficient for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} collided with the {m1_desc} sufficient for the {s_desc} to move?",
            "answer": "No",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc} sufficient for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} before its collision with the {m2_desc} necessary for the {s_desc} to move?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} collided with the {m1_desc} necessary for the {s_desc} to move?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc} necessary for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"The {m1_desc} moves toward the {s_desc} before its collision with the {m2_desc}.",
                f"The {m2_desc} collides with the {m1_desc}.",
                f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc} before its collision with the {m2_desc}.",
                f"The {m2_desc} collides with the {m1_desc}.",
                f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}.",
                f"The {as_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc} before its collision with the {m2_desc}.",
                f"The {m2_desc} collides with the {m1_desc}.",
                f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}.",
                f"The {as1_desc} is present.",
                f"The {as2_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc} before its collision with the {m2_desc}.",
                f"The {m2_desc} collides with the {m1_desc}.",
                f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}.",
                f"The {am_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc} before its collision with the {m2_desc}.",
                f"The {m2_desc} collides with the {m1_desc}.",
                f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}.",
                f"The {am1_desc} is present.",
                f"The {am2_desc} is present.",
                f"None.",
            ]
        mapping = {
            "HP": [f"The {m2_desc} collides with the {m1_desc}.", f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}."],
            "BV": [f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}."],
            "DBV": [f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}."],
            "Boc": [f"The {m2_desc} collides with the {m1_desc}.", f"The {m1_desc} moves toward the {s_desc} after its collision with the {m2_desc}."],
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
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} before its collision with the {m2_desc} responsible for the {s_desc} moving?",
            "answer": "No",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })
        
        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} collided with the {m1_desc} responsible for the {s_desc} moving?",
            "answer": "No",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })
        
        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} after its collision with the {m2_desc} responsible for the {s_desc} moving?",
            "answer": "Yes",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        return qa_pairs
    
    def generate_cr_templates(self, data: Dict, analysis: Dict) -> Dict[str, Any]:
        """Generate causal reasoning templates including causal graph and twin networks."""
        if analysis['M1'] is None or analysis['M2'] is None or analysis['S'] is None:
            return {
                'causal_graph': None,
                'twin_network': None
            }
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects)
        s_desc = self._get_object_description(analysis['S'], objects)
        
        # Add distractor descriptions by setting type
        as_desc = None
        as1_desc = None
        as2_desc = None
        am_desc = None
        am1_desc = None
        am2_desc = None
        
        if hasattr(self, 'scenario_name'):
            if hasattr(self, 'setting') and self.setting:
                # Assign distractor object descriptions by setting type
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
                "Z": f"The {m2_desc}'s collision with the {m1_desc}",
                "X1": f"The {m1_desc}'s motion toward the {s_desc} with colliding with the {m2_desc}",
                "X2": f"The {m1_desc}'s motion toward the {s_desc} without colliding with the {m2_desc}",
                "Y": f"The {s_desc}'s motion"
            },
            "edges": [
                "Z -> X1",
                "Z -> X2",
                "X1 -> Y",
                "X2 -> Y"
            ]
        }

        twin_network = {
            "factual_world": "Z=1, X1=1, X2=0, Y=1",
            "counterfactual_world": {
                "do(Z=0)": "X1=0, X2=1, Y=1"
            }
        }

        if self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            causal_graph["variables"]["S"] = f"The {as_desc}'s presence"
            twin_network["factual_world"] += f", S=1"
            twin_network["counterfactual_world"]["do(Z=0)"] += f", S=1"
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            causal_graph["variables"]["S1"] = f"The {as1_desc}'s presence"
            causal_graph["variables"]["S2"] = f"The {as2_desc}'s presence"
            twin_network["factual_world"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(Z=0)"] += f", S1=1, S2=1"
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            causal_graph["variables"]["M"] = f"The {am_desc}'s presence"
            twin_network["factual_world"] += f", M=1"
            twin_network["counterfactual_world"]["do(Z=0)"] += f", M=1"
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            causal_graph["variables"]["M1"] = f"The {am1_desc}'s presence"
            causal_graph["variables"]["M2"] = f"The {am2_desc}'s presence"
            twin_network["factual_world"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(Z=0)"] += f", M1=1, M2=1"

        return {
            "causal_graph": causal_graph,
            "twin_network": twin_network
        }