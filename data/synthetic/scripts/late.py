import random
import numpy as np
from typing import List, Dict, Any
from enum import Enum
from base_generator import BaseScenarioQAGenerator, SettingType


class LateScenarioQAGenerator(BaseScenarioQAGenerator):
    """
    QA generator for the Late scenario.
    """
    
    def __init__(self):
        super().__init__('late')
    
    def analyze_scenario(self, data: Dict, setting: SettingType = None) -> Dict[str, Any]:
        """Analyze important information and events in the late scenario."""
        objects = data['object_property']
        trajectory = data['motion_trajectory']
        collisions = data['collision']
        
        analysis = {
            'M1': None,  # Object that actually hits S
            'M2': None,  # Alternative hitter
            'S': None,   # Stationary target
            'events': [],
            'collision_info': collisions,
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
            if len(object_info['moving_objects']) >= 2:
                analysis['M1'] = object_info['moving_objects'][0]
                analysis['M2'] = object_info['moving_objects'][1]
        else:
            # Original identification logic (backward compatible)
            first_frame = trajectory[0]['objects']
            for obj in first_frame:
                velocity_mag = self._calculate_velocity_magnitude(obj['velocity'])
                if velocity_mag < 0.01:
                    analysis['S'] = obj['object_id']
                    break
            
            if analysis['S'] is None:
                return analysis
            
            # Find object that collided with S as M1
            if collisions:
                collision = collisions[0]
                collision_objects = collision['object_ids']
                for obj_id in collision_objects:
                    if obj_id != analysis['S']:
                        analysis['M1'] = obj_id
                        break
            
            # Find another moving object as M2
            for obj in first_frame:
                velocity_mag = self._calculate_velocity_magnitude(obj['velocity'])
                obj_id = obj['object_id']
                if velocity_mag > 0.01 and obj_id != analysis['M1'] and obj_id != analysis['S']:
                    analysis['M2'] = obj_id
                    break
        
        # Analyze key events
        self._analyze_late_events(data, analysis)
        
        return analysis
    
    def _analyze_late_events(self, data: Dict, analysis: Dict):
        """Analyze key events in the late scenario."""
        trajectory = data['motion_trajectory']
        M1, M2, S = analysis['M1'], analysis['M2'], analysis['S']
        
        events = []
        
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
            
            # Check M2 moving toward S
            if M2 is not None and S is not None:
                if M2 in frame_objects and S in frame_objects:
                    m2_obj = frame_objects[M2]
                    s_obj = frame_objects[S]
                    
                    if self._is_moving_towards(m2_obj['location'], m2_obj['velocity'],
                                             s_obj['location'], s_obj['velocity']):
                        events.append({
                            'type': 'moving_towards',
                            'frame': i,
                            'subject': M2,
                            'target': S,
                            'description': f"Object {M2} is moving towards object {S}"
                        })
            
            # Check S starts moving (after collision)
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
        """Generate QA pairs for the late scenario."""
        if analysis['M1'] is None or analysis['S'] is None:
            return []
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        s_desc = self._get_object_description(analysis['S'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects) if analysis['M2'] else "another object"
        
        # 根据设置类型添加干扰物体描述
        as_desc = None
        as1_desc = None
        as2_desc = None
        am_desc = None
        am1_desc = None
        am2_desc = None
        
        if hasattr(self, 'scenario_name'):  # 检查是否有场景名称属性
            if hasattr(self, 'setting') and self.setting:
                # 根据设置类型赋值干扰物体描述
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
            "question": f"Does the {m2_desc}'s collision with the {s_desc} affect the {s_desc}'s motion?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Does the {m1_desc}'s collision with the {s_desc} affect the {m2_desc}'s collision with the {s_desc}?",
            "answer": "Yes",
            "question_type": "causality_identification",
            "question_rung": "discovery",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc}.",
                f"Because the {m2_desc} moved toward the {s_desc}.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc}.",
                f"Because the {m2_desc} moved toward the {s_desc}.",
                f"Because the {as_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc}.",
                f"Because the {m2_desc} moved toward the {s_desc}.",
                f"Because the {as1_desc} was present.",
                f"Because the {as2_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc}.",
                f"Because the {m2_desc} moved toward the {s_desc}.",
                f"Because the {am_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s_desc}.",
                f"Because the {m2_desc} moved toward the {s_desc}.",
                f"Because the {am1_desc} was present.",
                f"Because the {am2_desc} was present.",
                f"Because the {s_desc} moved spontaneously.",
            ]
        correct_answers = [f"Because the {m1_desc} moved toward the {s_desc}."]
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
            "question": f"If we force the {m1_desc} not to move toward the {s_desc}, will the {m2_desc} cause the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If we force the {m2_desc} not to move toward the {s_desc}, will the {m1_desc} cause the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {m1_desc} had not moved toward the {s_desc}, would the {s_desc} still have moved?",
            "answer": "Yes",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {m2_desc} had not moved toward the {s_desc}, would the {s_desc} still have moved?",
            "answer": "Yes",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} sufficient for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} moved toward the {s_desc} sufficient for the {s_desc} to move?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} necessary for the {s_desc} to move?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} moved toward the {s_desc} necessary for the {s_desc} to move?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m2_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} collides with the {s_desc}.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m2_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} collides with the {s_desc}.",
                f"The {as_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m2_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} collides with the {s_desc}.",
                f"The {as1_desc} is present.",
                f"The {as2_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m2_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} collides with the {s_desc}.",
                f"The {am_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"The {m1_desc} moves toward the {s_desc}.",
                f"The {m2_desc} moves toward the {s_desc}.",
                f"The {m1_desc} collides with the {s_desc}.",
                f"The {m2_desc} collides with the {s_desc}.",
                f"The {am1_desc} is present.",
                f"The {am2_desc} is present.",
                f"None.",
            ]
        mapping = {
            "HP": [f"The {m1_desc} moves toward the {s_desc}.", f"The {m1_desc} collides with the {s_desc}."],
            "BV": [f"The {m1_desc} moves toward the {s_desc}.", f"The {m1_desc} collides with the {s_desc}."],
            "DBV": [f"The {m1_desc} moves toward the {s_desc}.", f"The {m1_desc} collides with the {s_desc}."],
            "Boc": [f"The {m1_desc} moves toward the {s_desc}.", f"The {m1_desc} collides with the {s_desc}."],
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
            "question": f"Was the fact that the {m1_desc} moved toward the {s_desc} responsible for the {s_desc} moving?",
            "answer": "Yes",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m2_desc} moved toward the {s_desc} responsible for the {s_desc} moving?",
            "answer": "No",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        return qa_pairs
    
    def generate_cr_templates(self, data: Dict, analysis: Dict) -> Dict[str, Any]:
        """Generate causal reasoning templates including causal graph and twin networks."""
        if analysis['M1'] is None or analysis['S'] is None:
            return {
                'causal_graph': None,
                'twin_network': None
            }
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        s_desc = self._get_object_description(analysis['S'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects) if analysis['M2'] else "another object"
        
        # 根据设置类型添加干扰物体描述
        as_desc = None
        as1_desc = None
        as2_desc = None
        am_desc = None
        am1_desc = None
        am2_desc = None
        
        if hasattr(self, 'scenario_name'):  # 检查是否有场景名称属性
            if hasattr(self, 'setting') and self.setting:
                # 根据设置类型赋值干扰物体描述
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
                "X1": f"The {m1_desc}'s motion toward the {s_desc}",
                "X2": f"The {m2_desc}'s motion toward the {s_desc}",
                "Z1": f"The {m1_desc}'s collision with the {s_desc}",
                "Z2": f"The {m2_desc}'s collision with the {s_desc}",
                "Y": f"The {s_desc}'s motion"
            },
            "edges": [
                "X1 -> Z1",
                "X2 -> Z2",
                "Z1 -> Z2",
                "Z1 -> Y",
                "Z2 -> Y"
            ]
        }

        twin_network = {
            "factual_world": "X1=1, X2=1, Z1=1, Z2=0, Y=1",
            "counterfactual_world": {
                "do(X1=0)": "X2=1, Z1=0, Z2=1, Y=1",
                "do(X2=0)": "X1=1, Z1=1, Z2=0, Y=1"
            }
        }

        if self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            causal_graph["variables"]["S"] = f"The {as_desc}'s presence"
            twin_network["factual_world"] += f", S=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", S=1"
            twin_network["counterfactual_world"]["do(X2=0)"] += f", S=1"
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            causal_graph["variables"]["S1"] = f"The {as1_desc}'s presence"
            causal_graph["variables"]["S2"] = f"The {as2_desc}'s presence"
            twin_network["factual_world"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(X2=0)"] += f", S1=1, S2=1"
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            causal_graph["variables"]["M"] = f"The {am_desc}'s presence"
            twin_network["factual_world"] += f", M=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", M=1"
            twin_network["counterfactual_world"]["do(X2=0)"] += f", M=1"
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            causal_graph["variables"]["M1"] = f"The {am1_desc}'s presence"
            causal_graph["variables"]["M2"] = f"The {am2_desc}'s presence"
            twin_network["factual_world"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(X2=0)"] += f", M1=1, M2=1"


        return {
            'causal_graph': causal_graph,
            'twin_network': twin_network
        }