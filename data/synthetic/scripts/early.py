import random
import numpy as np
from typing import List, Dict, Any
from enum import Enum
from base_generator import BaseScenarioQAGenerator, SettingType


class EarlyScenarioQAGenerator(BaseScenarioQAGenerator):
    """
    QA generator for the Early scenario.
    M1 moves toward S1, M2 moves toward S2. M1 hits S1 first (S1 stays
    stationary), then M2 hits S2. If M1 were removed, M2 would hit S2,
    and S2 would then hit S1, causing S1 to move.
    """
    
    def __init__(self):
        super().__init__('early')
    
    def analyze_scenario(self, data: Dict, setting: SettingType = None) -> Dict[str, Any]:
        """Analyze important information and events in the early scenario."""
        objects = data['object_property']
        trajectory = data['motion_trajectory']
        collisions = data['collision']
        
        analysis = {
            'M1': None,  # Moving object that hits S1
            'M2': None,  # Moving object that hits S2
            'S1': None,  # Stationary target 1 (ultimately stationary)
            'S2': None,  # Stationary target 2 (hit by M1)
            'events': [],
            'collision_info': collisions,
            'collision_sequence': [],  # Collision sequence
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
            # Original identification logic (backward compatible)
            # Find initial stationary and moving sets
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
            
            # Analyze collision sequence to determine roles
            collision_sequence = sorted(collisions, key=lambda x: x['frame_id'])
            analysis['collision_sequence'] = collision_sequence
            
            # Keep only moving-static collisions as candidates
            ms_collisions: List[Dict[str, Any]] = []
            for c in collision_sequence:
                objs = c.get('object_ids', [])
                if len(objs) != 2:
                    continue
                a, b = objs[0], objs[1]
                cond = ((a in moving_objects and b in static_objects) or
                        (b in moving_objects and a in static_objects))
                if cond:
                    ms_collisions.append(c)

            # Extract initial positions/velocities for geometric validation
            first_positions = {obj['object_id']: obj['location'] for obj in first_frame}
            first_velocities = {obj['object_id']: obj['velocity'] for obj in first_frame}

            def are_collinear(p1: List[float], p2: List[float], p3: List[float], eps: float = 1e-2) -> bool:
                v1 = np.array(p2) - np.array(p1)
                v2 = np.array(p3) - np.array(p1)
                cross = np.linalg.norm(np.cross(v1, v2))
                return cross <= eps * (np.linalg.norm(v1) + np.linalg.norm(v2) + 1e-6)

            def is_between(p_left: List[float], p_mid: List[float], p_right: List[float], rel_eps: float = 1e-2) -> bool:
                d_lr = self._calculate_distance(p_left, p_right)
                d_lm = self._calculate_distance(p_left, p_mid)
                d_mr = self._calculate_distance(p_mid, p_right)
                return abs((d_lm + d_mr) - d_lr) <= rel_eps * max(1.0, d_lr)

            # From moving-static collision candidates, pick two in order satisfying:
            # (1) First (M1, S1), second (M2, S2)
            # (2) Initially M1→S1 and M2→S2
            # (3) Initially M2, S2, S1 are collinear and S2 lies between them
            if len(ms_collisions) >= 2:
                selected = False
                for i in range(len(ms_collisions) - 1):
                    first_c = ms_collisions[i]
                    second_c = ms_collisions[i + 1]

                    f_objs = first_c['object_ids']
                    s_objs = second_c['object_ids']

                    # 解析 (M1, S1)
                    m1 = f_objs[0] if f_objs[0] in moving_objects else f_objs[1]
                    s1 = f_objs[0] if f_objs[0] in static_objects else f_objs[1]

                    # 解析 (M2, S2)
                    m2 = s_objs[0] if s_objs[0] in moving_objects else s_objs[1]
                    s2 = s_objs[0] if s_objs[0] in static_objects else s_objs[1]

                    # Uniqueness constraint
                    if len({m1, m2}) < 2 or len({s1, s2}) < 2:
                        continue

                    # Initial direction validation
                    if (m1 in first_positions and s1 in first_positions and
                        m1 in first_velocities and s1 in first_velocities and
                        m2 in first_positions and s2 in first_positions and
                        m2 in first_velocities and s2 in first_velocities):
                        m1_towards_s1 = self._is_moving_towards(first_positions[m1], first_velocities[m1],
                                                               first_positions[s1], first_velocities[s1])
                        m2_towards_s2 = self._is_moving_towards(first_positions[m2], first_velocities[m2],
                                                               first_positions[s2], first_velocities[s2])
                        if not (m1_towards_s1 and m2_towards_s2):
                            continue
                    else:
                        # 缺少位姿信息则跳过该组合
                        continue

                    # Collinearity and between-ness validation
                    p_m2, p_s2, p_s1 = first_positions[m2], first_positions[s2], first_positions[s1]
                    if not are_collinear(p_m2, p_s2, p_s1):
                        continue
                    if not is_between(p_m2, p_s2, p_s1):
                        continue

                    # 满足全部约束，记录
                    analysis['M1'] = m1
                    analysis['S1'] = s1
                    analysis['M2'] = m2
                    analysis['S2'] = s2
                    selected = True
                    break

                # Fallback: take earliest two moving-static collisions
                if not selected:
                    first_collision = ms_collisions[0]
                    second_collision = ms_collisions[1]
                    f_objs = first_collision['object_ids']
                    s_objs = second_collision['object_ids']
                    analysis['M1'] = f_objs[0] if f_objs[0] in moving_objects else f_objs[1]
                    analysis['S1'] = f_objs[0] if f_objs[0] in static_objects else f_objs[1]
                    analysis['M2'] = s_objs[0] if s_objs[0] in moving_objects else s_objs[1]
                    analysis['S2'] = s_objs[0] if s_objs[0] in static_objects else s_objs[1]
        
        # Analyze key events
        self._analyze_early_events(data, analysis)
        
        return analysis
    
    def _analyze_early_events(self, data: Dict, analysis: Dict):
        """Analyze key events in the early scenario."""
        trajectory = data['motion_trajectory']
        M1, M2, S1, S2 = analysis['M1'], analysis['M2'], analysis['S1'], analysis['S2']
        
        events = []
        s1_final_state = 'unknown'
        
        # Analyze each frame
        for i, frame in enumerate(trajectory):
            frame_objects = {obj['object_id']: obj for obj in frame['objects']}
            
            # Check M1 moving toward S1
            if M1 is not None and S1 is not None:
                if M1 in frame_objects and S1 in frame_objects:
                    m1_obj = frame_objects[M1]
                    s1_obj = frame_objects[S1]
                    
                    if self._is_moving_towards(m1_obj['location'], m1_obj['velocity'],
                                             s1_obj['location'], s1_obj['velocity']):
                        events.append({
                            'type': 'moving_towards',
                            'frame': i,
                            'subject': M1,
                            'target': S1,
                            'description': f"Object {M1} is moving towards object {S1}"
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
            
            # Check if S1 starts moving (to determine final state)
            if S1 is not None and S1 in frame_objects:
                s1_obj = frame_objects[S1]
                velocity_mag = self._calculate_velocity_magnitude(s1_obj['velocity'])
                
                if velocity_mag > 0.1 and i > 0:
                    prev_frame_objects = {obj['object_id']: obj for obj in trajectory[i-1]['objects']}
                    if S1 in prev_frame_objects:
                        prev_s1_obj = prev_frame_objects[S1]
                        prev_velocity_mag = self._calculate_velocity_magnitude(prev_s1_obj['velocity'])
                        
                        if prev_velocity_mag < 0.01:
                            events.append({
                                'type': 'start_moving',
                                'frame': i,
                                'subject': S1,
                                'description': f"Object {S1} starts moving"
                            })
            
            # Check if S2 starts moving
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
            
        # Early scenarios do not rely on camera view (inside_camera_view)
        
        # Determine S1's final state (last few frames)
        if len(trajectory) > 10 and S1 is not None:
            final_frames = trajectory[-5:]
            s1_moving_count = 0
            
            for frame in final_frames:
                frame_objects = {obj['object_id']: obj for obj in frame['objects']}
                if S1 in frame_objects:
                    s1_obj = frame_objects[S1]
                    velocity_mag = self._calculate_velocity_magnitude(s1_obj['velocity'])
                    if velocity_mag > 0.01:
                        s1_moving_count += 1
            
            # If S1 moves in most of the last frames, consider it moving finally
            s1_final_state = 'moving' if s1_moving_count >= 3 else 'stationary'
        
        analysis['s1_final_state'] = s1_final_state
        
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
        """Generate QA pairs for the early scenario."""
        if any(x is None for x in [analysis['M1'], analysis['M2'], analysis['S1'], analysis['S2']]):
            return []
        
        objects = data['object_property']
        m1_desc = self._get_object_description(analysis['M1'], objects)
        m2_desc = self._get_object_description(analysis['M2'], objects)
        s1_desc = self._get_object_description(analysis['S1'], objects)
        s2_desc = self._get_object_description(analysis['S2'], objects)
        
        # Add distractor descriptions by setting type
        as_desc = None
        as1_desc = None
        as2_desc = None
        am_desc = None
        am1_desc = None
        am2_desc = None
        
        if hasattr(self, 'scenario_name'):
            if hasattr(self, 'setting') and self.setting:
                # Assign distractor descriptions by setting type
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
            "question": f"Does the {m1_desc}'s motion toward the {s1_desc} affect the {s1_desc}'s motion?",
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
                f"Because the {m1_desc} moved toward the {s1_desc}.",
                f"Because the {s2_desc} moved toward the {s1_desc}.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s1_desc}.",
                f"Because the {s2_desc} moved toward the {s1_desc}.",
                f"Because the {as_desc} was present.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s1_desc}.",
                f"Because the {s2_desc} moved toward the {s1_desc}.",
                f"Because the {as1_desc} was present.",
                f"Because the {as2_desc} was present.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s1_desc}.",
                f"Because the {s2_desc} moved toward the {s1_desc}.",
                f"Because the {am_desc} was present.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"Because the {m1_desc} moved toward the {s1_desc}.",
                f"Because the {s2_desc} moved toward the {s1_desc}.",
                f"Because the {am1_desc} was present.",
                f"Because the {am2_desc} was present.",
                f"Because the {s1_desc} moved spontaneously.",
            ]
        correct_answers = [f"Because the {m1_desc} moved toward the {s1_desc}."]
        shuffled_options = random.sample(options, len(options))
        answer_indices = [shuffled_options.index(ans) for ans in correct_answers]
        answer_indices.sort()
        qa_pairs.append({
            "question": f"Why did the {s1_desc} move?",
            "answer": answer_indices,
            "question_type": "causal_attribution",
            "question_rung": "discovery",
            "answer_type": "multi_choice",
            "options": shuffled_options
        })

        qa_pairs.append({
            "question": f"If we force the {m1_desc} not to move toward the {s1_desc}, will the {s2_desc} cause the {s1_desc} to move?",
            "answer": "Yes",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If we force the {s2_desc} not to move toward the {s1_desc}, will the {m1_desc} cause the {s1_desc} to move?",
            "answer": "Yes",
            "question_type": "individual_causal_effect",
            "question_rung": "intervention",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {m1_desc} had not moved toward the {s1_desc}, would the {s1_desc} still have moved?",
            "answer": "Yes",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"If the {s2_desc} had not moved toward the {s1_desc}, would the {s1_desc} still have moved?",
            "answer": "Yes",
            "question_type": "counterfactual_reasoning",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s1_desc} sufficient for the {s1_desc} to move?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {s2_desc} moved toward the {s1_desc} sufficient for the {s1_desc} to move?",
            "answer": "Yes",
            "question_type": "sufficient_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s1_desc} necessary for the {s1_desc} to move?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {s2_desc} moved toward the {s1_desc} necessary for the {s1_desc} to move?",
            "answer": "No",
            "question_type": "necessary_cause",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        options = []
        if not self.setting:
            options = [
                f"The {m1_desc} moves toward the {s1_desc}.",
                f"The {s2_desc} moves toward the {s1_desc}.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            options = [
                f"The {m1_desc} moves toward the {s1_desc}.",
                f"The {s2_desc} moves toward the {s1_desc}.",
                f"The {as_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            options = [
                f"The {m1_desc} moves toward the {s1_desc}.",
                f"The {s2_desc} moves toward the {s1_desc}.",
                f"The {as1_desc} is present.",
                f"The {as2_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            options = [
                f"The {m1_desc} moves toward the {s1_desc}.",
                f"The {s2_desc} moves toward the {s1_desc}.",
                f"The {am_desc} is present.",
                f"None.",
            ]
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            options = [
                f"The {m1_desc} moves toward the {s1_desc}.",
                f"The {s2_desc} moves toward the {s1_desc}.",
                f"The {am1_desc} is present.",
                f"The {am2_desc} is present.",
                f"None.",
            ]
        mapping = {
            "HP": [f"The {m1_desc} moves toward the {s1_desc}."],
            "BV": [f"None."],
            "DBV": [f"The {m1_desc} moves toward the {s1_desc}."],
            "Boc": [f"The {m1_desc} moves toward the {s1_desc}."],
        }
        shuffled_options = random.sample(options, len(options))
        answer = {}
        for definition in mapping:
            correct_answers = mapping[definition]
            answer_indices = [shuffled_options.index(ans) for ans in correct_answers]
            answer_indices.sort()
            answer[definition] = answer_indices
        qa_pairs.append({
            "question": f"What is the actual cause of the {s1_desc} moving?",
            "answer": answer,
            "question_type": "actual_cause",
            "question_rung": "counterfactual",
            "answer_type": "multi_choice",
            "options": shuffled_options
        })

        qa_pairs.append({
            "question": f"Was the fact that the {m1_desc} moved toward the {s1_desc} responsible for the {s1_desc} moving?",
            "answer": "Yes",
            "question_type": "responsibility",
            "question_rung": "counterfactual",
            "answer_type": "yes_no",
        })

        qa_pairs.append({
            "question": f"Was the fact that the {s2_desc} moved toward the {s1_desc} responsible for the {s1_desc} moving?",
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
        
        # Add distractor descriptions by setting type
        as_desc = None
        as1_desc = None
        as2_desc = None
        am_desc = None
        am1_desc = None
        am2_desc = None
        
        if hasattr(self, 'scenario_name'):
            if hasattr(self, 'setting') and self.setting:
                # Assign distractor descriptions by setting type
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
                "X1": f"The {m1_desc}'s motion toward the {s1_desc}",
                "X2": f"The {s2_desc}'s motion toward the {s1_desc}",
                "Y": f"The {s1_desc}'s motion"
            },
            "edges": [
                "X1 -> X2",
                "X1 -> Y",
                "X2 -> Y"
            ]
        }

        twin_network = {
            "factual_world": "X1=1, X2=0, Y=1",
            "counterfactual_world": {
                "do(X1=0)": "X2=1, Y=1"
            }
        }

        if self.setting == SettingType.ADD_ONE_STATIC and as_desc:
            causal_graph["variables"]["S"] = f"The {as_desc}'s presence"
            twin_network["factual_world"] += f", S=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", S=1"
        elif self.setting == SettingType.ADD_TWO_STATIC and as1_desc and as2_desc:
            causal_graph["variables"]["S1"] = f"The {as1_desc}'s presence"
            causal_graph["variables"]["S2"] = f"The {as2_desc}'s presence"
            twin_network["factual_world"] += f", S1=1, S2=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", S1=1, S2=1"
        elif self.setting == SettingType.ADD_ONE_MOVING and am_desc:
            causal_graph["variables"]["M"] = f"The {am_desc}'s presence"
            twin_network["factual_world"] += f", M=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", M=1"
        elif self.setting == SettingType.ADD_TWO_MOVING and am1_desc and am2_desc:
            causal_graph["variables"]["M1"] = f"The {am1_desc}'s presence"
            causal_graph["variables"]["M2"] = f"The {am2_desc}'s presence"
            twin_network["factual_world"] += f", M1=1, M2=1"
            twin_network["counterfactual_world"]["do(X1=0)"] += f", M1=1, M2=1"

        return {
            'causal_graph': causal_graph,
            'twin_network': twin_network
        }