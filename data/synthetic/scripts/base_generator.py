import numpy as np
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum


class SettingType(Enum):
    """Scenario setting types"""
    ADD_ONE_STATIC = "add_one_static"      # Add one static object
    ADD_TWO_STATIC = "add_two_static"      # Add two static objects  
    ADD_ONE_MOVING = "add_one_moving"      # Add one moving object
    ADD_TWO_MOVING = "add_two_moving"      # Add two moving objects

# Base class: Scenario QA generator
class BaseScenarioQAGenerator(ABC):
    """
    Base class for scenario QA generators
    """
    
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.setting = None  # Add setting attribute
    
    def set_setting(self, setting: SettingType):
        """Set scenario setting type"""
        self.setting = setting
    
    @abstractmethod
    def analyze_scenario(self, data: Dict, setting: SettingType = None) -> Dict[str, Any]:
        """
        Analyze important information and events in the scenario
        
        Args:
            data (Dict): simulation data
            setting (SettingType): scenario setting type
            
        Returns:
            Dict: analysis results
        """
        pass
    
    @abstractmethod
    def generate_qa_templates(self, data: Dict, analysis: Dict) -> List[Dict]:
        """
        Generate QA pairs based on analysis results
        
        Args:
            data (Dict): simulation data
            analysis (Dict): scenario analysis results
            
        Returns:
            List[Dict]: list of QA pairs
        """
        pass
    
    @abstractmethod
    def generate_cr_templates(self, data: Dict, analysis: Dict) -> Dict[str, Any]:
        """
        Generate causal reasoning templates including causal graph and twin networks
        
        Args:
            data (Dict): simulation data
            analysis (Dict): scenario analysis results
            
        Returns:
            Dict: dictionary containing 'causal_graph' and 'twin_networks'
        """
        pass
    
    def _calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate distance between two positions"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def _calculate_velocity_magnitude(self, velocity: List[float]) -> float:
        """Calculate velocity magnitude"""
        return np.sqrt(sum(v ** 2 for v in velocity))
    
    def _is_moving_towards(self, obj1_pos: List[float], obj1_vel: List[float], 
                          obj2_pos: List[float], obj2_vel: List[float]) -> bool:
        """Check if obj1 is moving towards obj2"""
        direction_to_obj2 = [obj2_pos[i] - obj1_pos[i] for i in range(3)]
        direction_magnitude = np.sqrt(sum(d ** 2 for d in direction_to_obj2))
        
        if direction_magnitude == 0:
            return False
            
        normalized_direction = [d / direction_magnitude for d in direction_to_obj2]
        dot_product = sum(obj1_vel[i] * normalized_direction[i] for i in range(3))
        
        return dot_product > 0.1
    
    def _get_object_description(self, obj_id: int, objects: List[Dict]) -> str:
        """Get object description"""
        for obj in objects:
            if obj['object_id'] == obj_id:
                return f"{obj['color']} {obj['material']} {obj['shape']}"
        return f"object {obj_id}"
    
    def _identify_objects_by_setting(self, trajectory: List[Dict], setting: SettingType) -> Dict[str, Any]:
        """
        Identify main objects and distractor objects based on setting type
        
        Args:
            trajectory: motion trajectory data
            setting: setting type
            
        Returns:
            Dict: dictionary containing main objects and distractor object IDs
        """
        first_frame = trajectory[0]['objects']
        
        # Identify initial state
        static_objects = []
        moving_objects = []
        
        for obj in first_frame:
            velocity_mag = self._calculate_velocity_magnitude(obj['velocity'])
            if velocity_mag < 0.01:  # Considered stationary
                static_objects.append(obj['object_id'])
            else:
                moving_objects.append(obj['object_id'])
        
        result = {
            'static_objects': static_objects,
            'moving_objects': moving_objects,
            'added_static': [],      # AS: Added Static objects
            'added_moving': []       # AM: Added Moving objects
        }
        
        # Assign objects based on scenario type and setting type
        scenario_name = self.scenario_name

        if setting == SettingType.ADD_ONE_STATIC:
            if scenario_name in ['bogus', 'early']:
                # Bogus and Early scenarios need 3 static objects, 2 moving objects (2 main + 1 distractor)
                if len(static_objects) >= 3 and len(moving_objects) >= 2:
                    result['added_static'] = [static_objects[2]]  # Third static object
                    result['static_objects'] = static_objects[:2]  # First two static objects
            else:
                print(len(static_objects), len(moving_objects))
                # Other scenarios need 2 static objects, 2 moving objects (1 main + 1 distractor)
                if len(static_objects) >= 2 and len(moving_objects) >= 2:
                    result['added_static'] = [static_objects[1]]  # Second static object
                    result['static_objects'] = [static_objects[0]]  # First static object
                
        elif setting == SettingType.ADD_TWO_STATIC:
            if scenario_name in ['bogus', 'early']:
                # Bogus and Early scenarios need 4 static objects, 2 moving objects (2 main + 2 distractors)
                if len(static_objects) >= 4 and len(moving_objects) >= 2:
                    result['added_static'] = static_objects[2:4]  # Third and fourth static objects
                    result['static_objects'] = static_objects[:2]  # First two static objects
            else:
                # Other scenarios need 3 static objects, 2 moving objects (1 main + 2 distractors)
                if len(static_objects) >= 3 and len(moving_objects) >= 2:
                    result['added_static'] = static_objects[1:3]  # Second and third static objects
                    result['static_objects'] = [static_objects[0]]  # First static object
                
        elif setting == SettingType.ADD_ONE_MOVING:
            if scenario_name == 'double':
                # Double scenario needs 1 static object, 4 moving objects
                if len(static_objects) >= 1 and len(moving_objects) >= 4:
                    result['added_moving'] = [moving_objects[3]]  # Fourth moving object
                    result['moving_objects'] = moving_objects[:3]  # First three moving objects
            else:
                # Other scenarios need 1 static object, 3 moving objects
                if len(static_objects) >= 1 and len(moving_objects) >= 3:
                    result['added_moving'] = [moving_objects[2]]  # Third moving object
                    result['moving_objects'] = moving_objects[:2]  # First two moving objects
                
        elif setting == SettingType.ADD_TWO_MOVING:
            if scenario_name == 'double':
                # Double scenario needs 1 static object, 5 moving objects
                if len(static_objects) >= 1 and len(moving_objects) >= 5:
                    result['added_moving'] = moving_objects[3:5]  # Fourth and fifth moving objects
                    result['moving_objects'] = moving_objects[:3]  # First three moving objects
            else:
                # Other scenarios need 1 static object, 4 moving objects
                if len(static_objects) >= 1 and len(moving_objects) >= 4:
                    result['added_moving'] = moving_objects[2:4]  # Third and fourth moving objects
                    result['moving_objects'] = moving_objects[:2]  # First two moving objects
        
        return result