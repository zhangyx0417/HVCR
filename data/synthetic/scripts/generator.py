import json
import os
import re
from typing import List, Dict
from overdetermination import OverdeterminationScenarioQAGenerator
from switch import SwitchScenarioQAGenerator
from late import LateScenarioQAGenerator
from early import EarlyScenarioQAGenerator
from double import DoubleScenarioQAGenerator
from bogus import BogusScenarioQAGenerator
from base_generator import SettingType


BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..')

# Main QA generator class
class QAPairGenerator:
    """
    Main QA pair generator responsible for scenario assignment and file operations
    """
    
    def __init__(self, scenario_name: str, setting: str = None, **kwargs):
        self.valid_scenarios = {
            'bogus', 'double', 'early', 'late', 
            'overdetermination', 'switch', 'short'
        }
        
        if scenario_name not in self.valid_scenarios:
            raise ValueError(f"Invalid scenario name: {scenario_name}. "
                           f"Must be one of {self.valid_scenarios}")
        
        self.scenario_name = scenario_name
        self.setting = self._parse_setting(setting) if setting else None
        self.kwargs = kwargs
        
        # Scenario generator mapping
        self.scenario_generators = {
            'late': LateScenarioQAGenerator(),
            'bogus': BogusScenarioQAGenerator(),
            'double': DoubleScenarioQAGenerator(),
            'early': EarlyScenarioQAGenerator(),
            'overdetermination': OverdeterminationScenarioQAGenerator(),
            'switch': SwitchScenarioQAGenerator(),
        }
        
        self.scenario_generator = self.scenario_generators.get(scenario_name)
        if self.scenario_generator is None:
            raise NotImplementedError(f"Scenario '{scenario_name}' is not implemented yet")
        
        # Set scenario generator's setting
        if self.setting:
            self.scenario_generator.set_setting(self.setting)
    
    def _parse_setting(self, setting_str: str) -> SettingType:
        """Parse setting string to SettingType enum."""
        setting_mapping = {
            'add_one_static': SettingType.ADD_ONE_STATIC,
            'add_two_static': SettingType.ADD_TWO_STATIC,
            'add_one_moving': SettingType.ADD_ONE_MOVING,
            'add_two_moving': SettingType.ADD_TWO_MOVING,
        }
        
        if setting_str not in setting_mapping:
            return None
        
        return setting_mapping[setting_str]
    
    def _load_simulation_files(self, scenario_path: str) -> List[Dict]:
        """Load simulation data files in order."""
        simulation_files = []

        # Determine correct simulation directory based on setting type
        if self.setting:
            # Read from setting-specific directory
            simulation_dir = os.path.join(BASE_DIR, "data", "synthetic", self.scenario_name, self.setting.value, "simulations")
        else:
            # Read from basic directory (backward compatible)
            simulation_dir = scenario_path

        if not os.path.exists(simulation_dir):
            print(f"Warning: Simulation directory {simulation_dir} does not exist")
            return simulation_files

        for filename in os.listdir(simulation_dir):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(simulation_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    simulation_files.append(data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
                
        return simulation_files
    
    def _save_qa_pairs_to_json(self, qa_data: Dict, scenario: str, scene_index: int) -> bool:
        """Generic QA pair saving function."""
        # Determine output path based on setting type
        if self.setting:
            output_dir = os.path.join(BASE_DIR, "data", "synthetic", scenario, self.setting.value, "questions")
        else:
            output_dir = os.path.join(BASE_DIR, "data", "synthetic", scenario, "basic", "questions")
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"questions_{scene_index:05d}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
            print(f"Saved QA pairs for scene {scene_index} to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving QA pairs to {output_path}: {e}")
            return False
    
    def generate_qa_pairs(self) -> List[Dict]:
        """Generate QA pairs."""
        if not self.setting:
            scenario_path = os.path.join(BASE_DIR, "data", "synthetic", self.scenario_name, "basic", "simulations")
        else:
            scenario_path = os.path.join(BASE_DIR, "data", "synthetic", self.scenario_name, self.setting.value, "simulations")
        simulation_files = self._load_simulation_files(scenario_path)
        
        all_qa_pairs = []
        
        for data in simulation_files:
            # Use corresponding scenario generator to analyze scenario, pass setting parameters
            analysis = self.scenario_generator.analyze_scenario(data, self.setting)
            
            # Generate QA pairs
            qa_pairs = self.scenario_generator.generate_qa_templates(data, analysis)
            
            # Generate causal reasoning templates
            cr_templates = self.scenario_generator.generate_cr_templates(data, analysis)

            if not qa_pairs:
                continue
            
            video_filename = data["video_filename"]
            match = re.search(r'(\d{2}\.mp4)$', video_filename)
            if match:
                video_filename = match.group(1)

            video_idx = int(video_filename.split(".")[0])
            video_filename = f"video_{video_idx:05d}.mp4"
            
            # print(video_filename, qa_pairs[4]["question"])

            # Create video information
            video_info = {
                'scene_index': data['scene_index'],
                'video_filename': video_filename,
                'scenario': self.scenario_name,
                'setting': self.setting.value if self.setting else "basic",
                # 'analysis_result': analysis,
                'total_frames': len(data['motion_trajectory']),
            }

            # print(video_info)
            
            # Create complete QA data structure
            video_qa_data = {
                'video_info': video_info,
                'qa_pairs': qa_pairs,
                'causal_graph': cr_templates.get('causal_graph'),
                'twin_network': cr_templates.get('twin_network'),
                'generation_metadata': {
                    'scenario': self.scenario_name,
                    'setting': self.setting.value if self.setting else "basic",
                    'total_questions': len(qa_pairs),
                    'question_types': list(set([qa['question_type'] for qa in qa_pairs])),
                    'answer_types': list(set([qa['answer_type'] for qa in qa_pairs]))
                }
            }
            
            # Save QA pairs to JSON file
            self._save_qa_pairs_to_json(video_qa_data, self.scenario_name, data['scene_index'])
            
            # Add video information to return value
            for qa in qa_pairs:
                qa['video_info'] = video_info
            all_qa_pairs.extend(qa_pairs)
        
        print(f"Generated and saved QA pairs for {len(simulation_files)} videos in {self.scenario_name} scenario")
        if self.setting:
            print(f"Setting: {self.setting.value}")
        print(f"Total QA pairs generated: {len(all_qa_pairs)}")
        
        return all_qa_pairs
    
    def get_scenario_info(self) -> Dict:
        """Get current scenario information."""
        return {
            'scenario_name': self.scenario_name,
            'setting': self.setting.value if self.setting else None,
            'valid_scenarios': list(self.valid_scenarios),
            'implemented_scenarios': list(self.scenario_generators.keys()),
            'valid_settings': [s.value for s in SettingType]
        }

# Usage example
if __name__ == "__main__":
    
    for scenario in ['overdetermination', 'switch', 'late', 'early', 'double', 'bogus']:
        for setting in ['basic', 'add_one_static', 'add_two_static', 'add_one_moving', 'add_two_moving']:
            generator = QAPairGenerator(scenario, setting)
            print(f"Generating QA pairs for scenario: {scenario}")
            if setting:
                print(f"Setting: {setting}")
            qa_pairs = generator.generate_qa_pairs()
            print("Done!")
