import os
import json
import glob
from datasets import load_from_disk


def videoac_get_documents(dataset_path):
    """Load documents from the dataset path"""
    try:
        # First try to load as HuggingFace dataset
        dataset = load_from_disk(dataset_path)
        return dataset
    except Exception as e:
        print(f"Trying to load custom format from {dataset_path}")
        # Try to load custom format
        return load_custom_videoac_data(dataset_path)


def load_custom_videoac_data(dataset_path):
    """Load custom videoac data format"""
    documents = []
    
    # Load questions
    questions_dir = os.path.join(dataset_path, "questions")
    videos_dir = os.path.join(dataset_path, "videos")
    
    if not os.path.exists(questions_dir) or not os.path.exists(videos_dir):
        print(f"Required directories not found: {questions_dir}, {videos_dir}")
        return []
    
    # Get all question files
    question_files = glob.glob(os.path.join(questions_dir, "questions_*.json"))
    
    for question_file in question_files:
        try:
            with open(question_file, 'r') as f:
                data = json.load(f)
                
            video_filename = data["video_info"]["video_filename"]
            video_path = os.path.join(videos_dir, f"video_{question_file.split('_')[-1].replace('.json', '')}.mp4")
            
            # Create a document for each QA pair
            for qa in data["qa_pairs"]:
                doc = {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "video_path": video_path,
                    "question_type": qa["question_type"],
                    "question_rung": qa["question_rung"],
                    "answer_type": qa["answer_type"]
                }
                documents.append(doc)
                
        except Exception as e:
            print(f"Error loading {question_file}: {e}")
            continue
    
    print(f"Loaded {len(documents)} documents from custom format")
    return documents


def videoac_doc_to_visual(doc):
    """Convert document to visual input (video path)"""
    if "video_path" in doc:
        return [doc["video_path"]]
    elif "video" in doc:
        return [doc["video"]]
    else:
        # Try to find video field in document
        for key in doc.keys():
            if "video" in key.lower() or "clip" in key.lower():
                return [doc[key]]
        return []


def videoac_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Convert document to text prompt"""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    # Get question from document
    question = doc.get("question", "")
    if not question:
        # Try alternative field names
        for key in ["text", "prompt", "query"]:
            if key in doc:
                question = doc[key]
                break
    
    # Handle choices for multiple choice questions
    choices_text = ""
    if "choices" in doc:
        choices = doc["choices"]
        for choice_letter, choice_text in choices:
            choices_text += f"({choice_letter}) {choice_text}\n"
    
    # Get pre and post prompts from config
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    
    if choices_text:
        return f"{pre_prompt}Question: {question}\nOptions:\n{choices_text}{post_prompt}"
    else:
        return f"{pre_prompt}{question}{post_prompt}"


def videoac_process_results(doc, results):
    """Process model results for evaluation"""
    # Extract the answer from results
    if isinstance(results, list):
        result = results[0] if results else ""
    else:
        result = str(results)
    
    # Extract answer letters from the new format: "Answer: A B C END"
    answer_letters = []
    if "Answer:" in result:
        # Extract the letters after "Answer:"
        parts = result.split("Answer:")
        if len(parts) > 1:
            answer_part = parts[1].strip().replace(" END", "")  # Remove the "END" part
            # Find all capital letters (A, B, C, D, E, F, G, H, I, J, ...)
            answer_letters = [char for char in answer_part.split() if char.isupper() and char.isalpha()]
    
    # Validate against available choices if provided
    if answer_letters and "choices" in doc and isinstance(doc["choices"], list) and len(doc["choices"]) > 0:
        valid_letters = [chr(65 + i) for i in range(len(doc["choices"]))]  # [A, B, C, ...]
        answer_letters = [letter for letter in answer_letters if letter in valid_letters]
    
    # Handle both single choice and multiple choice answers
    expected = doc.get("answer")
    
    # Convert expected to list of lists if necessary (support for the new format [['A']] or [['A', 'B']])
    if isinstance(expected, str):
        # Try to parse if it looks like a list string
        if expected.startswith('[') and expected.endswith(']'):
            try:
                import ast
                expected = ast.literal_eval(expected)
            except:
                expected = [[expected]]
        else:
            expected = [[expected]]
    elif expected is None:
        expected = []
    
    # Initialize accuracy list
    accuracy = []
    
    # Calculate accuracy for multiple choice (strict match: all correct and no extras)
    if expected and answer_letters:
        # Convert to sets for comparison (order doesn't matter)
        expected_sets = [set(str(e).upper() for e in group) for group in expected]
        answer_set = set(letter.upper() for letter in answer_letters)
        
        # For "actual_cause" type questions, we expect four sets of expected answers
        question_type = doc.get("question_type")
        
        if question_type == "actual_cause" and len(expected_sets) == 4:
            # Ensure we have 4 accuracy checks (one for each HP, BV, DBV, Boc)
            for exp_set in expected_sets:
                accuracy.append(1 if exp_set == answer_set else 0)
        else:
            # For other question types, just compare to the first expected set
            if expected_sets[0] == answer_set:
                accuracy.append(1)
            else:
                accuracy.append(0)
    
    # Prepare the response with additional information
    result_data = {
        "answer": answer_letters,
        "raw_result": result,
        "accuracy": accuracy,
        "question": doc.get("question"),
        "question_rung": doc.get("question_rung"),
        "question_type": doc.get("question_type"),
        "options": doc.get("choices")
    }
    
    return result_data


def videoac_accuracy(items):
    """Aggregate accuracy: average per-sample accuracy.
    items can be:
      - per-sample numeric values 0/1
      - per-sample dictionaries containing 'accuracy' key
    """
    if not items:
        return 0.0
    
    total = 0.0
    count = 0
    
    for item in items:
        if isinstance(item, (int, float)):
            # If item is a single number (0 or 1), add directly
            total += float(item)
            count += 1
        elif isinstance(item, dict):
            acc_list = item.get("accuracy")
            if acc_list is not None and isinstance(acc_list, list):
                # For per-sample accuracy lists
                if len(acc_list) > 1:
                    # If accuracy list has multiple elements, check if any is 1 (correct)
                    total += 1 if 1 in acc_list else 0
                else:
                    # If only one element, add directly
                    total += float(acc_list[0])
                count += 1
    
    if count == 0:
        return 0.0
    
    # Calculate average accuracy
    return total / count
