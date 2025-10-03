import os
import json
import glob
from datasets import Dataset, DatasetDict

def convert_videoac_to_hf_dataset(data_path, output_path):
    """Convert videoac data to HuggingFace dataset format"""
    
    documents = []
    
    # Load questions
    questions_dir = os.path.join(data_path, "questions")
    videos_dir = os.path.join(data_path, "videos")
    
    if not os.path.exists(questions_dir) or not os.path.exists(videos_dir):
        print(f"Required directories not found: {questions_dir}, {videos_dir}")
        return None
    
    # Get all question files
    question_files = glob.glob(os.path.join(questions_dir, "questions_*.json"))
    
    for question_file in question_files:
        try:
            with open(question_file, 'r') as f:
                data = json.load(f)
                
            video_filename = data["video_info"]["video_filename"]
            video_path = os.path.join(videos_dir, f"video_{question_file.split('_')[-1].replace('.json', '')}.mp4")
            
            # Create a document for each QA pair
            for i, qa in enumerate(data["qa_pairs"]):
                # Normalize different answer types and options
                answer_type = qa.get("answer_type", "")
                options = qa.get("options", []) or []
                choices = []
                answer = None

                try:
                    if answer_type == "yes_no":
                        # Yes/No questions
                        choices = [["A", "Yes"], ["B", "No"]]
                        raw_ans = qa.get("answer")
                        if raw_ans == "Yes":
                            answer = [["A"]]
                        elif raw_ans == "No":
                            answer = [["B"]]
                        else:
                            print("Error!")
                    elif answer_type == "multi_choice":
                        # Multiple choice questions
                        choices = [[chr(65 + j), str(option)] for j, option in enumerate(options)]  # A, B, C, ...
                        raw_ans = qa.get("answer")
                        answer = []
                        if isinstance(raw_ans, list):
                            for ans in raw_ans:
                                if isinstance(ans, int):
                                    answer.append([chr(65 + ans)])
                                else:
                                    print("Error!")
                        elif isinstance(raw_ans, dict):
                            for _, value in raw_ans.items():
                                value = [chr(65 + val) for val in value]
                                answer.append(value)
                        else:
                            print("Error!")
                        print(answer)
                    else:
                        print("Error!")
                except Exception as e:
                    print(f"Normalize error in {question_file} idx {i}: {e}. Skipping QA.")
                    continue

                # Final coercions to ensure pyarrow compatibility
                if answer is None:
                    answer = []

                question_text = qa.get("question", "")
                if not isinstance(question_text, str):
                    question_text = str(question_text)

                question_type = qa.get("question_type", "")
                question_rung = qa.get("question_rung", "")
                atype = answer_type if isinstance(answer_type, str) else str(answer_type)

                # Ensure choices is list of [str, str]
                safe_choices = []
                for pair in (choices or []):
                    try:
                        a, b = pair
                        safe_choices.append([str(a), str(b)])
                    except Exception:
                        # Skip malformed pair
                        continue

                doc = {
                    "id": f"{os.path.basename(question_file).replace('.json', '')}_{i}",
                    "question": question_text,
                    "answer": answer,
                    "video_path": video_path,
                    "question_type": str(question_type),
                    "question_rung": str(question_rung),
                    "answer_type": atype,
                    "n": i,
                    "choices": safe_choices,
                }
                documents.append(doc)
                
        except Exception as e:
            print(f"Error loading {question_file}: {e}")
            continue
    
    print(f"Loaded {len(documents)} documents")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(documents)
    
    # Create DatasetDict with validation split
    dataset_dict = DatasetDict({
        "valid": dataset
    })
    
    # Save to disk
    dataset_dict.save_to_disk(output_path)
    print(f"Saved dataset to {output_path}")
    
    return dataset_dict

if __name__ == "__main__":

    # Synthetic
    scenarios = ["overdetermination", "switch", "late", "early", "double", "bogus"]
    settings = ["basic", "add_one_static", "add_two_static", "add_one_moving", "add_two_moving"]
    for scenario in scenarios:
        for setting in settings:
            data_path = f".../data/synthetic/{scenario}/{setting}"
            output_path = f".../data/synthetic/{scenario}/{setting}_hf"
            os.makedirs(output_path, exist_ok=True)
            convert_videoac_to_hf_dataset(data_path, output_path)

    # Realistic
    scenarios = ["switch", "late", "bogus"]
    settings = ["basic"]
    for scenario in scenarios:
        for setting in settings:
            data_path = f".../data/realistic/{scenario}/{setting}"
            output_path = f".../data/realistic/{scenario}/{setting}_hf"
            os.makedirs(output_path, exist_ok=True)
            convert_videoac_to_hf_dataset(data_path, output_path)