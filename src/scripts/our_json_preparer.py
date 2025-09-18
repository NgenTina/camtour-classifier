import json
import pandas as pd
import os

def prepare_tourism_data_simple(file_paths):

    questions = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    messages = data.get('messages', [])
                    # Extract user questions
                    for msg in messages:
                        if msg.get('role') == 'user':
                            questions.append({
                                'question': msg.get('content', ''),
                                'is_tourism': 1
                            })
                except json.JSONDecodeError:
                    continue

    # Create DataFrame
    df = pd.DataFrame(questions)
    return df


# Example usage
TEST_PATH = r'D:\Codes\Websites\camtour\classifier\dataset\original_data'
file_paths = [os.path.join(TEST_PATH, '_part_1.jsonl')]
tourism_df = prepare_tourism_data_simple(file_paths)
print(f"Extracted {len(tourism_df)} tourism questions")

# Save for evaluation
tourism_df.to_csv(os.path.join(TEST_PATH, 'tourism_questions_for_eval.csv'), index=False)
