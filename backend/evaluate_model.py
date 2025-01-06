from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def normalize_text(text):
    """Normalize text to handle Turkish characters and whitespace"""
    if pd.isna(text):
        return ""
    return str(text).strip()

def evaluate_saved_model(model_path, test_data_path, batch_size=32, max_length=512):
    # Load the saved model and tokenizer
    print("Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Load test data with UTF-8-SIG encoding
    print("Loading test data...")
    test_data = pd.read_csv(test_data_path, encoding='utf-8-sig')

    # Normalize input and output text
    test_data['instruction_turkish'] = test_data['instruction_turkish'].apply(normalize_text)
    test_data['response_turkish'] = test_data['response_turkish'].apply(normalize_text)

    results = []
    total = 0
    correct = 0

    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch_data = test_data.iloc[i:i + batch_size]

        # Tokenize inputs
        inputs = tokenizer(
            batch_data['instruction_turkish'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        # Tokenize actual outputs for comparison
        actual_outputs = batch_data['response_turkish'].tolist()

        with torch.no_grad():
            # Generate predictions
            generated = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=4,
                length_penalty=1.0
            )

            # Decode predictions
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)

            # Compare and store results
            for pred, actual, instruction in zip(predictions, actual_outputs, batch_data['instruction_turkish']):
                total += 1
                # Normalize both prediction and actual text for comparison
                pred_normalized = normalize_text(pred)
                actual_normalized = normalize_text(actual)
                is_correct = pred_normalized == actual_normalized
                if is_correct:
                    correct += 1

                results.append({
                    'instruction': instruction,
                    'prediction': pred_normalized,
                    'actual': actual_normalized,
                    'correct': is_correct
                })

        # Clear memory after each batch
        clear_gpu_memory()

    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0

    # Save results to CSV with UTF-8-SIG encoding
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_evaluation_results.csv', index=False, encoding='utf-8-sig')

    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    # Print some example predictions
    print("\nExample Predictions:")
    for i in range(min(5, len(results))):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {results[i]['instruction']}")
        print(f"Predicted: {results[i]['prediction']}")
        print(f"Actual: {results[i]['actual']}")
        print(f"Correct: {results[i]['correct']}")

    # Add some analysis of partial matches
    print("\nPartial Match Analysis:")
    word_matches = 0
    total_words = 0

    for result in results:
        pred_words = set(result['prediction'].split())
        actual_words = set(result['actual'].split())
        word_matches += len(pred_words.intersection(actual_words))
        total_words += len(actual_words)

    if total_words > 0:
        word_match_rate = word_matches / total_words
        print(f"Word-level match rate: {word_match_rate:.4f}")

# Usage example:
if __name__ == "__main__":
    MODEL_PATH = '/content/drive/MyDrive/model training/deneme 3 full 10 epoch/results'  # Path to your saved model
    TEST_DATA_PATH = '/content/drive/MyDrive/processed_training_turkish_first5k.csv'  # Path to your test data

    evaluate_saved_model(MODEL_PATH, TEST_DATA_PATH)