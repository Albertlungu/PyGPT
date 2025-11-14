from datasets import load_dataset


def save_dolly(path):
    """
    Load the Databricks Dolly-15k dataset and save it in a formatted text file.

    Args:
        path (str): Output file path for the formatted dataset
    """
    print("Loading Databricks Dolly-15k dataset...")
    ds = load_dataset("databricks/databricks-dolly-15k")
    ds = ds['train']
    print(f"Loaded {len(ds)} examples")

    print(f"Writing to {path}...")
    with open(path, 'w', encoding='utf-8') as f:
        for idx, ex in enumerate(ds):
            # Create formatted entry
            text = f"Instruction:\n{ex['instruction']}\nContext:\n{ex['context']}\nResponse:\n{ex['response']}\n"
            f.write(text + "\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")


def load_text_file(path):
    """
    Load a text file for training.

    Args:
        path (str): Path to the text file

    Returns:
        list: List of text strings
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to separate examples
    examples = [ex.strip() for ex in content.split('\n\n') if ex.strip()]
    return examples


if __name__ == "__main__":
    save_dolly("training_data/pygpt_training_corpus.txt")