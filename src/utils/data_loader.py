from datasets import load_dataset


def save_dolly(path):
    """
    Load the Databricks Dolly-15k dataset and save it in a formatted text file.

    Args:
        path (str): Output file path for the formatted dataset
    """
    print("Loading Databricks Dolly-15k dataset...")
    dataset_len = 0
    ds = load_dataset("databricks/databricks-dolly-15k")
    ds = ds['train']
    ds = ds.select(range(dataset_len)) if dataset_len != 0 else ds
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

def save_general_knowledge(path):
    """
    Load the MuskumPillerum/General-Knowledge dataset from HuggingFace

    Args:
        path (str): Output file path for the formatted dataset
    """
    print("Loading dataset:")
    dataset_len = 10000
    ds = load_dataset("MuskumPillerum/General-Knowledge")
    print(ds)
    ds = ds['train']
    ds = ds.select(range(dataset_len)) if dataset_len > 0 else ds
    print(f"Loaded {dataset_len} examples")

    with open(path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            text = f"Instruction:\n{ex['Question']}\nResponse:\n{ex['Answer']}\n"
            f.write(text + "\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")

def save_alpaca(path):
    """
    Load the Alpaca dataset from HuggingFace

    Args:
        path (str): Output file path for the formatted dataset
    """
    ds_len = 0
    ds = load_dataset("tatsu-lab/alpaca")
    ds = ds['train']
    ds = ds.select(range(ds_len)) if ds_len > 0 else ds
    with open(path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            text = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput: {ex['output']}\n"
            f.write(text + "\n\n")

            if (idx + 1) % 1000 == 0:
                    print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")

def save_trivia_qa(path):
    ds_len = 20000
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")
    ds = ds['train']
    ds = ds.select(range(ds_len)) if ds_len > 0 else ds

    with open(path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            # Check if 'answer' is a dict with 'value' and 'aliases'
            answer_info = ex['answer']
            if isinstance(answer_info, dict):
                main_answer = answer_info.get('value', '')
                aliases = answer_info.get('aliases', [])
                aliases_str = ', '.join(aliases) if aliases else 'N/A'
            else:
                main_answer = answer_info
                aliases_str = 'N/A'

            text = f"Question:\n{ex['question']}\nAnswer:\n{main_answer}\nAliases:\n{aliases_str}\n"
            f.write(text + "\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")

def save_flan(path, ds_len):
    ds = load_dataset("Open-Orca/FLAN")

def save_general_instruct(path, ds_len):
    skipped_count = 0
    saved_count = 0
    ds = load_dataset("teknium/GPTeacher-General-Instruct")['train']
    ds = ds.select(range(ds_len)) if ds_len > 0 else ds

    with open(path, "w", encoding='utf-8') as f:
        for idx, ex in enumerate(ds):
            response = ex['response'].strip()

            if not response or response.lower in ['<nooutput>', '<no output>', 'none', '']:
                skipped_count += 1
                continue

            text = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput: {response}\n"
            saved_count += 1
            f.write(text + "\n\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx+1}/{ds_len} examples")

    print(f"Successfully saved {saved_count} examples to {path}")
    print(f"Skipped {skipped_count} examples")

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
    # save_dolly("training_data/pygpt_training_corpus.txt")
    # save_general_knowledge("training_data/general_knowledge.txt")
    save_alpaca("training_data/alpaca.txt")
    # save_trivia_qa("training_data/trivia.txt")
    pass