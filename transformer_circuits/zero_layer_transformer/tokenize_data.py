from datasets import load_from_disk
from fire import Fire
from transformers import PreTrainedTokenizerFast


def tokenize_data(context_length=16, num_proc=32, save_file='tokenized_wiki_dataset'):
    dataset = load_from_disk('cleaned_wiki_dataset')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='tokenizer-wiki.json')

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_dataset = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names, num_proc=num_proc
    )
    tokenized_dataset.save_to_disk(save_file)


if __name__ == '__main__':
    Fire(tokenize_data)
