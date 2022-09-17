from datasets import load_from_disk
from fire import Fire
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers, Tokenizer
from tokenizers.models import BPE


def dataset_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]['text']


def init_tokenizer():
    tokenizer = Tokenizer(
        models.BPE(continuing_subword_prefix='##', end_of_word_suffix='##')
    )
    normalizer = normalizers.Sequence([
        normalizers.Strip(), 
        normalizers.StripAccents(), 
        normalizers.Lowercase()
    ])
    pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Punctuation(),
        pre_tokenizers.WhitespaceSplit(), 
        pre_tokenizers.ByteLevel(use_regex=False)
    ])
    decoder = decoders.ByteLevel()
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.decoder = decoder
    return tokenizer


def init_trainer():
    return trainers.BpeTrainer(
        vocab_size=20000,
        show_progress=True,
        continuing_subword_prefix='##',
        # end_of_word_suffix='##',
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )


def train_tokenizer(batch_size=1000, dataset_size=None, save_file="tokenizer-wiki.json"):
    tokenizer = init_tokenizer()
    trainer = init_trainer()
    dataset = load_from_disk('cleaned_wiki_dataset')
    if dataset_size is None:
        dataset_size = len(dataset)
    dataset = dataset.select(range(dataset_size))
    tokenizer.train_from_iterator(
        dataset_iterator(dataset, batch_size=batch_size), 
        trainer, 
        length=dataset_size
    )
    tokenizer.save(save_file)


if __name__ == '__main__':
    Fire(train_tokenizer)
