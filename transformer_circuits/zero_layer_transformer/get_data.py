from datasets import load_dataset

def clean_wikipedia_formatting(ex, title_len_thresh=10):
    return {
        'text': ' '.join(filter(lambda text: len(text.split()) > title_len_thresh, ex['text'].split('\n')))
    }


if __name__ == '__main__':
    wiki_dataset = load_dataset("wikipedia", "20220301.en")
    wiki_dataset = wiki_dataset['train']
    wiki_dataset = wiki_dataset.map(clean_wikipedia_formatting, num_proc=32)
    wiki_dataset.save_to_disk('cleaned_wiki_dataset')
