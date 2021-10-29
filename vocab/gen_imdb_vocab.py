from allennlp.common.util import START_SYMBOL
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from datasets import load_dataset
from collections import Counter
import json
import io
import pathlib
from tqdm import tqdm


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


batch_size = 100
target_file = pathlib.Path('./vocab/vocab_imdb.json')

if __name__ == "__main__":
    ds = load_dataset('imdb', split="test")
    sentence_splitter = SpacySentenceSplitter(rule_based=0)
    tokenizer = SpacyTokenizer(start_tokens=[START_SYMBOL])
    c = Counter()
    cleaned = [t['text'].replace("<br>", " ").replace("<br />", " ") for t in ds]
    i = 0
    for batched_text in tqdm(batch(cleaned, n=batch_size), total=25000 // batch_size):
        txt = " ".join(batched_text)
        for lines in sentence_splitter.split_sentences(txt):
            # print(lines)
            tokens = [str(t).lower() for t in tokenizer.tokenize(lines)]
            c.update(tokens)
        i += 1
        if (i % 5) == 0:
            json.dump(dict(c.most_common()), io.open(target_file, 'w'), indent=4)
    json.dump(dict(c.most_common()), io.open(target_file, 'w'), indent=4)
