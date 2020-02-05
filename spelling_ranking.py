import tensorflow as tf
import tensorflow_hub as hub

from bert import tokenization
from bert import modeling

# English
BERT_VOCAB = 'uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = 'uncased_L-12_H-768_A-12/bert_config.json'
LOWER_CASE = True
INPUT_FILE = 'data/test.en.tsv'


# Russian
# BERT_VOCAB = 'rubert_cased_L-12_H-768_A-12_v2/vocab.txt'
# BERT_INIT_CHKPNT = 'rubert_cased_L-12_H-768_A-12_v2/bert_model.ckpt'
# BERT_CONFIG = 'rubert_cased_L-12_H-768_A-12_v2/bert_config.json'
# LOWER_CASE = False
# INPUT_FILE = 'data/test3.ru.tsv'


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            examples.append(line)

    return examples


def tokens_to_masked_ids(tokenizer, tokens, mask_ind):
    masked_tokens = tokens[:]
    masked_tokens[mask_ind] = "[MASK]"
    masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids




def get_scores():

    tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=BERT_VOCAB, do_lower_case=LOWER_CASE)

    inputs = read_examples(INPUT_FILE)
    print(inputs)

    tokens_to_masked_ids(tokenizer, )




if __name__ == "__main__":
    get_scores()
