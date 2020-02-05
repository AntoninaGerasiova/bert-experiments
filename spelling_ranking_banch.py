import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from bert import tokenization
from bert import modeling
from bert import extract_features

InputExample = extract_features.InputExample
import numpy as np

# English
# BERT_VOCAB = 'uncased_L-12_H-768_A-12/vocab.txt'
# BERT_INIT_CHKPNT = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
# BERT_CONFIG = 'uncased_L-12_H-768_A-12/bert_config.json'
# LOWER_CASE = True
# INPUT_FILE = 'data/test.en.tsv'

# Russian
BERT_VOCAB = 'rubert_cased_L-12_H-768_A-12_v2/vocab.txt'
BERT_INIT_CHKPNT = 'rubert_cased_L-12_H-768_A-12_v2/bert_model.ckpt'
BERT_CONFIG = 'rubert_cased_L-12_H-768_A-12_v2/bert_config.json'
LOWER_CASE = False
INPUT_FILE = 'data/test3.ru.tsv'

# constant for now
SEQ_LEN = 20


class Model:
    def __init__(self, bert_config):
        self.input_ids = \
            tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
        self.input_mask = \
            tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
        self.token_type = \
            tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            # token_type_ids=self.token_type,
            use_one_hot_embeddings=False)

        output_layer = model.get_sequence_output()
        embedding = model.get_embedding_table()
        with tf.variable_scope('cls/predictions'):
            with tf.variable_scope('transform'):
                input_tensor = tf.layers.dense(
                    output_layer,
                    units=bert_config.hidden_size,
                    activation=modeling.get_activation(bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        bert_config.initializer_range
                    ),
                )
                input_tensor = modeling.layer_norm(input_tensor)

            output_bias = tf.get_variable(
                'output_bias',
                shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer(),
            )
            logits = tf.matmul(input_tensor, embedding, transpose_b=True)
            self.logits = tf.nn.bias_add(logits, output_bias)


def sentence_to_masked_msents(sent):
    words = sent.split()
    masked_sentences = []
    for i in range(len(words)):
        masked_words = words[:i] + list("[MASK]") + words[i + 1:]
        yield " ".join(masked_words)
        # masked_sentences.append(" ".join(masked_words))
    # return masked_sentences


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


def get_all_tokens(str_list, tokenizer, seq_len):
    not_masked_ids = []
    all_input_ids = []
    all_input_masks = []
    all_segment_ids = []
    for line in str_list:
        tokens = tokenizer.tokenize(line)
        not_masked_ids.append(tokenizer.convert_tokens_to_ids(tokens))
        for mask_ind in range(len(tokens)):
            masked_tokens = tokens_to_masked_ids(tokenizer, tokens, mask_ind)
            all_input_ids.append(masked_tokens + (seq_len - len(masked_tokens)) * [0])
            input_mask = len(masked_tokens) * [1] + (seq_len - len(masked_tokens)) * [0]
            all_input_masks.append(input_mask)
            all_segment_ids.append([0] * seq_len)

    return not_masked_ids,\
           [all_input_ids, all_input_masks, all_segment_ids]


def tokens_to_masked_ids(tokenizer, tokens, mask_ind):
    masked_tokens = tokens[:]
    masked_tokens[mask_ind] = "[MASK]"
    masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids


def get_scores():
    tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
    tokenization.validate_case_matches_checkpoint(LOWER_CASE, BERT_INIT_CHKPNT)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=BERT_VOCAB, do_lower_case=LOWER_CASE)
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

    # for v in tf.train.list_variables(BERT_INIT_CHKPNT):
    #     print(v)

    tf.reset_default_graph()
    sess = tf.Session()
    model = Model(bert_config)
    tvars = tf.trainable_variables()
    print(len(tvars))
    (assignment_map, initialized_variable_names
     ) = modeling.get_assignment_map_from_checkpoint(tvars, BERT_INIT_CHKPNT)
    tf.train.init_from_checkpoint(BERT_INIT_CHKPNT, assignment_map)
    sess.run(tf.global_variables_initializer())

    inputs = read_examples(INPUT_FILE)
    # inputs = [inputs[0]]

    not_masked_ids, arrays = \
        get_all_tokens(inputs, tokenizer, SEQ_LEN)
    print(arrays[0])
    preds, logits, inp = sess.run([tf.nn.softmax(model.logits), model.logits, model.input_ids],
                     feed_dict={model.input_ids: arrays[0],
                                model.input_mask: arrays[1],
                                model.token_type: arrays[2]})

    print("input:", inp)
    print("logits:", logits)
    print("softmax:", preds)
    print(preds.shape)
    first_index = 0
    sent_probs = []
    for ids in not_masked_ids:
        print(ids)
        sent_preds = preds[first_index:first_index + len(ids), :, :]
        word_probs = [sent_preds[i, i+1, x] for i, x in enumerate(ids)]
        print(word_probs)
        sent_prob = np.prod(word_probs)
        sent_probs.append(sent_prob)
        first_index += len(ids)

    print(list(zip(inputs, sent_probs)))
    probs = np.array(sent_probs)/sum(sent_probs)
    print(list(zip(inputs, probs)))



if __name__ == "__main__":
    get_scores()
