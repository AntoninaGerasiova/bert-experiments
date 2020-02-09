import collections

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


# constants
MASKED_TOKEN = "[MASK]"
# constant for now
SEQ_LEN = 40
MAX_PREDICTIONS_PER_SEQ = 20


class Model:
    def __init__(self, bert_config):
        self.bert_config = bert_config
        self.input_ids = \
            tf.placeholder(shape=[None, SEQ_LEN], dtype=tf.int32, name="input_ids")
        self.input_mask = \
            tf.placeholder(shape=[None, SEQ_LEN], dtype=tf.int32, name="input_mask")
        self.token_type = \
            tf.placeholder(shape=[None, SEQ_LEN], dtype=tf.int32, name="segment_ids")
        self.masked_lm_positions = \
            tf.placeholder(shape=[None, MAX_PREDICTIONS_PER_SEQ], dtype=tf.int32, name="masked_lm_positions")
        self.masked_lm_ids = \
            tf.placeholder(shape=[None, MAX_PREDICTIONS_PER_SEQ], dtype=tf.int32, name="masked_lm_ids")

        model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.token_type,
            use_one_hot_embeddings=False)

        self.input_tensor = model.get_sequence_output()
        self.output_weights = model.get_embedding_table()

        self.masked_lm_example_loss = self.get_masked_lm_output()
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        (assignment_map, initialized_variable_names) \
            = modeling.get_assignment_map_from_checkpoint(tvars, BERT_INIT_CHKPNT)
        tf.train.init_from_checkpoint(BERT_INIT_CHKPNT, assignment_map)
        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                      init_string)

    def get_masked_lm_output(self):
        self.input_tensor = self.gather_indexes()
        with tf.variable_scope("cls/predictions"):
            # We apply one more non-linear transformation before the output layer.
            # This matrix is not used after pre-training.
            with tf.variable_scope("transform"):
                self.input_tensor = tf.layers.dense(
                    self.input_tensor,
                    units=self.bert_config.hidden_size,
                    activation=modeling.get_activation(self.bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        self.bert_config.initializer_range))
                self.input_tensor = modeling.layer_norm(self.input_tensor)
            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            output_bias = tf.get_variable(
                "output_bias",
                shape=[self.bert_config.vocab_size],
                initializer=tf.zeros_initializer())
            logits = tf.matmul(self.input_tensor, self.output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            flat_masked_lm_ids = tf.reshape(self.masked_lm_ids, [-1])
            one_hot_labels = tf.one_hot(
                flat_masked_lm_ids,
                depth=self.bert_config.vocab_size,
                dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

            # TODO: dynamic gather from per_example_loss???
            loss = tf.reshape(per_example_loss, [-1, tf.shape(self.masked_lm_positions)[1]])
            return loss


    def gather_indexes(self):
        sequence_shape = modeling.get_shape_list(self.input_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(self.masked_lm_positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(self.input_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor


InputFeatures = collections.namedtuple('InputFeatures',
                                       ['input_ids',
                                        'input_mask',
                                        'segment_ids',
                                        'masked_lm_positions',
                                        'masked_lm_ids'])


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


def is_subtoken(x):
    return x.startswith("##")


def create_masked_lm_prediction(input_ids, mask_position, tokenizer, mask_count=1):
    masked_id = tokenizer.convert_tokens_to_ids([MASKED_TOKEN])[0]
    new_input_ids = input_ids[:]
    masked_lm_labels = []
    masked_lm_positions = list(range(mask_position, mask_position + mask_count))
    for i in masked_lm_positions:
        new_input_ids[i] = masked_id
        masked_lm_labels.append(input_ids[i])
    return new_input_ids, masked_lm_positions, masked_lm_labels


def create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids,
                           max_predictions_per_seq, tokenizer):
    """Mask each token/word sequentially"""
    features = []
    i = 1
    while i < len(input_tokens) - 1:
        mask_count = 1
        while is_subtoken(input_tokens[i + mask_count]):
            mask_count += 1

        input_ids_new, masked_lm_positions, masked_lm_labels = \
            create_masked_lm_prediction(input_ids, i, tokenizer, mask_count)

        masked_lm_positions = masked_lm_positions + \
                              [0]*(max_predictions_per_seq - len(masked_lm_positions))

        masked_lm_labels = masked_lm_labels + \
                            [0]*(max_predictions_per_seq - len(masked_lm_labels))

        assert len(masked_lm_positions) == max_predictions_per_seq
        assert len(masked_lm_labels) == max_predictions_per_seq
        feature = InputFeatures(
            input_ids=input_ids_new,
            input_mask=input_mask,
            segment_ids=segment_ids,
            masked_lm_positions=masked_lm_positions,
            masked_lm_ids=masked_lm_labels)
        features.append(feature)
        i += mask_count
    return features


def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    tokens = tokenizer.tokenize(example)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    input_tokens = tokens[:]
    input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens) + \
                (max_seq_length - len(input_tokens)) * [0]
    input_mask = [1] * len(input_tokens) + \
                 (max_seq_length - len(input_tokens)) * [0]
    segment_ids = [0] * max_seq_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # log first five sentences
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in input_tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    features = create_sequential_mask(input_tokens, input_ids, input_mask, segment_ids,
                                      MAX_PREDICTIONS_PER_SEQ, tokenizer)
    return features, input_tokens


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    all_features = []
    all_tokens = []
    for (ex_index, example) in enumerate(examples):
        features, tokens = convert_single_example(ex_index, example,
                                                  max_seq_length, tokenizer)
        all_features.extend(features)
        all_tokens.extend(tokens)
    return all_features, all_tokens

def features_to_vectors(features):
    input_ids = []
    input_mask = []
    segment_ids = []
    masked_lm_positions = []
    masked_lm_ids = []
    for i in range(len(features)):
        input_ids.append(features[i].input_ids)
        input_mask.append(features[i].input_mask)
        segment_ids.append(features[i].segment_ids)
        masked_lm_positions.append(features[i].masked_lm_positions)
        masked_lm_ids.append(features[i].masked_lm_ids)
    return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids



def parse_result(result, all_tokens, output_file=None):
    with tf.gfile.GFile(output_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        i = 0
        sentences = []
        for word_loss in result:
            # print(all_tokens)
            # print(word_loss)
            # start of a sentence
            if all_tokens[i] == "[CLS]":
                sentence = {}
                tokens = []
                sentence_loss = 0.0
                word_count_per_sent = 0
                i += 1

            # add token
            # print(i, all_tokens[i])
            # print(word_loss[0])
            tokens.append({"token": tokenization.printable_text(all_tokens[i]),
                           "prob": float(np.exp(-word_loss[0]))})
            # print(tokens)
            sentence_loss += word_loss[0]
            word_count_per_sent += 1
            i += 1

            token_count_per_word = 0
            while is_subtoken(all_tokens[i]):
                token_count_per_word += 1
                # print(i, all_tokens[i])
                # print(word_loss[token_count_per_word])
                tokens.append({"token": tokenization.printable_text(all_tokens[i]),
                               "prob": float(np.exp(-word_loss[token_count_per_word]))})
                sentence_loss += word_loss[token_count_per_word]
                i += 1

            # end of a sentence
            if all_tokens[i] == "[SEP]":
                print(tokens)
                print(sentence_loss,  word_count_per_sent)

                sentence["tokens"] = tokens
                sentence["ppl"] = float(np.exp(sentence_loss / word_count_per_sent))
                sentences.append(sentence)
                i += 1


def get_scores():
    tf.compat.v1.logging.set_verbosity(tf.logging.INFO)

    tokenization.validate_case_matches_checkpoint(LOWER_CASE, BERT_INIT_CHKPNT)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=BERT_VOCAB, do_lower_case=LOWER_CASE)
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

    inputs = read_examples(INPUT_FILE)
    features, all_tokens = convert_examples_to_features(inputs, SEQ_LEN, tokenizer)

    input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids = \
        features_to_vectors(features)
    print(masked_lm_ids)

    tf.reset_default_graph()
    sess = tf.Session()
    model = Model(bert_config)
    sess.run(tf.global_variables_initializer())

    losses = sess.run(model.masked_lm_example_loss,
                      feed_dict={model.input_ids: input_ids,
                                 model.input_mask: input_mask,
                                 model.token_type: segment_ids,
                                 model.masked_lm_positions: masked_lm_positions,
                                 model.masked_lm_ids: masked_lm_ids})

    parse_result(losses, all_tokens)




if __name__ == "__main__":
    get_scores()