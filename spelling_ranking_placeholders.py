import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from bert import tokenization
from bert import modeling
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


class Model:
    def __init__(self, bert_config):
        self.X = tf.placeholder(tf.int32, [None, None])

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=self.X,
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


def get_score(sent, tokenizer, sess, model):
    tokens = tokenizer.tokenize(sent)
    input_ids = [tokens_to_masked_ids(tokenizer, tokens, i) for i in range(len(tokens))]
    print(input_ids)
    preds, logits, inp = \
        sess.run([tf.nn.softmax(model.logits), model.logits, model.X],
                 feed_dict={model.X: input_ids})
    print("input: ", inp)
    print("logits:", logits)
    print("softmax:", preds)


    print(preds.shape)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(tokens_ids)
    print([preds[i, i + 1, x] for i, x in enumerate(tokens_ids)])
    return np.prod([preds[i, i + 1, x] for i, x in enumerate(tokens_ids)])


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
    scores = [get_score(sent, tokenizer, sess, model) for sent in inputs]
    prob_scores = np.array(scores) / np.sum(scores)
    formatted_scores = list(zip(inputs, prob_scores))
    print(formatted_scores)



if __name__ == "__main__":
    get_scores()
