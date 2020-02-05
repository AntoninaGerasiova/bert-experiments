import tensorflow as tf
import tensorflow_hub as hub

from bert import modeling

# English
BERT_VOCAB = 'uncased_L-12_H-768_A-12/vocab.txt'
BERT_INIT_CHKPNT = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
BERT_CONFIG = 'uncased_L-12_H-768_A-12/bert_config.json'
LOWER_CASE = True
MODULE_FOLDER = "bert-module-uncased_L-12_H-768_A-12"


# Russian
# BERT_VOCAB = 'rubert_cased_L-12_H-768_A-12_v2/vocab.txt'
# BERT_INIT_CHKPNT = 'rubert_cased_L-12_H-768_A-12_v2/bert_model.ckpt'
# BERT_CONFIG = 'rubert_cased_L-12_H-768_A-12_v2/bert_config.json'
# LOWER_CASE = False
# MODULE_FOLDER = "bert-module-rubert_cased_L-12_H-768_A-12_v2"


def build_module_fn(config_path, vocab_path, do_lower_case=True):
    def bert_module_fn(is_training):
        """Spec function for a token embedding module."""

        input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")

        bert_config = modeling.BertConfig.from_json_file(config_path)
        model = modeling.BertModel(config=bert_config, is_training=is_training,
                                   input_ids=input_ids)

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
            logits = tf.nn.bias_add(logits, output_bias)

        config_file = tf.constant(value=config_path, dtype=tf.string, name="config_file")
        vocab_file = tf.constant(value=vocab_path, dtype=tf.string, name="vocab_file")
        lower_case = tf.constant(do_lower_case)

        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, config_file)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_file)

        input_map = {"input_ids": input_ids}

        output_map = {"logits": logits}

        output_info_map = {"vocab_file": vocab_file,
                           "do_lower_case": lower_case}

        hub.add_signature(name="tokens", inputs=input_map, outputs=output_map)
        hub.add_signature(name="tokenization_info", inputs={}, outputs=output_info_map)

    return bert_module_fn


def create_module():
    tf.compat.v1.logging.set_verbosity(tf.logging.INFO)

    tags_and_args = []
    tags = set()
    tags_and_args.append((tags, dict(is_training=False)))
    module_fn = build_module_fn(BERT_CONFIG, BERT_VOCAB, LOWER_CASE)
    spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args)
    try:
        spec.export(MODULE_FOLDER,
                    checkpoint_path=BERT_INIT_CHKPNT)
    except tf.errors.AlreadyExistsError:
        tf.compat.v1.logging.warning("Path already exists")


if __name__ == "__main__":
    create_module()
