import tensorflow as tf

def convert_tflite(save_mode_dir):
    #Way 1
    # converter = tf.lite.TFLiteConverter.from_saved_model(save_mode_dir)
    # tflite_model = converter.convert()
    #
    # #Save model
    # with open('../model/text_detection_model.tflite', 'wb') as f:
    #     f.write(tflite_model)

    #Way 2
    # converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    #     graph_def_file='../model/frozen_east_text_detection.pb',
    #     input_arrays=['input_images'],
    #     output_arrays=['feature_fusion/concat_3']
    # )
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # tflite_model = converter.convert()
    # with tf.io.gfile.GFile('../model/text_detection_model.tflite', 'wb') as f:
    #     f.write(tflite_model)

    #Way 3


def get_infor(saved_model_dir: str):
    # Way 1
    # gf = tf.compat.v1.GraphDef()
    # m_file = open(saved_model_dir, 'rb')
    # gf.ParseFromString(m_file.read())
    #
    # for b in gf.node:
    #     print(b.name)

    #Way 2
    with tf.compat.v1.gfile.GFile(saved_model_dir, 'rb') as f:
        grapdef = tf.compat.v1.GraphDef()
        grapdef.ParseFromString(f.read())

    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(grapdef, name='prefix')

    for op in graph.get_operations():
        abs = graph.get_tensor_by_name(op.name + ":0")
        print(abs)

if __name__ == '__main__':
    # convert_tflite('../model/frozen_east_text_detection.pb')
    get_infor('../model/frozen_east_text_detection.pb')


