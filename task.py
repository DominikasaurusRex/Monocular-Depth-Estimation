#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile as directory_handler
import numpy as math_library
import tensorflow as tensorflow
from dataset import DataSet
from dataset import output_predictions_into_images
import model as original_model
import new_model as maurice_model
import train_operation as train_operation

MAX_EPOCH = 1000
LOG_DEVICE_PLACEMENT = False
PRINT_TENSORFLOW_VARIABLES = True
BATCH_SIZE = 8
TRAIN_FILE = "data/train.csv"
COARSE_DIR = "coarse_checkpoints"
REFINE_DIR = "refine_checkpoints"

REFINE_TRAIN = True
FINE_TUNE = True

USE_ORIGINAL_MODEL = True


def train():
    with tensorflow.Graph().as_default():
        dataset = DataSet(BATCH_SIZE)
        input_images, depth_maps, depth_maps_sigma = dataset.create_trainingbatches_from_csv(TRAIN_FILE) #rename variables
       
        #initialize tensorflow variablen 
        global_step = tensorflow.Variable(0, trainable=False) #why?
        keep_conv = tensorflow.placeholder(tensorflow.float32)
        keep_hidden = tensorflow.placeholder(tensorflow.float32)
        
        if REFINE_TRAIN:
            print("refine train.")
            if USE_ORIGINAL_MODEL:
                coarse = original_model.globalDepthMap(input_images, keep_conv, trainable=False)
                logits = original_model.localDepthMap(input_images, coarse, keep_conv, keep_hidden)
                loss = original_model.loss(logits, depth_maps, depth_maps_sigma)
                
                
                #logits, f3_d, f3, f2, f1_d, f1, pf1 = original_model.localDepthMap(images, coarse, keep_conv, keep_hidden)
                #o_p_logits = tensorflow.Print(logits, [logits], summarize=100)
                #o_p_f3_d = tensorflow.Print(f3_d, [f3_d], "fine3_dropout", summarize=100)
                #o_p_f3 = tensorflow.Print(f3, [f3], "fine3", summarize=100)
                #o_p_f2 = tensorflow.Print(f2, [f2], "fine2", summarize=100)
                #o_p_f1_d = tensorflow.Print(f1_d, [f1_d], "fine1_dropout", summarize=100)
                #o_p_f1 = tensorflow.Print(f1, [f1], "fine1", summarize=100)
                #o_p_pf1 = tensorflow.Print(pf1, [pf1], "pre_fine1", summarize=100)
            else:
                coarse = maurice_model.globalDepthMap(input_images, keep_conv, trainable=False)
                logits = maurice_model.localDepthMap(input_images, coarse, keep_conv, keep_hidden)
                loss = maurice_model.loss(logits, depth_maps, depth_maps_sigma)
        else:
            print("coarse train.")
            if USE_ORIGINAL_MODEL:
                logits = original_model.globalDepthMap(input_images, keep_conv, keep_hidden)
                loss = original_model.loss(logits, depth_maps, depth_maps_sigma)
            else:
                logits = maurice_model.globalDepthMap(input_images, keep_conv, keep_hidden)
                loss = maurice_model.loss(logits, depth_maps, depth_maps_sigma)
        train_op = train_operation.train(loss, global_step, BATCH_SIZE)
            
            
        # Tensorboard
        #merged = tf.summary.merge_all()
        writer = tensorflow.summary.FileWriter("/tmp/graph_data/train")
        
        # Initialize all Variables
        init_op = tensorflow.global_variables_initializer()

        # Session
        sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        writer.add_graph(sess.graph)
        sess.run(init_op)    

        # define saver
        coarse_params, refine_params = order_tensorflow_variables()
        saver_coarse = tensorflow.train.Saver(coarse_params)
        if REFINE_TRAIN:
            saver_refine = tensorflow.train.Saver(refine_params)
        # fine tune
        if FINE_TUNE:
            coarse_ckpt = tensorflow.train.get_checkpoint_state(COARSE_DIR)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if REFINE_TRAIN:
                print("trying to load models")
                refine_ckpt = tensorflow.train.get_checkpoint_state(REFINE_DIR)
                print(refine_ckpt)
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    print("Pretrained refine Model Loading.")
                    saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                    print("Pretrained refine Model Restored.")
                else:
                    print("No Pretrained refine Model.")

        # train
        coord = tensorflow.train.Coordinator()
        threads = tensorflow.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(MAX_EPOCH):
            index = 0
            for i in range(1000):
                _, loss_value, logits_val, images_val, depths_val = sess.run([train_op, loss, logits, input_images, depth_maps], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                
                #_, loss_value, logits_val, images_val, depths_val, _, _, = sess.run([train_op, loss, logits, images, depths, o_p_logits, o_p_f3_d], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                #_, loss_value, logits_val, images_val, summary = sess.run([train_op, loss, logits, images, merged], feed_dict={keep_conv: True, keep_hidden: True})
                #writer.add_summary(summary, step)
                if index % 100 == 0:
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                    assert not math_library.isnan(loss_value), 'Model diverged with loss = NaN'
                if index % 100 == 0:
                    if REFINE_TRAIN:
                        output_predictions_into_images(logits_val, images_val, depths_val, "data/predict_refine_%05d_%05d" % (step, i))
                    else:
                        output_predictions_into_images(logits_val, images_val, depths_val, "data/predict_%05d_%05d" % (step, i))
                index += 1

            if step % 5 == 0 or (step * 1) == MAX_EPOCH:
                if REFINE_TRAIN:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                else:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()
        
def order_tensorflow_variables():
    coarse_params = {}
    refine_params = {}
    if REFINE_TRAIN:
        for variable in tensorflow.global_variables():
            if variable.name.find("/") < 0 or variable.name.count("/") != 1:
                continue
            if variable.name.find('coarse') >= 0:
                coarse_params[variable.name] = variable
            if variable.name.find('fine') >= 0:
                refine_params[variable.name] = variable
    else:
        for variable in tensorflow.trainable_variables():
            if variable.name.find("/") < 0 or variable.name.count("/") != 1:
                continue
            if variable.name.find('coarse') >= 0:
                coarse_params[variable.name] = variable
            if variable.name.find('fine') >= 0:
                refine_params[variable.name] = variable
    print("Tensorflow Variables:")
    print(coarse_params)
    print(refine_params)
    return coarse_params, refine_params  


def main(args=None):
    createCheckpointDirectorys()
    train()


def createCheckpointDirectorys():
    if not directory_handler.Exists(COARSE_DIR):
        directory_handler.MakeDirs(COARSE_DIR)
    if not directory_handler.Exists(REFINE_DIR):
        directory_handler.MakeDirs(REFINE_DIR)


if __name__ == '__main__':
    tensorflow.app.run()
    