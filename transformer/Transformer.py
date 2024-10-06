import time as time

import tensorflow as tf
from keras import backend as backend
from keras import optimizers as optimizers

from transformer_architecture.optimzers.TensorflowCustomSchedule import TensorflowCustomSchedule
from transformer_architecture.transformer.LossAndAccuracyCalculator import create_loss_object, \
    generate_train_loss_object, generate_train_accuracy_object
from transformer_architecture.transformer.TensorFlowTransformer import TensorFlowTransformer


class Transformer:
    '''
    For d_model variable, the recommended value is 512. We are set 128.
    For nb_layers variable, the recommended value is 6 layers. We are set 4.
    For ffn_units variable, the recommended value is 2048. We are set 512.
    All the information above is to run in a local environment and quickly.
    '''

    d_model = 128
    nb_layers = 4
    ffn_units = 512
    nb_proj = 8
    dropout_rate = 0.1

    @classmethod
    def training_model(cls, inputs, outputs, dataset):
        backend.clear_session()
        transformer = TensorFlowTransformer(vocab_size_enc=inputs,
                                            vocab_size_dec=outputs,
                                            d_model=Transformer.d_model,
                                            nb_layers=Transformer.nb_layers,
                                            FFN_units=Transformer.ffn_units,
                                            nb_proj=Transformer.nb_proj,
                                            dropout_rate=Transformer.dropout_rate)
        # Calculating loss and accuracy
        loss_object = create_loss_object()
        train_loss = generate_train_loss_object()
        train_accuracy = generate_train_accuracy_object()
        '''
        5.3 Optimizer
        We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and  = 10−9
        . We varied the learning rate over the course of training, according to the formula:
        lrate = d−0.5model · min(step_num−0.5, step_num · warmup_steps−1.5) (3)
        This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
        and decreasing it thereafter proportionally to the inverse square root of the step number. We used
        warmup_steps = 4000.
        '''
        l_rate = TensorflowCustomSchedule(d_model=Transformer.d_model)
        optimizer = optimizers.Adam(learning_rate=l_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        checkpoint_path = '/utils/files/output'
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, transformer=transformer)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)
        if checkpoint_manager.latest_checkpoint: checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print('Latest checkpoint restored.')
        epochs = 10
        for epoch in range(epochs):
            print(f'Starting epoch {epoch + 1}')
            start = time.time()

            train_loss.reset_state()
            train_accuracy.reset_state()

            for (batch, (input_en_data_tokenized, targets)) in enumerate(dataset):
                dec_inputs_shifted_right = targets[:, :-1]
                dec_outputs_real = targets[:, :1]

                with tf.GradientTape() as tape:
                    predictions = transformer(dec_inputs_shifted_right, dec_outputs_real, training=True)
                    loss = loss_object(dec_outputs_real, dec_outputs_real)

                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
                train_loss(loss)
                train_accuracy(dec_outputs_real, predictions)

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(),
                                                                                 train_accuracy.result()))

                checkpoint_save_path = checkpoint_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, checkpoint_save_path))
                print('Time taken for 1 epoch {} secs\n'.format(time.time() - start))










