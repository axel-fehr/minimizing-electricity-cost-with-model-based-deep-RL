"""This file contains a class of a Bayesian neural network that acts as a state transition model in a reinforcement learning system."""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pickle

class StateTransitionModel:
    """Class that creates and trains a Bayesian neural network to approximate the state transtions."""

    def __init__(self, neurons_in_each_layer):
        self.neurons_in_each_layer = neurons_in_each_layer
        self.model = self._create_nn()
        self.output_stds = 0.01 * np.ones([1,self.neurons_in_each_layer[-1]], dtype=np.float32)
        num_GPUs = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=num_GPUs)

    def _create_nn(self):
        """Creates a Bayesian neural network and returns the Keras model."""
        with tf.name_scope('state_transition_model'):
            model = tf.keras.Sequential(name='state_transition_model')

            for layer_idx, num_neurons in enumerate(self.neurons_in_each_layer):
                is_output_layer = (layer_idx == len(self.neurons_in_each_layer) - 1)
                if is_output_layer:
                    model.add(tfp.layers.DenseReparameterization(num_neurons))
                elif layer_idx == 0:
                    model.add(tfp.layers.DenseReparameterization(num_neurons, input_shape=(None, num_neurons)))
                else:
                    model.add(tfp.layers.DenseReparameterization(num_neurons, activation=tf.nn.relu))

        return model

    def train(self, x_data, y_data, num_epochs, pricing_model, save_weights_path='./stored_data/weights/state_transition_model/state_transition_model',  
              validation_set_size=0.2, batch_size=128, load_weights=False, load_path=None, plot_errors=False):
        """Trains the neural network.

        Keyword arguments:
        x_data -- input data
        y_data -- labels of the data samples in x_data
        num_epochs -- number of training epochs
        pricing_model -- the used electricity pricing model;
                         accepted values: 'constant', 'normally_distributed', 'deterministic_market', 'noisy_market'
        batch_size -- size of each mini-batch the network will be trained with
        save_weights_path -- file path, where the weights will be stored after training
        load_weights -- Boolean value that determines whether weights from another
                        version of the network (e.g. a trained one) will be loaded before training
        load_path -- file path to the weights that will be loaded if load_weights is set to True        
        plot_errors -- Boolean variable that determines whether the development of the validation and the training error
                       over the course of training will be plotted
        """
        with self.sess as sess:
            self._specified_batch_size = batch_size
            training_iterator, validation_iterator = self._get_training_and_validation_set_iterator(x_data, y_data, 
                                                                                                    validation_set_size)
            feedable_iterator, iterator_handle = self._get_feedable_iterator_with_handle(training_iterator)
            batch_features, batch_labels = feedable_iterator.get_next(name='get_next_batch')

            neg_elbo_loss = self._get_negative_elbo_loss(batch_features, batch_labels)
            mse = self._get_mse_tensor(batch_labels)
            r_squared = self._get_r_squared_tensor(batch_labels)
            overall_mse = tf.math.reduce_mean(tf.square(tf.math.subtract(batch_labels, self.logits)))
            tf.summary.scalar('ELBO_loss', tf.reshape(neg_elbo_loss, []))

            train_op = tf.train.AdamOptimizer().minimize(neg_elbo_loss, var_list=self.model.trainable_variables, name='STM_minimize_ELBO')

            merged_tensorboard_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./stored_data/Tensorboard/state_transition_model/')
            #writer.add_graph(sess.graph)

            self.train_neg_elbo_loss = np.zeros(num_epochs)
            self.train_mse = np.zeros((num_epochs, y_data.shape[1]))
            self.overall_train_mse = np.zeros(num_epochs)
            self.validation_mse = np.zeros((num_epochs, y_data.shape[1]))
            self.validation_r_squared = np.zeros((num_epochs, y_data.shape[1]))
            self.overall_validation_mse = np.zeros(num_epochs)

            self._initialize_local_and_global_variables()

            if load_weights:
                self.load_weights(load_path)

            training_set_handle = sess.run(training_iterator.string_handle())
            validation_set_handle = sess.run(validation_iterator.string_handle())

            for epoch in range(num_epochs):
                sess.run([training_iterator.initializer, validation_iterator.initializer])
                op_list = [train_op, neg_elbo_loss, overall_mse, mse, r_squared, merged_tensorboard_summary]
                (self.overall_validation_mse[epoch],
                 self.validation_mse[epoch, :],
                 self.validation_r_squared[epoch, :]) = self._compute_validation_error(op_list[2:],
                                                                                       iterator_handle,
                                                                                       validation_set_handle,
                                                                                       writer,
                                                                                       epoch)
                (self.train_neg_elbo_loss[epoch],
                 self.overall_train_mse[epoch],
                 self.train_mse[epoch, :]) = self._train_for_one_epoch(op_list[:-2],
                                                                       iterator_handle,
                                                                       training_set_handle)

                print("Finished " + str(epoch+1) + " epochs out of " + str(num_epochs))

            if num_epochs > 0:
                self.save_weights_and_monitoring_data(save_weights_path, pricing_model)

        if plot_errors == True:
            self._plot_errors()

    def _get_training_and_validation_set_iterator(self, x_data, y_data, validation_set_size):
        """Returns two iterators iterating through the training and validation set.

        Keyword arguments:
        x_data -- features of the whole dataset
        y_data -- labels of the whole dataset
        validation_set_size -- size of the dataset to be used for validation (ranging from 0 to 1)
        """
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_data.astype(np.float32),
                                                                                  y_data.astype(np.float32),
                                                                                  test_size=validation_set_size,
                                                                                  random_state=42)

        training_iterator = self._get_training_iterator()
        validation_iterator = self._get_validation_iterator()

        return training_iterator, validation_iterator

    def _get_training_iterator(self):
        """Returns an iterator over the training set."""
        with tf.name_scope('training_set_iterator'):
            training_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
            training_data = training_data.batch(self._specified_batch_size)
            training_iterator = training_data.make_initializable_iterator()

        return training_iterator

    def _get_validation_iterator(self):
        """Returns an iterator over the validation set."""
        with tf.name_scope('validation_set_iterator'):
            validation_data = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid))
            validation_data = validation_data.batch(self.x_valid.shape[0])
            validation_iterator = validation_data.make_initializable_iterator()

        return validation_iterator

    def _get_feedable_iterator_with_handle(self, training_iterator):
        """Returns a feedable iterator that can switch between iterating through the 
           training and validation set.

        Keyword arguments:
        training_iterator -- iterator for the training set
        """
        handle = tf.placeholder(tf.string, shape=[])
        feedable_iterator = tf.data.Iterator.from_string_handle(handle,
                                                                training_iterator.output_types,
                                                                training_iterator.output_shapes)

        return feedable_iterator, handle

    def _get_negative_elbo_loss(self, batch_features, batch_labels):
        """Returns the tensor for the negative evidence lower bound (ELBO), which serves as the loss.

        Keyword arguments:
        batch_features -- input features from a mini-batch of data
        batch_labels -- labels from a mini-batch of data
        """
        self.output_distribution = self._get_output_distribution(batch_features)
        neg_log_likelihood = -tf.reduce_mean(self.output_distribution.log_prob(batch_labels))
        kl_divergence = sum(self.model.losses) / self.x_train.shape[0] # has to be scaled when doing mini-batch training
        neg_elbo_loss = neg_log_likelihood + kl_divergence

        return neg_elbo_loss

    def _get_output_distribution(self, batch_features):
        """Returns a tensor representing distribution of the output of the model.
           The distribution is normal, where the means are the logits.

        Keyword arguments:
        batch_features -- input features from a mini-batch of data
        """
        self.logits = self.model(batch_features)
        output_distribution = tfd.Normal(loc=self.logits, scale=self.output_stds)

        return output_distribution

    def _get_mse_tensor(self, batch_labels):
        """Returns a Tensorflow tensor for the computation of the mean squared error (MSE)
           between the labels and the predictions with a given mini-batch of data.
           The returned tensor contains the MSE for each output element of the network 
           (i.e. it has as many elements as the network has output neurons).

        Keyword arguments:
        batch_labels -- labels from a mini-batch of data
        """
        squared_errors = tf.square(tf.math.subtract(batch_labels, self.logits))
        mses = tf.math.reduce_mean(squared_errors, axis=0)
        #tf.summary.scalar('average_MSE', tf.reduce_mean(mses))
        #tf.summary.histogram('average_MSE', mses)
        #tf.summary.tensor_summary('average_MSE', mses)

        return mses

    def _get_r_squared_tensor(self, batch_labels):
        """Returns a Tensorflow tensor for the computation of the R^2 value (or coefficient of determination)
           between the labels and the predictions with a given mini-batch of data. The returned tensor
           contains the R^2 value for each output element of the network (i.e. it has as many elements as 
           the network has output neurons).

        Keyword arguments:
        batch_labels -- labels from a mini-batch of data
        """
        total_error = tf.reduce_sum(tf.square(tf.math.subtract(batch_labels, tf.reduce_mean(batch_labels, axis=0))), axis=0)
        unexplained_error = tf.reduce_sum(tf.square(tf.math.subtract(batch_labels, self.logits)), axis=0)
        r_squared = tf.math.subtract(1.0, tf.math.divide(unexplained_error, total_error))
        #tf.summary.scalar('average_R_squared', tf.reduce_mean(r_squared))
        #tf.summary.histogram('average_R_squared', r_squared)
        #tf.summary.tensor_summary('average_R_squared', r_squared)

        return r_squared

    def _initialize_local_and_global_variables(self):
        """Runs the initializers for global and local variables in a Tensorflow session."""
        variables_initialization_op = tf.group(tf.global_variables_initializer(),
                                               tf.local_variables_initializer())
        self.sess.run(variables_initialization_op)

    def _train_for_one_epoch(self, op_list, iterator_handle, training_set_handle):
        """Trains the BNN on all the training data once (one epoch).

        Keyword arguments:
        op_list -- list of operations (adjusting the weights with gradients, error metrics) 
                   to be performed for each mini-batch
        iterator_handle -- placeholder for the handle of an iterator
        training_set_handle -- handle of the iterator that iterates through the training set
        """
        num_batches = int(np.ceil(self.y_train.shape[0] / self._specified_batch_size))
        train_elbo_loss = 0
        train_mse = np.zeros([1,self.y_train.shape[1]])
        overall_mse = 0

        for batch_idx in range(num_batches):
            feed_dict = {iterator_handle: training_set_handle}
            _, batch_train_loss, overall_mse_batch, batch_train_mse, logits = self.sess.run(op_list + [self.logits], feed_dict=feed_dict)

            train_elbo_loss += batch_train_loss
            size_of_current_batch = self._get_current_batch_size(batch_idx)
            train_mse += batch_train_mse * (size_of_current_batch / self.y_train.shape[0])
            overall_mse += overall_mse_batch * (size_of_current_batch / self.y_train.shape[0])

        return train_elbo_loss, overall_mse, train_mse

    def _get_current_batch_size(self, batch_idx):
        """Returns the size of the current mini-batch of training data.
           This function is needed because the last mini-batch of an epoch
           might have a different size than the other mini-batches.

        Keyword arguments:
        batch_idx -- index ranging from 0 to N-1, where N is the 
                     number of mini-batches used for training
        """
        num_batches = int(np.ceil(self.x_train.shape[0] / self._specified_batch_size))
        is_last_batch = (batch_idx == num_batches-1)
        num_samples_is_multiple_of_batch_size = (self.x_train.shape[0] % self._specified_batch_size == 0)

        if is_last_batch and not num_samples_is_multiple_of_batch_size:
            return self.x_train.shape[0] % self._specified_batch_size
        return self._specified_batch_size

    def _compute_validation_error(self, error_metrics_list, iterator_handle, validation_set_handle,
                                  tensorboard_writer, epoch):
        """Returns the validation error(s).

        Keyword arguments:
        error_metrics_list -- list of Tensorflow operations / Tensors that define error metrics
        iterator_handle -- placeholder for the handle of an iterator
        validation_set_handle -- handle of the iterator that iterates through the validation set
        tensorboard_writer -- Tensorboard file writer
        epoch -- number indicating after what training epoch the validation is done (starts from 0)
        """
        feed_dict = {iterator_handle: validation_set_handle}
        overall_mse, validation_mse, validation_r_squared, validation_summary, logits = self.sess.run(error_metrics_list + [self.logits], feed_dict=feed_dict)
        tensorboard_writer.add_summary(validation_summary, epoch)

        return overall_mse, validation_mse, validation_r_squared

    def load_weights(self, file_path):
        """Loads weights from a file specified by the file path.

        Keyword arguments:
        file_path -- file path to the file containing the weights
        """
        self.model.load_weights(file_path)
        print("\nrestored weights of the state transition model.\n")

    def save_weights_and_monitoring_data(self, file_path, pricing_model):
        """Loads weights from a file specified by the file path.

        Keyword arguments:
        file_path -- file path where the weights will be stored
        pricing_model -- the used electricity pricing model as a string
        """
        file_name_end = '_' + pricing_model + '.pkl'

        self.model.save_weights(file_path + '_' + pricing_model + '.h5')
        print("\nsaved weights of the state transition model to disk.\n")

        with open('./stored_data/monitoring/state_transition_model/training_negative_ELBO' + file_name_end, 'wb') as f:
            pickle.dump(self.train_neg_elbo_loss, f)
        with open('./stored_data/monitoring/state_transition_model/training_mses' + file_name_end, 'wb') as f:
            pickle.dump(self.train_mse, f)
        with open('./stored_data/monitoring/state_transition_model/validation_mses' + file_name_end, 'wb') as f:
            pickle.dump(self.validation_mse, f)
        print("saved STM monitoring data to disk.\n")

    def _plot_errors(self):
        """Plots the development of the training and validation errors over the course of training."""
        self.energy_system_features = ['generation', 'load', 'SoC']
        self.weather_features = ['global radiation', 'diffuse radiation', 'time']
        self.pricing_features = ['purchase price', 'selling price']
        _, axes = plt.subplots(2, 4)

        self._plot_elbo(axes[0, 0])
        self._plot_overall_mse(axes[1, 0])
        self._plot_MSEs_on_energy_system_features(axes[0, 1])
        self._plot_r_squared_on_energy_system_features(axes[1, 1])
        self._plot_MSEs_on_weather_features(axes[0, 2])
        self._plot_r_squared_on_weather_features(axes[1, 2])
        self._plot_MSEs_on_pricing_features(axes[0, 3])
        self._plot_r_squared_on_pricing_features(axes[1, 3])

        plt.show()

    def _plot_elbo(self, axes):
        """Plots the development of the ELBO over the course of training.

        Keyword arguments:
        axes -- the axes object of a subplot
        """
        axes.plot(self.train_neg_elbo_loss)
        axes.set_xlabel('Epoch')
        axes.set_ylabel('-ELBO')
        axes.set_title('Negative ELBO on training set')

    def _plot_overall_mse(self, axes):
        """Plots the development of the MSE of all features on the training and validation set.

        Keyword arguments:
        axes -- the axes object of a subplot
        """
        axes.plot(self.overall_train_mse, label='overall MSE on training set')
        axes.plot(self.overall_validation_mse, label='overall MSE on validation set')
        axes.legend()
        axes.set_xlabel('Epoch')
        axes.set_ylabel('MSE')
        axes.set_title('Overall MSE on all features')

    def _plot_MSEs_on_energy_system_features(self, axes):
        """Plots the development of the MSEs on features related to the energy 
           system on the training and validation set.

        Keyword arguments:
        axes -- the axes object of a subplot
        """
        for i, feature in enumerate(self.energy_system_features):
            axes.plot(self.train_mse[:,i], label=feature + ' MSE on training set')
            axes.plot(self.validation_mse[:,i], label=feature + ' MSE on validation set')
        axes.legend()
        axes.set_xlabel('Epoch')
        axes.set_ylabel('MSE')
        axes.set_title('MSE on energy system features (with the logits)')

    def _plot_MSEs_on_weather_features(self, axes):
        """Plots the development of the MSEs on features related to the weather
           on the training and validation set.

        Keyword arguments:
        axes -- the axes object of a subplot
        """
        for i, feature in enumerate(self.weather_features, start=len(self.energy_system_features)):
            axes.plot(self.train_mse[:,i], label=feature + ' MSE on training set')
            axes.plot(self.validation_mse[:,i], label=feature + ' MSE on validation set')
        axes.legend()
        axes.set_xlabel('Epoch')
        axes.set_ylabel('MSE')
        axes.set_title('MSE on weather features (with the logits)')
    
    def _plot_MSEs_on_pricing_features(self, axes):
        """Plots the development of the MSEs on features related to electricity pricing.

        Keyword arguments:
        axes -- the axes object of a subplot
        """
        num_non_pricing_features = len(self.energy_system_features) + len(self.weather_features) 
        for i, feature in enumerate(self.pricing_features, start=num_non_pricing_features):
            axes.plot(self.train_mse[:,i], label=feature + ' MSE on training set')
            axes.plot(self.validation_mse[:,i], label=feature + ' MSE on validation set')
        axes.legend()
        axes.set_xlabel('Epoch')
        axes.set_ylabel('MSE')
        axes.set_title('MSE on pricing features (with the logits)')

    def _plot_r_squared_on_energy_system_features(self, axes):
        """Plots the development of the R^2 values on features related to the energy
           system on the training and validation set.

        Keyword arguments:
        axes -- the axes object of a subplot
        """
        for i, feature in enumerate(self.energy_system_features):
            axes.plot(self.validation_r_squared[:,i], label=feature + ' $R^2$ on validation set')
        axes.legend()
        axes.set_xlabel('Epoch')
        axes.set_ylabel('$R^2$ value')
        axes.set_title('$R^2$ on energy system features (with the logits)')

    def _plot_r_squared_on_weather_features(self, axes):
        """Plots the development of the R^2 values on features related to the weather
           on the training and validation set.

        Keyword arguments:
        axes -- the axes object of a subplot
        """
        for i, feature in enumerate(self.weather_features, start=len(self.energy_system_features)):
            axes.plot(self.validation_r_squared[:,i], label=feature + ' $R^2$ on validation set')
        axes.legend()
        axes.set_xlabel('Epoch')
        axes.set_ylabel('$R^2$ value')
        axes.set_title('$R^2$ on weather features (with the logits)')

    def _plot_r_squared_on_pricing_features(self, axes):
        """Plots the development of the R^2 values on features related to the weather
           on the training and validation set.

        Keyword arguments:
        axes -- the axes object of a subplot
        """
        num_non_pricing_features = len(self.energy_system_features) + len(self.weather_features) 
        for i, feature in enumerate(self.pricing_features, start=num_non_pricing_features):
            axes.plot(self.validation_r_squared[:,i], label=feature + ' $R^2$ on validation set')
        axes.legend()
        axes.set_xlabel('Epoch')
        axes.set_ylabel('$R^2$ value')
        axes.set_title('$R^2$ on pricing features (with the logits)')
