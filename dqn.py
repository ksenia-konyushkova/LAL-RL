# This code is the modified version of code from 
# ksenia-konyushkova/intelligent_annotation_dialogs/exp1_IAD_RL.ipynb

import tensorflow as tf
import numpy as np
import os
from estimator import Estimator

class DQN:
    """The DQN class that learns a RL policy.

    
    Attributes:
        session: A tensorflow session used for all computations.
        i_train: An integer counter of how many times function "train" was called.
        i_actions_taken: An integer counter of how many times function "get_action" was called.
        
        estimator: An object of class Estimator that is used for q-value prediction
        target_estimator: An object of class Estimator that is a lagging copy of estimator
        
        _initialized: A boolean variable indicating if the variables were initialized
        _reward_placeholder: A tf placeholder of floats of size batch_size for storing the rewards
        _terminal_placeholder: A tf placeholder of boolean of size batch_size for indicating if a transactional was terminal
        _next_best_prediction: A tf placeholder of floats of size batch_size 
            which contains the best prediction of the next step according to target_estimator
        _td_error: A tf vector containing tf errors on the batch
        _loss: A tf variable containing the loss on the batch
        _train_op: A tensorflow operation for training
        _copy_op: A tensorflow operation for copying (partially) variables of estimator into target_estimator
    """
    
    def __init__(self,
               experiment_dir,
               observation_length=6,
               learning_rate=1e-3,
               batch_size=32,
               target_copy_factor=0.001,
               bias_average=0,
               action_length=3,
              ):
        """Inits the DQN object.
        
        Args:
            experiment_dir: A string with parth to the folder where to save the agent and training data.
            observation_length: An integer indicating the number of features characterizing the classifier.
            learning_rate: A float with a learning rate for Adam optimiser.
            batch_size: An integer indicating the size of a batch to be sampled from replay buffer for estimator update.
            target_copy_factor: A float used for updates of target_estimator, 
                with a rule (1-target_copy_factor)*target_estimator weights
                + target_copy_factor*estimator
            bias_average: A float for the initial bias in the last layer of estimator. 
                Helps for optimisation because our samples do not have 0 mean at all.
                Usually initialise this bias to the mean episode duration of a few warm start episodes.
            action_length: An integer indicating the number of features characterizing the action (datapoint).
                Defined by the environment.
        """
        self.session = tf.Session()
        self.i_train = 0
        self.i_actions_taken = 0
        self._initialized = False
        
        # TARGET ESTIMATOR
        with tf.variable_scope("target_dqn"):
            self.target_estimator = Estimator(observation_length, 
                                              action_length, 
                                              is_target_dqn=False, 
                                              var_scope_name="target_dqn", 
                                              bias_average=bias_average)
        # ESTIMATOR
        with tf.variable_scope("dqn"):
            self.estimator = Estimator(observation_length, 
                                       action_length, 
                                       is_target_dqn=True, 
                                       var_scope_name="dqn", 
                                       bias_average=bias_average)    
            # placeholders for transactions from replay buffer
            self._reward_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size))
            self._terminal_placeholder = tf.placeholder(dtype=tf.bool, shape=(batch_size))
            # placeholder for the max of the next prediction by target estimator
            self._next_best_prediction = tf.placeholder(dtype=tf.float32, shape=(batch_size))
            ones = tf.ones(shape=(batch_size))
            zeros = tf.zeros(shape=(batch_size))
            # Contains 1 where not terminal, 0 where terminal.
            # dimensionality (batch_size x 1)
            terminal_mask = tf.where(self._terminal_placeholder, zeros, ones)
            # For samples that are not terminal, masked_target_predictions contains 
            # max next step action value predictions. 
            # dimensionality (batch_size x 1)
            masked_target_predictions = self._next_best_prediction * terminal_mask
            # Target values for actions taken (actions_taken_targets)
            #   = r + Q_target(s', a')  , for non-terminal transitions
            #   = r                     , for terminal transitions
            # dimensionality (batch_size x 1)
            actions_taken_targets = self._reward_placeholder + masked_target_predictions
            actions_taken_targets = tf.reshape(actions_taken_targets, (batch_size, 1))
            # Define temporal difference error 
            self._td_error = actions_taken_targets - self.estimator.predictions
            # Loss function
            self._loss = tf.reduce_sum(tf.square(self._td_error))
            # Training operation with Adam optimiser
            opt = tf.train.AdamOptimizer(learning_rate)
            self._train_op = opt.minimize(self._loss, var_list=tf.get_collection('dqn'))
            # Operation to copy parameter values (partially) to target estimator
            copy_factor_complement = 1 - target_copy_factor
            self._copy_op = [target_var.assign(target_copy_factor * my_var + copy_factor_complement * target_var)
                              for (my_var, target_var)
                              in zip(tf.get_collection('dqn'), tf.get_collection('target_dqn'))]
        # STATS FOR TENSORBOARD
        summaries_dir = os.path.join(experiment_dir, "summaries")
        summary_dir = os.path.join(summaries_dir, "summaries_{}".format("dqn"))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        self.summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())
        # SAVE VALUES OF VARIABLES
        self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver()
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.saver.restore(self.session, latest_checkpoint)
            print("Loading checkpoint {}...\n".format(latest_checkpoint))
            self._initialized = True
    
    def _check_initialized(self):
        """This functions checks is tf variables are initialised.
        
        If variables are initializes, it does nothing, 
        otherwise it initialises variables and 
        set the binary indiactor to true.
        """
        if not self._initialized:
            self.session.run(tf.global_variables_initializer())      
            self._initialized = True
    
    def get_action(self, classifier_state, action_state, n_tensorboard=1000000):
        """Get the best action in a state.
        
        This function returns the best action according to 
        Q-function estimator: the action with the highest 
        expected return in a given classification state 
        among all available action with given action states.
        
        Args:
            classifier_state: A numpy.ndarray characterizing the current classifier
                size number of datasamples in dataset.state_data.
            action_state: A numpy.ndarray where each column corresponds to the vector characterizing each possible action
                size #features characterising actions (check env, now 3) x #unlabelled datapoints. 
            n_tensorboard: An integer indicating how often stats should be written into tensorboard.
            
        Returns:
            max_action: An integer indicating the index of the best action 
                in list of actions in action_state.
        """
        # Counter of how many times this function was called
        self.i_actions_taken += 1
        self._check_initialized()
        # Repeat classification_state so that we have a copy of classification state for each possible action
        classifier_state = np.repeat([classifier_state], np.shape(action_state)[1], axis=0)        
        # Predict q-values with current estimator
        predictions = self.session.run(
            self.estimator.predictions,
            feed_dict = {self.estimator.classifier_placeholder: classifier_state, 
                         self.estimator.action_placeholder: action_state.T})
        # select one action with the highest prediction
        max_action = np.random.choice(np.where(predictions == predictions.max())[0])         
        # Every n_tensorboard iterations, add stats to tensorboard
        if self.i_actions_taken%n_tensorboard == 0:
            action_summary = tf.Summary()
            action_summary.value.add(simple_value=predictions.max(), tag="action_values/max_action_value")
            action_summary.value.add(simple_value=np.mean(predictions), tag="action_values/mean_action_value")
            action_summary.value.add(simple_value=np.min(predictions), tag="action_values/min_action_value")
            self.summary_writer.add_summary(action_summary, self.i_actions_taken)
            self.summary_writer.flush()
        return max_action
         
    def train(self, minibatch, n_tensorboard=10000):
        """Train a q-function estimator on a minibatch.
        
        Train estimator on minibatch, partially copy 
        optimised parameters to target_estimator. 
        We use double DQN that means that estimator is 
        used to select the best action, but target_estimator
        predicts the q-value.
        
        Args:
            minibatch: An object of class Minibatch containing transitions that will be used for training.
            n_tensorboard: An integer indicating how often stats should be written into tensorboard.
            
        Returns:
            _td_error: A tf vector containing tf errors on the batch.
                It is needed for priorotised replay. 
        """
        # NEXT BEST Q-VALUES
        # For bootstrapping, the target for update function depends on the q-function 
        # of the best next action. So, compute max_prediction_batch that represents 
        # Q_target_estimator(s', a_best_by estimator)
        self._check_initialized()
        max_prediction_batch = []
        i = 0
        # Counter of how many times this function was called
        self.i_train += 1
        # for every transaction in minibatch
        for next_classifier_state in minibatch.next_classifier_state:
            # Predict q-value function value for all available actions
            n_next_actions = np.shape(minibatch.next_action_state[i])[1]
            next_classifier_state = np.repeat([next_classifier_state], n_next_actions, axis=0)
            # Use target_estimator
            target_predictions = self.session.run(
                [self.target_estimator.predictions],
                feed_dict = {self.target_estimator.classifier_placeholder: next_classifier_state, 
                             self.target_estimator.action_placeholder: minibatch.next_action_state[i].T})
            # Use estimator
            predictions = self.session.run(
                [self.estimator.predictions],
                feed_dict = {self.estimator.classifier_placeholder: next_classifier_state, 
                             self.estimator.action_placeholder: minibatch.next_action_state[i].T})            
            target_predictions = np.ravel(target_predictions)
            predictions = np.ravel(predictions)
            # Follow Double Q-learning idea of van Hasselt, Guez, and Silver 2016
            # Select the best action according to predictions of estimator
            best_action_by_estimator = np.random.choice(np.where(predictions == np.amax(predictions))[0])
            # As the estimate of q-value of the best action, 
            # take the prediction of target estimator for the selecting action
            max_target_prediction_i = target_predictions[best_action_by_estimator]
            max_prediction_batch.append(max_target_prediction_i)
            i += 1
            
        # OPTIMIZE
        # Update Q-function value estimation
        _, loss, summ, _td_error = self.session.run(
            [self._train_op, self._loss, self.estimator.summaries, self._td_error],
            feed_dict = {self.estimator.classifier_placeholder: minibatch.classifier_state,
                self.estimator.action_placeholder: minibatch.action_state,
                self._next_best_prediction: max_prediction_batch,
                self._reward_placeholder: minibatch.reward,
                self._terminal_placeholder: minibatch.terminal,
                self.target_estimator.classifier_placeholder: minibatch.next_classifier_state,
                self.target_estimator.action_placeholder: minibatch.action_state})
        # Update target_estimator by partially copying the parameters of estimator 
        self.session.run(self._copy_op)
        
        # SAVE 
        # Save model and add summaries to tensorboard
        if self.i_train % n_tensorboard == 0:
            self.saver.save(self.session, os.path.join(self.checkpoint_dir, "model"))
            train_summary = tf.Summary()
            train_summary.value.add(simple_value=loss, tag="episode/loss")
            self.summary_writer.add_summary(train_summary, self.i_train)
            self.summary_writer.add_summary(summ, self.i_train)
            self.summary_writer.flush()
    
        return _td_error