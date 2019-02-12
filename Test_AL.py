import numpy as np
from scipy.special import entr

# This file contains a set of functions that are needed for testing RL agents and other strategies

def check_performance(all_scores):
    """This function computes the statistics on the duration of the episodes.
    
    Given quality scores for iterations of multiple episodes, 
    compute the average duration, standard deviation on it, 
    median duration and the maximum duration. The statistics
    are printed out.
    
    Args:
        all_scores: A list of lists of floats where 
            all_scores[0] contains accuracies of episode 0
            len(all_scores) is the number of episodes andz
            len(all_scores[0]) is a list of scores of the 0th episode.
    Returns:
        all_scores: The same as input.
        all_durations: A list of durations of all the episodes.
    """
    all_durations = []
    for score in all_scores:
        all_durations.append(len(score))
    all_durations = np.array(all_durations)
    print('mean +/- std is ', np.mean(all_durations), '+/-', np.std(all_durations))
    print('median is ', np.median(np.array(all_durations)))
    print('maximum is ', max(all_durations))
    return all_scores, all_durations

# Functions for various AL strategies
def policy_random(n_actions):
    """Random sampling selects a datapoint at random.
    
    Args:
        n_actions: Number of available for labelling datapoints.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint."""
    action = np.random.randint(0, n_actions)
    return action

def policy_uncertainty(next_action_prob):
    """Select an action according to uncertainty sampling strategy.
    
    Args:
        next_action_prob: A numpy.ndarray of size 1 x # datapoints
            available for labelling. Contains the probability to belong 
            to one of the classes for each of the available datapoints.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint.
    """
    # compute the distance to the boundary
    criterion = abs(next_action_prob-0.5)
    # select one of the datapoints that is the closest to the boundary
    max_action = np.random.choice(np.where(criterion == criterion.min())[0])
    action = max_action
    return action

def policy_rl(agent, state, next_action_state):
    """Select an action accourding to a RL agent.
    
    Args:
        agent: An object of DQN class.
        state: A numpy.ndarray characterizing the current classifier 
            The size is number of datasamples in dataset.state_data
        next_action_state: A numpy.ndarray 
            of size #features characterising actions (currently, 3) x #unlabelled datapoints 
            where each column corresponds to the vector characterizing each possible action.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint.
    """
    action = agent.get_action(state, next_action_state)
    return action

def policy_LAL(dataset, env, lal_model):
    """Selects an action according to LAL model containted in lal_model.
    
    Args:
        dataset: An object of class dataset.
        env: An object of class environment.
        lal_model: A regressor model. Should implement function predict.
            For example, can be RF regressor.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint.
    """
    unknown_data = dataset.train_data[env.indeces_unknown,:]
    known_labels = dataset.train_labels[env.indeces_known]
    n_lablled = np.size(env.indeces_known)
    n_dim = np.shape(dataset.train_data)[1]
    # FEATURES FOR LAL-INDEPENDENT or LAL-ITERATIVE
    # Get prediction of each tree
    temp = np.array([tree.predict_proba(unknown_data)[:,0] for tree in env.model_rf.estimators_])
    # Mean predictions           
    f_1 = np.mean(temp, axis=0)
    # Standard deviation of prediction
    f_2 = np.std(temp, axis=0)
    # Proportion of positive points
    f_3 = (sum(known_labels>0)/n_lablled)*np.ones_like(f_1)
    # Score estimated on out of bag estimate
    f_4 = env.model_rf.oob_score_*np.ones_like(f_1)
    # Coeficient of variance of feature importance
    f_5 = np.std(env.model_rf.feature_importances_/n_dim)*np.ones_like(f_1)
    # Variance of forest by avergae of variance of its predictions
    f_6 = np.mean(f_2, axis=0)*np.ones_like(f_1)
    # Average depth of the trees in the forest
    f_7 = np.mean(np.array([tree.tree_.max_depth for tree in env.model_rf.estimators_]))*np.ones_like(f_1)
    # Number of labelled datapoints
    f_8 = n_lablled*np.ones_like(f_1)
    # Concatenate all the features and get predictions for them
    LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
    LALfeatures = np.transpose(LALfeatures)
    LALprediction = lal_model.predict(LALfeatures)
    selectedIndex1toN = np.argmax(LALprediction)
    action = selectedIndex1toN
    return action

def policy_ALBE(qs, trn_ds, env, dataset):
    """Select an action to take according to ALBE strategy.
    
    Relies on the package provided by authors.
    
    Args:
        qs: An instance of ActiveLearningByLearning class from libact.
        trn_ds: An instance of Dataset class from libact.
        env: An instance of Environment class.
        dataset: An instance of Dataset class.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint.
    """
    selected_index = qs.make_query()
    selected_index1toN = np.where(env.indeces_unknown==selected_index)
    action = selected_index1toN[0][0]
    trn_ds.update(selected_index, dataset.train_labels[selected_index])
    return action

def policy_QUIRE(qs, trn_ds, env, dataset):
    """Select an action to take according to ALBE strategy.
    
    Relies on the package provided by authors of ALBE.
    
    Args:
        qs: An instance of QUIRE class from libact.
        trn_ds: An instance of Dataset class from libact.
        env: An instance of Environment class.
        dataset: An instance of Dataset class.
    Returns:
        action: An action to be taken in the environment: the index of 
            the selected datapoint.
    """
    selected_index = qs.make_query()
    selected_index1toN = np.where(env.indeces_unknown==selected_index)
    action = selected_index1toN[0][0]
    trn_ds.update(selected_index, dataset.train_labels[selected_index])
    return action

def check_performance_for_figure(all_scores, max_duration):
    """Compute the relative performance scores used for a figure.
    
    In taget quality environment, each episode lasts until a pre-defined 
    quality is reached. To be able to plot the performance for different 
    episodes together (where pre-defined quality and duration might differ) 
    we make pre-defined quality everywhere to be 1 and fill in missing 
    iterations at the end with 1s.
    
    Args:
        all_scores: A list of lists of floats where 
            all_scores[0] contains accuracies of episode 0
            len(all_scores) is the number of episodes andz
            len(all_scores[0]) is a list of scores of the 0th episode.
        max_duration: The duration of the longest episode.
    Returns:
        scores_relative_rand: A numpy ndarray of dimensionality 
            # episodes x duration of the longest episode.
    """
    scores_relative_rand = np.ones((len(all_scores), max_duration))
    j = 0
    for score in all_scores:
        i = 0
        for s in score:
            scores_relative_rand[j, i] = s/score[-1]
            i += 1
        j += 1
    return scores_relative_rand