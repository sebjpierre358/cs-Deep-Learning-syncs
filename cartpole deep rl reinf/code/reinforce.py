import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        learn_schedule = tf.keras.optimizers.schedules.PolynomialDecay(.08,1000,.001,power=0.5)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learn_schedule) #used to be .002

        self.d1 = tf.keras.layers.Dense(self.num_actions, activation='softmax', bias_initializer='truncated_normal')

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        policy_distrib = self.d1(states)

        return policy_distrib

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        action_indices = [[x,actions[x]] for x in range(len(actions))]
        action_probs = tf.gather_nd(self.call(states), action_indices)

        loss = tf.math.multiply(tf.math.reduce_sum(tf.math.multiply(tf.math.log(action_probs), discounted_rewards)), tf.convert_to_tensor(-1.0, dtype=tf.float32))

        return loss
        
