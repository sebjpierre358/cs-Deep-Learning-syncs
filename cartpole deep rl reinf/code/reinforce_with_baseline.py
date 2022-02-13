import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        learn_schedule = tf.keras.optimizers.schedules.PolynomialDecay(.05,650,.001,power=0.5)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learn_schedule)

        self.actor_d1 = tf.keras.layers.Dense(self.num_actions, activation='softmax', bias_initializer='truncated_normal')

        self.critic_d1 = tf.keras.layers.Dense(25, activation='relu', bias_initializer='truncated_normal')
        self.critic_d2 = tf.keras.layers.Dense(1, bias_initializer='truncated_normal')

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
        policy_distrib = self.actor_d1(states)

        return policy_distrib

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        critic_1_out = self.critic_d1(states)
        value = self.critic_d2(critic_1_out)

        return value

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        action_indices = [[x,actions[x]] for x in range(len(actions))]
        action_probs = tf.gather_nd(self.call(states), action_indices)
        def advantage():
            return tf.math.subtract(discounted_rewards, tf.squeeze(self.value_function(states)))

        actor_loss = tf.math.multiply(tf.math.reduce_sum(tf.math.multiply(tf.math.log(action_probs),
                     tf.stop_gradient(advantage()))), tf.convert_to_tensor(-1.0, dtype=tf.float32))

        critic_loss = tf.math.reduce_sum(tf.math.pow(advantage(), 2))

        return actor_loss + critic_loss
