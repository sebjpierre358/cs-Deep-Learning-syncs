import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from operator import mul
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline


def visualize_data(total_rewards):
    """
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep.
    Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    #may be possible to do recursively
    T = len(rewards) #timesteps in episode
    disc_factor_terms = [discount_factor**x for x in range(T)]
    discounted_rewards = []

    for t in range(T):
        #mapping slices should be same length
        assert len(disc_factor_terms[:T-t]) == len(rewards[t:])
        discounted_rewards.append(sum(list(map(mul, disc_factor_terms[:T-t], rewards[t:]))))

    assert len(rewards) == len(discounted_rewards)

    return discounted_rewards

def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        action_distrib = model.call(tf.expand_dims(state, axis=0))
        action = np.random.choice([0,1], p=action_distrib.numpy()[0])
        states.append(state)
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """
    total_episode_rewards = []

    for x in range(650):
        states, actions, rewards = generate_trajectory(env, model)
        discounted_rewards = discount(rewards)
        episode_rewards = sum(rewards)

        with tf.GradientTape() as tape:

            loss = model.loss(tf.convert_to_tensor(states, tf.float32), actions, discounted_rewards)
            print("EPOCH {}/650 REWARD: {}".format(x+1, episode_rewards))

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_episode_rewards.append(episode_rewards)

        if x == 10 or x == 100 or x == 325:
            play(env, model)

    return total_episode_rewards

def play(env, model):
    state = env.reset()
    done = False

    while not done:
        env.render()
        action_distrib = model.call(tf.expand_dims(state, axis=0))
        action = np.random.choice([0,1], p=action_distrib.numpy()[0])
        state, rwd, done, _ = env.step(action)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions)
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)

    print("<-----------------------TRAINING>----------------------->\n")
    total_episode_rewards = train(env, model)

    avg_recent_reward = mean(total_episode_rewards[:-50])
    print(avg_recent_reward)
    visualize_data(total_episode_rewards)
    play(env, model)

if __name__ == '__main__':
    main()
