import  gym
import numpy as np
from plot_creator import PlotCreator, draw_3D_chart
import pandas as pd

def find_index(arr, n, K,l):
    start = 0
    end = n - 1
    while start <= end: 
        mid = (start + end) // 2 
        if arr[l][mid] == K:
            return mid 
        elif arr[l][mid] < K:
            start = mid + 1
        else:
            end = mid-1
    return end

class QLearningAgent:
    def __init__(self, discount, epsilon, learning_rate, num_discrete):
        self.discount = discount
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.state = None
        self.action = None
        #left-right-don't accelerate
        number_action = 3
        
        self.num_discrete = num_discrete  
        number_states = num_discrete ** 2 
        #create discretespacenum_discrete
        self.discrete_states = [np.linspace(-1.2, 0.6, num_discrete + 2)[1:-1], 
                                np.linspace(-0.07, 0.07, num_discrete + 2)[1:-1]]
        self.qtable = np.full((number_states, number_action), 0.0)

    def find_state_in_qtable(self, observation):
        indexx = find_index(self.discrete_states, 
                            self.num_discrete, 
                            observation[0], 0)
        indexv = find_index(self.discrete_states,
                            self.num_discrete,
                            observation[1], 1)                                      
        state = indexx + (self.num_discrete) * indexv                                                           
        return state

    def start_action(self, observation):
        self.epsilon = self.epsilon*(0.995)
        self.learning_rate = self.learning_rate * (0.99995)
        if self.learning_rate < pow(10, -5):
            self.learning_rate = pow(10, -5)
        self.state = self.find_state_in_qtable(observation)
        nextact = 0
        maxq = np.max(self.qtable[self.state, :])
        if maxq == self.qtable[self.state, 0]:
            nextact = 0
        elif maxq == self.qtable[self.state,1]:
            nextact = 1
        else:
            nextact = 2 
        return nextact
    
    def choose_action(self, observation, reward):
        nextstate = self.find_state_in_qtable(observation)
        nextact = 0
        if (1 - self.epsilon) <= np.random.uniform():
            nextact = np.random.randint(0, 3)            
        else:
            maxq = np.max(self.qtable[nextstate, :])
            if maxq == self.qtable[nextstate, 0]:
                nextact = 0
            elif maxq == self.qtable[nextstate, 1]:
                nextact = 1
            else:
                nextact = 2                                      
        sample = reward + self.discount * (np.max(self.qtable[nextstate, :]))             
        self.qtable[self.state, self.action] += self.learning_rate * (sample - self.qtable[self.state, self.action])
        self.state = nextstate
        self.action = nextact
        return nextact



def videos_to_record(episode_id):
    return episode_id in [100, 1000, 2000, 3000, 5500]

def is_dividable(episode_id):
    return any (episode_id % element == 0 for element in [100, 1000, 2000, 3000, 5500])

def main():
    render = False
    seed = 42
    working_dir = "videos"
    num_episodes = 10000

    env = gym.make("MountainCar-v0")

    env.seed(seed)
    np.random.seed(seed)

    env = gym.wrappers.Monitor(env, working_dir, force=True, resume=False,
                                video_callable=videos_to_record)
    
    agent = QLearningAgent(
        discount=0.98,
        epsilon=0.9,
        learning_rate=0.29,
        num_discrete=18
    )

    plot_redraw_frequency = 10
    my_plot = PlotCreator(num_episodes=num_episodes)
    my_plot.create_plot()

    rewards = []
    actions = []
    episodes = []

    for episode_index in range(num_episodes):
        if is_dividable(episode_index):
            render = True
        else:
            render = False

        observation = env.reset()
        action = agent.start_action(observation)
        total_reward = 0
        timestep = 0
        done = False

        while not done:
            # make an action and get the new observations
            observation, reward, done, info = env.step(action)
            total_reward += reward
            timestep += 1

            if render:
                env.render()
                print(f"The car is rendered with action: {action}, reward: {reward}, position: {observation[0]} and velocity: {observation[1]}")

            # compute the next action
            action = agent.choose_action(observation, reward)

        rewards.append(total_reward)
        episodes.append(episode_index)
        actions.append(agent.action)

        print(f"We reached the goal(out of the while loop) in {episode_index + 1} after {timestep} timesteps with {total_reward} reward.")

        my_plot[episode_index] = total_reward
        if render or episode_index % plot_redraw_frequency == 0:
            my_plot.update_plot(episode_index)
    DF = pd.DataFrame(agent.qtable)
    DF.to_csv("qtable1.csv")        
    draw_3D_chart(episodes, rewards, actions)
    env.close()


if __name__ == "__main__":
    main()