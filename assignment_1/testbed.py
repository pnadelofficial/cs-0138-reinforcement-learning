import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import argparse

class TestBed:
    def __init__(self, k=10, runs=2000, increment_fnc=partial(np.random.normal, 0, 0.01), stationary=True):
        self.k = k
        self.runs = runs
        self.increment_fnc = increment_fnc
        self.stationary = stationary
        if self.stationary:
            self.learner = StationaryLearner
        else:
            self.learner = NonStationaryLearner

    def create_bandits(self,):
        q_star = np.full(self.k, 0.0)
        return q_star

    def get_reward(self, q_star, action):
        return np.random.normal(q_star[action], 1)

    def single_run(self, learning_algorithm, k=10, time_steps=1000): 
        q_star = self.create_bandits()
        optimal_action = np.argmax(q_star)

        rewards = []
        optimal_selections = []
        for i in range(time_steps):
            if i > 0:
                q_star += self.increment_fnc(k) # extra experiment
            
            action = learning_algorithm.select_action()
            reward = self.get_reward(q_star, action)
            rewards.append(reward)

            is_optimal = action == optimal_action
            optimal_selections.append(is_optimal)

            learning_algorithm.update(action, reward)
        return rewards, optimal_selections

    def evaluate_testbed(self):
        all_run_rewards = []
        all_run_optimal = []
        
        for run in range(self.runs):
            algorithm = self.learner()
            run_rewards, run_optimal = self.single_run(algorithm)
            all_run_rewards.append(run_rewards)
            all_run_optimal.append(run_optimal)
        
        self.average_rewards = np.mean(all_run_rewards, axis=0)
        label = "Stationary" if self.stationary else "Non-stationary"
        pd.DataFrame({"steps":range(len(self.average_rewards)), "rewards":self.average_rewards}).to_csv(f"{label}_{self.increment_fnc.func.__name__}_rewards.csv")
        print(f"Saved {label}_{self.increment_fnc.func.__name__}_rewards.csv")

        self.percent_optimal = np.mean(all_run_optimal, axis=0) * 100
        pd.DataFrame({"steps":range(len(self.percent_optimal)), "rewards":self.percent_optimal}).to_csv(f"{label}_{self.increment_fnc.func.__name__}_optimal.csv")
        print(f"Saved {label}_{self.increment_fnc.func.__name__}_optimal.csv")

    def plot_rewards(self):
        label = "Stationary" if self.stationary else "Non-stationary"
        plt.plot(range(len(self.average_rewards)), self.average_rewards, label=label)
        plt.title("Average rewards vs. Steps")
        plt.xlabel("Steps")
        plt.ylabel("Average rewards")
        plt.legend()

    def plot_optimal(self):
        label = "Stationary" if self.stationary else "Non-stationary"
        plt.plot(range(len(self.percent_optimal)), self.percent_optimal, label=label)
        plt.title("Percent optimal vs. Steps")
        plt.xlabel("Steps")
        plt.ylabel("Percent optimal")
        plt.legend()
        plt.show()

class StationaryLearner:
    def __init__(self, k=10, eps=0.1):
        self.k = k
        self.eps = eps
        self.q_array = np.zeros(self.k)
        self.arm_counts = {}
    
    def select_action(self):
        # non-greedy case
        if np.random.uniform(0,1) < self.eps:
            arm = np.random.randint(0, self.k)
        # greedy case
        else:
            arm = np.argmax(self.q_array)
        
        if arm in self.arm_counts:
            self.arm_counts[arm] += 1
        else:
            self.arm_counts[arm] = 1
        
        return arm

    def update(self, action, r):
        n = self.arm_counts[action]
        q = self.q_array[action]
        q = q + ((r-q)/n)
        self.q_array[action] = q
        return q
    
class NonStationaryLearner:
    def __init__(self, k=10, eps=0.1, alpha=0.1):
        self.k = k
        self.eps = eps
        self.q_array = np.zeros(self.k)
        self.alpha = alpha
    
    def select_action(self):
        # non-greedy case
        if np.random.uniform(0,1) < self.eps:
            arm = np.random.randint(0, self.k)
        # greedy case
        else:
            arm = np.argmax(self.q_array)
        return arm  
    
    def update(self, action, r):
        q = self.q_array[action]
        q = q + (self.alpha*(r - q))
        self.q_array[action] = q
        return q

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stationary", action="store_true")
    parser.add_argument("-t", "--nonstationary", action="store_true")
    parser.add_argument("-n", "--normal", action="store_true")
    parser.add_argument("-l", "--lognormal", action="store_true")
    parser.add_argument("-e", "--exponential", action="store_true")
    args = parser.parse_args()

    tbs = []
    if (args.stationary) and (args.normal):
        print("Running Stationary - Normal")
        stat_testbed_normal = TestBed(stationary=True)
        tbs.append(stat_testbed_normal)
    if (args.stationary) and (args.lognormal):
        print("Running Stationary - Lognormal")
        stat_testbed_lognormal = TestBed(increment_fnc=partial(np.random.lognormal, 0, 0.01), stationary=True)
        tbs.append(stat_testbed_lognormal)
    if (args.stationary) and (args.exponential):
        print("Running Stationary - Exponential")
        stat_testbed_exp = TestBed(increment_fnc=partial(np.random.exponential, 0.01), stationary=True)
        tbs.append(stat_testbed_exp)

    if (args.nonstationary) and (args.normal):
        print("Running Non-stationary - Normal")
        nonstat_testbed_normal = TestBed(stationary=False)
        tbs.append(nonstat_testbed_normal)
    if (args.stationary) and (args.lognormal):
        print("Running Non-stationary - Lognormal")
        nonstat_testbed_lognormal = TestBed(increment_fnc=partial(np.random.lognormal, 0, 0.01), stationary=False)
        tbs.append(nonstat_testbed_lognormal)
    if (args.stationary) and (args.exponential):
        print("Running Non-stationary - Exponential")
        nonstat_testbed_exp = TestBed(increment_fnc=partial(np.random.exponential, 0.01), stationary=False)
        tbs.append(nonstat_testbed_exp)

    for tb in tbs:
        tb.evaluate_testbed()

if __name__ == "__main__":
    main()