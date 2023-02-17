# Notes on Stable Baselines 3 (SB3) and RLlib

In this document, I will note some of my experience while setting up and running algorithms from RLlib and SB3.

## Comparison overview

### First impressions and ease of use

#### SB3

I found Stable Baselines 3 extremely intuitive and easy to work with, whether it was setting up vectorized environments, using different algorithms and environments, tracking, logging, and even hyperparameter tuning (with RL Baselines 3 Zoo).

#### RLlib

Unfortunately, RLlib was a pain to work with. I think the biggest reason for this is that Ray, the parent project of RLlib, is a general-purpose machine-learning library. It comes with many specific libraries including Data, Train, Tune, Serve, and RLlib. I imagine that, as promised, it is extremely powerful for general-purpose use cases, i.e., when you have a specific algorithm that you want to implement which is already implemented in their code-base, and when you want fast and scalable training to be performed in a distributed manner on clusters.

However, all of this power comes with a ton of abstractions in their code which are not so easy to parse. For instance, most training goes through their Tune library, which automatically tunes hyperparameters. Disabling that setting and running algorithms without hyperparameter tuning while still using their tracking and logging functionality (with Tensorboard) was not straightforward for an RL algorithm, since their library is not just built for RL and not really meant to be used without hyperparameter tuning.

### Speed

With a non-vectorized environment (CartPole-v1) and the same parameters for RL algorithms (PPO) on both libraries, SB3 is faster by 25% to 36% for training on 100,000 episodes to 2000 episodes respectively. I suspect that is partly (if not entirely) because of the overheads for parallellization that of course don't show up in this test case. Even with vectorized environments on a single local machine with environments parallelized across multiple cores, I found SB3 to be much faster than RLlib. I never tested RLlib on a cluster with multiple nodes each with its own cores and/or multiple GPUs, but I imagine that would be much faster than SB3 since SB3 doesn't allow that functionality (as of writing this).

### Reviews

These reviews largely match my experience: [1](https://www.reddit.com/r/reinforcementlearning/comments/sj457y/stable_baselines_vs_rllib_vs_cleanrl/), [2](https://www.reddit.com/r/reinforcementlearning/comments/ulx7fu/whats_the_best_rl_lib/).
