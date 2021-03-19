[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Using Deep Q-Learning for Navigation

## Problem to Solve.

For this project, an agent has to learn to navigate and collect bananas in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 

Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  The task is episodic and in order to solve the environment the agent must get an average score of at least +13 over 100 consecutive episodes.  

![Trained Agent][image1]

### Environment
The environment is built on Unity architecture.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

## Installation

### Python Dependencies

1. The software requires to install Python (3.6.1 or higher). We advocate to create a new environment with Python 3.6
     
     * **Linux or Mac**:
  
    ```sh
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    
    * **Windows**:
  
    ```sh
    conda create --name drlnd python=3.6 
    activate drlnd
    ```

2. Clone the repository, and navigate to the python/ folder. Then, install several dependencies.

    ```sh
        git clone https://github.com/udacity/deep-reinforcement-learning.git
        cd deep-reinforcement-learning/python
        pip install .
    ```
3. Create and activate IPython kernel for the drlnd environment.

    ```sh
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    
    In the jupyter notebook instance the kernel is activated from the dropdown menu _Kernel_



### Unity Packages and Environment
Besides the Python ML library `PyTorch` you will need to install the Unity Packages and Environments plus the relevant Python Packages following the instructions in [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

The ML-Agents Toolkit contains several components:

- Unity package `com.unity.ml-agents` contains the
  Unity C# SDK that will be integrated into your Unity project.  This package contains
  a sample to help you get started with ML-Agents.
- Unity package `com.unity.ml-agents.extensions` contains experimental C#/Unity components that are not yet ready to be part
  of the base `com.unity.ml-agents` package. `com.unity.ml-agents.extensions`
  has a direct dependency on `com.unity.ml-agents`.
- Three Python packages:
    - `mlagents` contains the machine learning algorithms that
      enables you to train behaviours in your Unity scene. Most users of ML-Agents
      will only need to directly install `mlagents`.
    - `mlagents_envs` contains a Python API to interact with
      a Unity scene. It is a foundational layer that facilitates data messaging
      between Unity scene and the Python machine learning algorithms.
      Consequently, `mlagents` depends on `mlagents_envs`.
    - `gym_unity` provides a Python-wrapper for your Unity scene
      that supports the OpenAI Gym interface.
- Download the Unity based environment `Banana` depending on the machine you are running it from:
    - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip "Mac OSX"): `"path/to/Banana.app"`
    - [Windows](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip "Windows"): `"path/to/Banana_Windows_x86_64/Banana.exe"`
    - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip "Linux"): `"path/to/Banana_Linux/Banana.x86_64"`
    - [Linux, headless](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip "Linux"): `"path/to/Banana_Linux_NoVis/Banana.x86_64"` 

For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:

## Instructions

You can either follow the steps in the python notebook `Navigation.ipynb` or run it locally from `Navigation.py`.

### Directory Structure

1. Main Python Notebook `Navigation.ipynb`
2. Main Python Code `Navigation.py`
3. Python module `dqn_agent.py` defines class Agent that learns by interacting with environment.
4. Python module `dqn_model.py` defines Q Function as a deep neural network. 
5. Python module `dqn_monitor.py` defines how the agent interacts with the environment either learning or following best policy. 
6. The directory `./Banana_Linux/Banana.x86_64` contains the Unity compiled program for the environment with visualization. The directory `./Banana_Linux_NoVis/Banana.x86_64` contains the same program but will not launch visualisation interface.
7. The PyTorch file `Trained_Agent.pth` is the trained model with weights of the Q Network.  



### GPU
If Cuda library available PyTorch will automatically run on GPU otherwise on cpu.

## License
MIT License

Copyright (c) [2021] [Polymathique]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
