1. Python main.py to start training, and will save two agent pth model.
2. Python Interactive_mode.py is to play the game with control keys shown in terminal. This is debug the transition and reward model
3. Visualize.py is to load the agent let them self play. Press R to restart the simulation.
4. Training includes immediate reward episodes and delayed reward episodes. Simply speaking, immediate is assign at the moment you make action, like you get + 50 when you shoot and the ball belongs to you. Delayed reward is for the shoot action, after waiting 18 time frame it will formally happen, then we assign the reward according to the result (like + 100 if goal, -20 if out of field)

5. SHOOT has only angle parameter, and power is fixed 1.0. MOVE and INTERCEPT has 2d direction parameter.