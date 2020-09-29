import doudizhuc.scripts.agents as agents
import numpy as np

if __name__ == '__main__':
    agent = agents.make_agent('CDQN', '2')
    predictor = agent.predictor
    result = predictor.predict([3, 3, 3, 4, 4, 4, 5, 5, 6, 6], [[], []], np.ones(60))
    print("hello world")
