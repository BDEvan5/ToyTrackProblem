from RaceSimulation import RaceSimulation, AgentComparison
from Config import create_sim_config

def run_sim():
    config = create_sim_config()
    mySim = RaceSimulation(config)
    mySim.debug_agent_test()
    # mySim.run_classical_agent()
    # mySim.run_agent_training()
    # mySim.test_agent()

    # myComp = AgentComparison()
    # myComp.train_agents()
    # myComp.run_agent_comparison()  





    
if __name__ == "__main__":
    run_sim()
    