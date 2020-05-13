from RaceSimulation import RaceSimulation
from Config import create_sim_config

def run_sim():
    config = create_sim_config()
    mySim = RaceSimulation(config)
    # mySim.plan_path()
    mySim.debug_agent_test()

    # mySim.test_agent()







    
if __name__ == "__main__":
    run_sim()
    