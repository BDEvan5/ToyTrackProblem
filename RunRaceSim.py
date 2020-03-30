from RaceSimulation import RaceSimulation
from Interface import InterfaceManager

def run_sim():
    mySim = RaceSimulation()
    mySim.run_sim_course()
    # mySim.test_agent()

    myPlay = InterfaceManager(mySim.env.track, 100)
    # myPlay.replay_best()
    myPlay.replay_tests()



    
if __name__ == "__main__":
    run_sim()