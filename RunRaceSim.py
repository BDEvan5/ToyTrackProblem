import RaceSimulation as sim
import Models
import Interface

import Configurations as config

def run_sim():
    myTrack = Models.TrackData()
    config.straight_track(myTrack)
    # single_corner(myTrack)
    # simple_maze(myTrack)
    # diag_path(myTrack)

    myCar = Models.CarModel()
    config.standard_car(myCar)

    mySim = sim.RaceSimulation(myTrack, myCar)
    mySim.run_learning_sim(200)
    # mySim.run_learning_sim(1)


    
    myPlay = Interface.ReplayEp(myTrack)
    # myPlay.replay_best()
    myPlay.replay_last()

    mySim.plot_rewards()

    
if __name__ == "__main__":
    run_sim()