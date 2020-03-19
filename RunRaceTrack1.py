import ToyTrackSimulation as tts
import TrackEnv1


def straight_track(myTrack):
    start_location = [50.0, 95.0]
    end_location = [50.0, 15.0]
    o1 = (0, 0, 30, 100)
    o2 = (70, 0, 100, 100)
    o3 = (35, 80, 52, 85)
    b = (1, 1, 99, 99)

    myTrack.add_locations(start_location, end_location)
    myTrack.boundary = b
    myTrack.add_obstacle(o1)
    myTrack.add_obstacle(o2)
    myTrack.add_hidden_obstacle(o3)

def single_corner(myTrack):
    start_location = [80.0, 95.0]
    end_location = [5.0, 20.0]
    o1 = (0, 0, 100, 5)
    o2 = (0, 35, 65, 100)
    o3 = (95, 0, 100, 100)
    b = (1, 1, 99, 99)

    myTrack.add_locations(start_location, end_location)
    myTrack.boundary = b
    myTrack.add_obstacle(o1)
    myTrack.add_obstacle(o2)
    myTrack.add_obstacle(o3)

def simple_maze(myTrack):
    start_location = [95.0, 85.0]
    end_location = [10.0, 10.0]
    o1 = (20, 0, 40, 70)
    o2 = (60, 30, 80, 100)
    b = (1, 1, 99, 99)

    myTrack.add_locations(start_location, end_location)
    myTrack.boundary = b
    myTrack.add_obstacle(o1)
    myTrack.add_obstacle(o2)

def diag_path(myTrack):
    start_location = [95.0, 85.0]
    end_location = [10.0, 10.0]
    # o1 = (20, 0, 40, 70)
    # o2 = (60, 30, 80, 100)
    b = (1, 1, 99, 99)

    myTrack.add_locations(start_location, end_location)
    myTrack.boundary = b
    # myTrack.add_obstacle(o1)
    # myTrack.add_obstacle(o2)

def standard_car(myCar):
    max_v = 5

    myCar.set_up_car(max_v)


def run_sim():
    myTrack = TrackEnv1.TrackData()
    straight_track(myTrack)
    # single_corner(myTrack)
    # simple_maze(myTrack)
    # diag_path(myTrack)

    myCar = TrackEnv1.CarModel()
    standard_car(myCar)

    mySim = tts.RacingSimulation(myTrack, myCar)
    # mySim.run_standard_simulation()
    mySim.run_learning_sim(10)

    
if __name__ == "__main__":
    run_sim()






    print("Finished")