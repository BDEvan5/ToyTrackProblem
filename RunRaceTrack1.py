import TrackEnv1

if __name__ == "__main__":
    myTrack = TrackEnv1.RaceTrack()
    myCar = TrackEnv1.SimpleCarBase()



    myTrack.add_car(myCar)
    end_location = TrackEnv1.Location(20, 20)
    myTrack.add_end_point(end_location)


    myTrack.show_track()



    print("Finished")