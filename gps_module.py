import serial


def read_gps_data():

    """ Description

    Function used to read the GPS Value from the connected device, GlobalSat BU-353S4 . If the model outputs gibberish values please check
    the mode set from SIRF TO NMEA.

    Attaching reference links to shift modes-
    https://www.linuxquestions.org/questions/linux-hardware-18/globalsat-bu-353-gps-device-need-some-setup-assistance-923351/
    http://www.usglobalsat.com/store/gpsfacts/bu353s4_gps_facts.html
    https://www.egr.msu.edu/classes/ece480/capstone/spring15/group14/uploads/4/2/0/3/42036453/wilsonappnote.pdf

    :return: Returns final coordinates
    """
    ser = serial.Serial('/dev/ttyUSB0', 4800, timeout=5)
    while True:
        line = ser.readline()
        splitline = line.split(',')

        if splitline[0] == '$GPGGA':
            latitude = splitline[2]
            latDirec = splitline[3]
            longitude = splitline[4]
            longDirec = splitline[5]
            #            print(line)

            if not latitude == ' ' and not longitude == ' ':
                print("location: ", to_degrees(latitude, longitude, latDirec, longDirec)[0], latDirec,
                to_degrees(latitude, longitude, latDirec, longDirec)[1], longDirec)
                final_coordinate = str(to_degrees(latitude, longitude, latDirec,longDirec)[0])+','+ str(to_degrees(latitude, longitude, latDirec,longDirec)[1])
            break
    return final_coordinate

def to_degrees(lats, longs, latDirec, longDirec):

    """ Description

    Use function to convert the GPS Coordinates to Degrees

    :return: Returns GPS Coordinates in Degrees along with the direction

    """
    lat_dd = lats[0:2]
    lat_ss = lats[2:]
    lat_str = float(lat_dd) + (float(lat_ss) / 60)

    if latDirec == 'S':
        lat_str = lat_str * (-1)

    lon_dd = longs[0:3]
    lon_ss = longs[3:]
    lon_str = float(lon_dd) + (float(lon_ss) / 60)
    
    if longDirec == 'W':
        lon_str = lon_str * (-1)

    return [lat_str, lon_str]






