import serial


def read_gps_data():
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
                print "location: ", to_degrees(latitude, longitude, latDirec, longDirec)[0], latDirec,
                to_degrees(latitude, longitude, latDirec, longDirec)[1], longDirec
                final_coordinate = to_degrees(latitude, longitude, latDirec,longDirec)[0]+','+ to_degrees(latitude, longitude, latDirec,longDirec)[1]
            break
    return final_coordinate

def to_degrees(lats, longs, latDirec, longDirec):

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






