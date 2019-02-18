from crontab import CronTab
import os
import sys
def run_service(username, time):
    my_cron = CronTab(user=username)
    job = my_cron.new("/home/suraj/Repositories/VirtualEnvironments/min_test/bin/python -u {0} {1} >> /home/suraj/loggss".format(os.path.realpath("smile_detection_demo.py"), os.path.realpath("configuration.json")))

    job.setall(time)
    my_cron.write()
    print job.is_valid()
    job.enable()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Enter the following to run the command \n Example: python nkdoo_service.py username '27 1 * * *' \n The '27 1 * * *' Dictates daily schedule to run it on")
    else:
        run_service(sys.argv[1], sys.argv[2])