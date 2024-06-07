import requests
import pandas as pd
import xmltodict, json

def createPositionURL(year,round):
    return "http://ergast.com/api/f1/" + str(year) + "/" + str(round) + "/results.json"

def createQuilURL(year,round):
    return 'http://ergast.com/api/f1/' + str(year) + '/' + str(round) + '/qualifying.json'

def getQuliData(number,yearsBack):
    rows = []
    for i in range(yearsBack):
        diff = yearsBack - i - 1
        year = 2024 - diff
        round = 1
        while(True):
            url = createPositionURL(year,round)
            response = requests.get(url)
            data = response.json()

            if (data["MRData"]["RaceTable"]["Races"] ==[]):
                print("Break")
                break
            race_info = data["MRData"]["RaceTable"]["Races"][0]
            raceResults = data["MRData"]["RaceTable"]["Races"][0]["Results"]
            driver_info = None
            for result in raceResults:
                try:
                    if result["Driver"]["permanentNumber"] == str(number):
                        driver_info = result
                        break
                except Exception as e:
                    pass

            if driver_info:
                single_row = {
                    "date": race_info["date"],
                    "grid": driver_info["grid"]
                }
                rows.append(single_row)
            round += 1
            print("round:" + str(round))
    df = pd.DataFrame(rows)
    df.to_pickle("Data/qual" + str(number)+".pkl")

##Effects: Creats a pandas Dataframe for the driver with the given permanet number, and writes that dataframe to number.pkl
def getDriverData(number,yearsBack):
    rows = []
    for i in range(yearsBack):
        diff = yearsBack - i - 1
        year = 2024 - diff
        print(year)
        round = 1
        while(True):
            url = createPositionURL(year,round)
            response = requests.get(url)
            data = response.json()
            if (data["MRData"]["RaceTable"]["Races"] ==[]):
                print("Break")
                break
            race_info = data["MRData"]["RaceTable"]["Races"][0]
            raceResults = data["MRData"]["RaceTable"]["Races"][0]["Results"]
            # Find the driver with number 33
            driver_33_info = None
            for result in raceResults:
                try:
                    if result["Driver"]["permanentNumber"] == str(number):
                        driver_33_info = result
                        break
                except Exception as e:
                    pass

            # Create the single-row table
            if driver_33_info:
                single_row = {
                    "raceName": race_info["raceName"],
                    "date": race_info["date"],
                    "circuitName": race_info["Circuit"]["circuitName"],
                    "driverNumber": driver_33_info["Driver"]["permanentNumber"],
                    "position": driver_33_info["position"]
                }
                rows.append(single_row)
            round += 1
            print("round:" + str(round))
    df = pd.DataFrame(rows)
    df.to_pickle("Data/" + str(number)+".pkl")



##Effects: gets the drivesrs last 5 races, and returns them as an array.
##         Number is the drivers permanent number
##         Year and Round and the year and round of the race to be predicted
def getPastPostion(number,year,round):
    results = []
    currRound = round - 1
    currYear = year
    completed = 0

    while (completed < 5):
        if (currRound <= 0):
            currRound = 30
            currYear -= 1

        url = createPositionURL(currYear,currRound)
        response = requests.get(url)
        data = response.json()
        if (data["MRData"]["RaceTable"]["Races"] !=[] ):
            race_info = data["MRData"]["RaceTable"]["Races"][0]
            raceResults = data["MRData"]["RaceTable"]["Races"][0]["Results"]
            driver_info = None

            for result in raceResults:
                try:
                    if result["Driver"]["permanentNumber"] == str(number):
                        driver_info = result
                        break
                except Exception as e:
                    pass

            if (driver_info):
                position = driver_info["position"]
                results.append(int(position))
                currRound -= 1
                completed += 1

    return results



def getPastQualifying(number,year,round):
    results = []
    currRound = round - 1
    currYear = year
    completed = 0

    while (completed < 5):
        if (currRound <= 0):
            currRound = 30
            currYear -= 1

        url = createPositionURL(currYear,currRound)
        response = requests.get(url)
        data = response.json()
        if (data["MRData"]["RaceTable"]["Races"] !=[] ):
            race_info = data["MRData"]["RaceTable"]["Races"][0]
            raceResults = data["MRData"]["RaceTable"]["Races"][0]["Results"]
            driver_info = None

            for result in raceResults:
                try:
                    if result["Driver"]["permanentNumber"] == str(number):
                        driver_info = result
                        break
                except Exception as e:
                    pass

            if (driver_info):
                position = driver_info["grid"]
                results.append(int(position))
                currRound -= 1
                completed += 1

    return results

