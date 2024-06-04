import requests
import pandas as pd
import xmltodict, json

def createURL(year,round):
    return "http://ergast.com/api/f1/" + str(year) + "/" + str(round) + "/results.json"

##Effects: Creats a pandas Dataframe for the driver with the given permanet number, and writes that dataframe to number.pkl
def getDriverData(number,yearsBack):
    rows = []
    for i in range(yearsBack):
        diff = yearsBack - i - 1
        year = 2024 - diff
        print(year)
        round = 1
        while(True):
            url = createURL(year,round)
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



##Effects: gets the drivesrs last 5 races, writes them to pandas dataframe, and writest he dataframe to a PastNumber.pkl
def getPastPostion(number):
    print("stub")

