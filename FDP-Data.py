'''
CS3220 F1 Data Project

Hermogenes Parente hp298@cornell.edu

Data Sources:
Formula 1 Histroic Data: http://Ergast.com/mrd
Historic Weather Data: https://www.visualcrossing.com/ and https://rapidapi.com/organization/visual-crossing-corporation
'''
import time
from datetime import datetime
import requests
import json
import csv
import os
import numpy as np

# Sorry this is a mess

# Rapid API key for Visual Crossing Weather API
API_KEY= 'somethingsomethingsomething'

country_UTC_time= {
    '''
    The time-zone difference from UTC for a country (generally)
    '''
    'UTC'.lower()          : 0,
    'Australia'.lower()    : 11,
    'Malaysia'.lower()     : 8,
    'Bahrain'.lower()      : 3,
    'Spain'.lower()        : 1,
    'Turkey'.lower()       : 3,
    'Monaco'.lower()       : 1,
    'Canada'.lower()       :-5,
    'France'.lower()       : 1,
    'UK'.lower()           : 0,
    'Germany'.lower()      : 1,
    'Hungary'.lower()      : 1,
    'Belgium'.lower()      : 1,
    'Italy'.lower()        : 1,
    'Singapore'.lower()    : 8,
    'Japan'.lower()        : 9,
    'China'.lower()        : 8,
    'Brazil'.lower()       :-3,
    'USA'.lower()          :-6,
    'Austria'.lower()      : 1,
    'UAE'.lower()          : 4,
    'Argentina'.lower()    :-3,
    'Portugal'.lower()     : 0,
    'South Africa'.lower() : 2,
    'Mexico'.lower()       :-6,
    'Korea'.lower()        : 9,
    'Netherlands'.lower()  : 1,
    'Sweden'.lower()       : 1,
    'Morocco'.lower()      : 1,
    'Switzerland'.lower()  : 1,
    'India'.lower()        : 5,
    'Russia'.lower()       : 3,
    'Azerbaijan'.lower()   : 4,
    'Vietnam'.lower()      : 7}


# Main
def main():
    print("------------------Format Race Data------------------")
    print("Starting...\n")

    test()

    with open("TrainingData.txt", "w") as fp:
        json.dump(data().trainingData(), fp)

    print("\nDone")

def test():
    temp= data()
    temp.getWeatherValues('297')

class data:
    '''
    Saves all data and organizes into a more usable format
    '''

    def __init__(self):
        '''
        Saves all ergstat data into one object
        '''
        self.training_data_set= [] # For the training data

        files = [
            'circuits',
            'constructor_results',
            'constructor_standings',
            'constructors',
            'driver_standings',
            'drivers',
            'lap_times',
            'pit_stops',
            'qualifying',
            'races',
            'results',
            'seasons',
            'status']

        self.data_list= {}
        for i in range(len(files)):
            items= files[i]
            self.data_list[items]= raceData(items).getData()

    def getDriverRef(self, driverId):
        '''
        :type driverID: str
        :rtype : str
        '''
        for driver in self.data_list['drivers']:
            if driver[0] == driverId:
                return(driver[1])

    def getRaceRef(self, raceId):
        '''
        :type raceId: str
        :rtype : str,str,str,str (year,round,name,date,circuitId)
        '''
        for race in self.data_list['races']:
            if race[0]== raceId:
                return race[1],race[2],race[4],race[5],race[3]

    def getWeatherValues(self, raceId):
        '''
        :type raceId: str
        :rtype : str_list[time, temperature, humidity, precipitation]
        '''
        # Vars
        lat= ''
        lng= ''
        time= ''
        date= ''
        circuitId= ''
        temp= 0
        humidity= 0
        precip= 0

        # Find race times, date, and track
        for race in self.data_list['races']:
            if race[0] == raceId:
                date= race[5]
                time= race[6]
                circuitId= race[3]
        
        # Find track lat/lng
        for circuits in self.data_list['circuits']:
            if circuits[0] == circuitId:
                lat= circuits[5]
                lng= circuits[6]

        # Time if no time available
        time= '13:00:00' if time== '\\N' else time

        # Sets up to get the weather
        params= {
            "second" : time[6:],
            "minute" : time[3:5],
            "hour"   : time[:2],
            "day"    : date[8:],
            "month"  : date[5:7],
            "year"   : date[:4] }
        
        # Gets weather
        weather= getWeatherData(lat, lng, params)
        
        # If no weather data available for that year, find the next year that it is available for
        while weather[0]['datetime'] == None:
            params["year"]= int(params["year"]) + 1
            weather= getWeatherData(lat, lng, params)

        # Find weather values
        try:
            temp= 0.5 * ( float(weather[0]['temp']) + float(weather[1]['temp']) )
        except:
            try:
                temp= float(weather[0]['temp'])
            except: 
                temp= 20.0

        try:
            humidity= 0.5 * ( float(weather[0]['humidity']) + float(weather[1]['humidity']) )
        except:
            try:
                humidity= float(weather[0]['humidity'])
            except:
                humidity= 8.0
        
        try:
            precip= 0.5 * ( float(weather[0]['precip']) + float(weather[1]['precip']) )
        except:
            try:
                precip= float(weather[0]['precip'])
            except:
                precip= 0.0

        return [str(time), str(temp), str(humidity), str(precip)]

    def getDriverStandings(self, raceId):
        '''
        Driver Standings for the given season/year
        :type raceID: str
        :rtype: str_list_of_list[[driverId, points]].sorted()
        '''
        driver_standings_list= []

        # Find the race, if first race then make sure no one has points (duh)
        for race in self.data_list['races']:
            if race[0]==raceId:
                # If its the first race, multiply all the point values be 0 to get zeros
                if race[2]=='1':
                    offset= 0
                else: # Otherwise get the race before this standings results
                    offset= 1
                break
        
        # The points are from the standings of the previous race (ik its confusing, but that what i got)
        for row in self.data_list['driver_standings']:
                # race before/ 0's if first race
                if row[0] == str(int(raceId)-offset):
                    driver_standings_list.append([str(row[1]),str(float(row[2])*offset),int(row[4])*offset]) # if first race everyone is zeros


        driver_standings_list.sort(key= lambda x: -float(x[1]))# sort it

        # into dictionary
        driver_win_dict= {}
        for driver in driver_standings_list:
            driver_win_dict[driver[0]]= driver[2]
        
        return driver_standings_list,driver_win_dict

    def getConstructorStandings(self, raceId):
        '''
        Constructor Standings for the given season/year
        :type raceID: str
        :rtype: str_list_of_list[[driverId, points]].sorted()
        '''
        constructor_standings_list= []

        # check if first race
        for race in self.data_list['races']:
                if race[0]==raceId:
                    if race[2]=='1':
                        offset= 0
                    else:
                        offset= 1
                    break
        
        # adds constructors 
        for row in self.data_list['constructor_standings']:
                if row[0] == str(int(raceId)-offset):
                    constructor_standings_list.append([row[1],int(row[2])*offset])
        
        # if there is no contrusctor standings (b/c incomplete data), find the next years standings
        try:
            if constructor_standings_list== []:
                print("WARNING: Contructor Standings not available for round %i of the %i world championship!(raceId: %i)"%(int(race[2]),int(race[1]),int(race[0])))
                for nrace in self.data_list['races']:
                    if nrace[1]== str(int(race[1])+1) and nrace[2]== race[2]:
                        new_raceId= nrace[0]
                        print("Trying round %i of the %i world championship!(raceId: %i)"%(int(nrace[2]),int(nrace[1]),int(nrace[0])))
                        break
            
            # Try to find standings of the next year, same round #
                constructor_standings_list= self.getConstructorStandings(str(new_raceId))

        except:
            print("ERROR: Something went wrong trying to find Constructor Standings for round %i of the %i world championship!(raceId: %i)"%(int(race[2]),int(race[1]),int(race[0])))
        
        # Sort the list unless its the first race
        if not offset: return constructor_standings_list
        constructor_standings_list.sort(key= lambda x: -float(x[1]))
        return constructor_standings_list
    
    def getDriverStatsEtc(self, raceId):
        '''
        returns the nubmer of laps led by a driver in a career and season
        '''
        # Find the race round/year
        for race in self.data_list['races']:
            if race[0]==raceId:
                break
        race_round= int(race[2])
        race_year= int(race[1])

        # Go through every race, make a list of races before the current one
        total_race_list= []
        season_race_list= []
        for race in self.data_list['races']:
            # same or earlier racing year
            if int(race[1]) <= race_year:
                # same racing year, earlier round
                if int(race[1])== race_year:
                    if int(race[2]) < race_round:
                        season_race_list.append(race[0])
                        total_race_list.append(race[0])
                # ealier racing year
                else:
                    total_race_list.append(race[0])
        
        
        # Make list of drivers on the grid
        driver_list= [] # List of driers
        total_laps_driven={} # Experience of driver
        total_laps_led={} # Laps led in career
        season_laps_led={} # Laps led in season
        fastest_laps={} # Fastest laps
        total_starts={} # Races raced
        total_points={} # Points scored in a career
        total_wins={} # Total wins
        
        # finding each driver in the race
        for result in self.data_list['results']:
                # found quali for this race
                if result[0]==raceId:
                    # add the drivers to the dictionaries/list
                    driver_list.append(result[1])
                    total_laps_led[result[1]]=0
                    season_laps_led[result[1]]=0
                    total_laps_driven[result[1]]=0
                    fastest_laps[result[1]]=0
                    total_starts[result[1]]=0
                    total_points[result[1]]=0
                    total_wins[result[1]]=0
        
        # result stats
        for result in self.data_list['results']: 
            # races before current race      
            if result[0] in total_race_list and result[1] in driver_list:
                # fastest lap of the race
                if result[11]== '1':
                    fastest_laps[result[1]]+=1
                # wins
                if result[5]== '1':
                    total_wins[result[1]]+=1
                # add to races raced
                total_starts[result[1]]+=1
                total_laps_driven[result[1]]+=int(result[7])
                total_points[result[1]]+=float(result[6])
                
        # Go thorugh each lap of each race
        for laps in self.data_list['lap_times']:
            # Valid race and driver
            if laps[0] in total_race_list and laps[1] in driver_list and laps[3]== '1':
                total_laps_led[laps[1]]+=1 # Total laps led in career so far
                # same season
                if laps[0] in season_race_list:
                    season_laps_led[laps[1]]+=1 # Total laps led in current season
        
        return driver_list, total_laps_driven, total_laps_led, season_laps_led, fastest_laps, total_starts, total_points, total_wins

    def qualiOrder(self, raceId):
        '''
        : type raceId: str
        : rtype : list_of_lists
        '''
        # Find the race round/year
        for race in self.data_list['races']:
            if race[0]==raceId:
                break
        race_year= int(race[1])
        race_round= int(race[2])

        # A quli-classification exists
        qualifying=[]
        if race_year >= 1995:
            for quali in self.data_list['qualifying']:
                if quali[0]==raceId:
                    qualifying.append([quali[1],quali[4]])

        # quali classification doesnt exits
        else:
            drivers= []
            # get list of drivers in the race
            for row in self.data_list['results']:
                if row[0]== raceId:
                    drivers.append(row[1])

            if race_round!= 1:
                # use the results from last race
                prev= []
                d_list= []
                for row in self.data_list['results']:
                    if row[0]== str(int(raceId)-1):
                        prev.append([row[1],row[5]])
                        d_list.append(row[1])

                prev.sort(key= lambda x: int(x[1]))
                pos= len(prev)+1
                # if not all the drivers are in the qualifying order
                if len(drivers) > len(qualifying):
                    for driver in drivers:
                        if driver not in d_list:
                            prev.append([driver,pos]) # add to qalifying list
                            pos+=1 # increment position
                qualifying= prev
            
            else:
                for i in range(len(drivers)):
                    qualifying.append([drivers[i],str(i+1)])

        # turn list into a dict    
        quali_dict= {}
        qualifying.sort(key= lambda x: int(x[1]))
        for driver in qualifying:
            quali_dict[str(driver[0])]= str(driver[1])

        return(quali_dict)

    def qualiTest(self, raceId):
        standings, wins= self.getDriverStandings(raceId)

        q= self.qualiOrder(raceId)
        for order in q:
            print(q[order]+ ". " +self.getDriverRef(order).capitalize()+ "(driverId: " +order+ ")")
            try:
                print("_____Points: "+standings[order])
            except:
                print("Not in standings - 0 points")

    def testDriverStats(self, raceId):
        '''
        Prints out driver stats and standings at the current race
        '''
        race_year,race_round,race_name,race_date= self.getRaceRef(raceId)
        driver_list, driver_tot_laps, driver_led_laps, driver_szn_led, driver_fst_laps, driver_tot_strt, driver_tot_pnts, driver_tot_wins \
            = self.getDriverStatsEtc(raceId)
        driver_szn_pnts, driver_szn_wins= self.getDriverStandings(raceId)

        print("For round "+ race_round +" is the "+ race_year +" "+ race_name +" on "+  race_date)
        print("_________________________________________________________")
        for driver in driver_list:
            d= self.getDriverRef(driver)
            print("                           "+d)

            print("Points:")
            try:
                print("Points this season: "+driver_szn_pnts[driver])
            except:
                print("!!!!!!!!!!!!!!!!!!! :Something went wrong with getDriverStandings: !!!!!!!!!!!!!!!!!!!!!")
                print("Points this season: 0")
            print("Career Points: "+str(driver_tot_pnts[driver])) # float

            print("Wins:")
            try:
                print("Wins this season: "+driver_szn_wins[driver])
            except:
                print("!!!!!!!!!!!!!!!!!!! :Something went wrong with getDriverStandings: !!!!!!!!!!!!!!!!!!!!!")
                print("Wins this season: 0")
            print("Career Wins: "+str(driver_tot_wins[driver])) # int

            print("Laps:")
            print("Laps led this season: "+str(driver_szn_led[driver])) # int
            print("Career laps led: "+str(driver_led_laps[driver])) # int
            print("Total laps driven: "+str(driver_tot_laps[driver])) # int
            
            print("Stats:")
            print("Total fastest laps: "+str(driver_fst_laps[driver])) # int
            print("Total race starts: "+str(driver_tot_strt[driver])) # int
            
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        print("_________________________________________________________")

    def raceVectorSet(self, raceId):
        '''
        Makes the vector for a single race incuding all the necessary features
        Returns the vector
        '''
        print("Making Vector (raceId: %i)"% int(raceId))
        data_vector= []

        # date/round
        try:
            year, round_num, race_name, date, circuitId= self.getRaceRef(str(raceId))
        except:
            print("Error: getRaceRef(raceId: %i)"% int(raceId))
        month= date[5:7]
        day= date[8:]

        # lat/lng/alt
        try:
            for circuits in self.data_list['circuits']:
                if circuits[0]== circuitId:
                    lat= circuits[5]
                    lng= circuits[6]
                    alt= circuits[7] if circuits[7] != '\\N' else 0
        except:
            print("Error: circuits data (raceId: %i, circuitId: %i)"% (int(raceId),int(circuitId)))
            
        # time/weather
        try:
            weather_values_data= self.getWeatherValues(str(raceId))
        except:
            print("Error: getWeatherValues(raceId: %i)"% int(raceId))
        
        time= weather_values_data[0]
        temp= weather_values_data[1]
        humi= weather_values_data[2]
        precip= weather_values_data[3]
        hour= time[:2]
        minute= time[3:5]

        # List of drivers and constructors
        drivers= []
        constructors= []
        teams= {}
        race_results= {}
        try:
            for result in self.data_list['results']:
                if result[0]== str(raceId):
                    drivers.append(result[1])
                    race_results[result[1]]= int(result[5])
                    if result[2] not in constructors:
                        constructors.append(result[2])
                        teams[result[2]]= [result[1]]
                    else:
                        teams[result[2]].append(result[1])
        except:
            print("Error: drivers & constructors (raceId: %i)"% int(raceId))
        
        # Drivers and Constructor standings
        constructor_standings= {}
        driver_standings= {}
        try:
            # Makes necessary dictionaries
            for team in constructors:
                constructor_standings[team]= 0
            for driver in drivers:
                driver_standings[driver]= 0
            
            # Call standings functions
            temp_d_stand, driver_wins= self.getDriverStandings(str(raceId)) # list, dict
            temp_c_stand= self.getConstructorStandings(str(raceId)) # list
        except:
            print("Error: Standings Setup (raceId: %i)"% int(raceId))

        try:
            # Update driver standings
            pos= 1
            for driver in temp_d_stand:
                if driver[0] in drivers:
                    driver_standings[driver[0]]= pos
                    pos+=1
            for driver in drivers:
                if driver not in driver_standings:
                    driver_standings[driver]= pos
                    pos+= 1
        except:
            print("Error: Driver Standings (raceId: %i)"% int(raceId))

        try:
            # Update constructor standings
            pos= 1
            for constr in temp_c_stand:
                if constr[0] in constructors:
                    constructor_standings[constr[0]]= pos
                    pos+= 1
            for constr in constructors:
                if constr not in constructor_standings:
                    constructor_standings[constr]= pos
                    pos+= 1
        except:
            print("Error: Constructor Standings (raceId: %i)"% int(raceId))

        # race qualifying
        qualifying_order= {}
        try:
            quali= self.qualiOrder(str(raceId))
            pos= 1
            not_in_quali= []
            for driver in drivers:
                try:
                    qualifying_order[driver]= int(quali[driver])
                    pos= int(quali[driver]) if pos < int(quali[driver]) else pos
                except:
                    not_in_quali.append(driver)
            for driver in not_in_quali:
                pos+= 1
                qualifying_order[driver]= pos
        except:
            print("Error: Qualifying Order (raceId: %i)"% int(raceId))

        # driver wins in a season
        season_wins= {}
        try:
            for driver in drivers:
                try:
                    season_wins[driver]= int(driver_wins[driver])
                except:
                    print("Driver not in wins dictonary. Adding them now!")
                    season_wins[driver]= 0
        except:
            print("Error: Driver Wins (raceId: %i)"% int(raceId))

        # stats etc
        # driver_list, total_laps_driven, total_laps_led, season_laps_led, fastest_laps, total_starts, total_points, total_wins
        try:
            stats_driver_list, tot_laps, tot_led, szn_led, tot_fast, tot_starts, tot_points, tot_wins= self.getDriverStatsEtc(str(raceId))
            
            #season_wins= driver_wins (caluclated above)
            season_laps_led= szn_led

            career_laps_driven= tot_laps
            career_laps_led= tot_led
            career_races_started= tot_starts
            career_fastest_laps= tot_fast
            career_points= tot_points
            career_wins= tot_wins
            
        except:
            print("Error: Driver Stats (raceId: %i)"% int(raceId))
        

        #-> Add raceId
        data_vector.append(int(raceId))                      # Race ID
        #-> Add circuitId
        data_vector.append(int(circuitId))                   # Circuit ID
        #-> Add round number
        data_vector.append(int(round_num))                   # Round Number

        #-> Add race year
        data_vector.append(int(year))                        # Year    \ 
        #-> Add race month                                              \
        data_vector.append(int(month))                       # Month     > Date of Race
        #-> Add race day                                                /
        data_vector.append(int(day))                         # Day     /
        #-> Add race hour 
        data_vector.append(int(hour))                        # Hour    \
        #-> Add race minute                                             > Time of Race
        data_vector.append(int(minute))                      # Minute  /

        #-> Add race latitude
        data_vector.append(float(lat))                       # Circuit Latitude
        #-> Add race longitude
        data_vector.append(float(lng))                       # Circuit Longitude
        #-> Add race altitude
        data_vector.append(float(alt))                       # Circuit Altitude 

        #-> Add race temperature
        data_vector.append(float(temp))                      # Temperature    \
        #-> Add race humidity                                                  \
        data_vector.append(float(humi))                      # Humidity         > Weather of Race
        #-> Add race precipitation                                             / 
        data_vector.append(float(precip))                    # Precipitation  /

        #-> Add constructors in the race
        data_vector.append(constructors) #list               # Constructors in the Race
        #-> Add the constructor positions before the race
        data_vector.append(constructor_standings) #dict      # Constructure Championship Standings Order
        #-> Add the teams: constructor -> drivers
        data_vector.append(teams) #dict                      # Constructors and thier Current Drivers
        
        #-> Add driver in the race
        data_vector.append(drivers) #list                    # Drivers in the Race
        #-> Add the driver positions before the race
        data_vector.append(driver_standings) #dict           # Driver's Championship Standings Order
        
        #-> Add the qualifying order of the race
        data_vector.append(qualifying_order) #dict           # Qualifying/Starting Order of the Race
        
        #-> Add the number of wins each driver had this season so far
        data_vector.append(season_wins) #dict                # Wins in the Season for Each Driver
        #-> Add the number of laps each driver had this season so far
        data_vector.append(season_laps_led) #dict            # Laps Led in the Season for Each Driver

       #-> Add the number of races each driver had started in their career
        data_vector.append(career_races_started) #dict       # Races Raced in a Career by Each Driver
        #-> Add the number of laps each driver had driven in their career
        data_vector.append(career_laps_driven) #dict         # Laps Driven in a Career by Each Driver
        #-> Add the number of laps each driver had led in their career
        data_vector.append(career_laps_led) #dict            # Laps Led in a Career by Each Driver
        #-> Add the number of laps each driver had the fastests in their career
        data_vector.append(career_fastest_laps) #dict        # Fastest Laps in a Career by Each Driver
        #-> Add the number of points each driver had in their career
        data_vector.append(career_points) #dict              # Points Won in a Career by Each Driver
        #-> Add the number of wins each driver had in their career
        data_vector.append(career_wins) #dict                # Races Won in a Career by Each Driver

        #-> Add the fininshing order of the race
        data_vector.append(race_results) #dict               # Finishing Order of the Race

        # Returns the vector
        return data_vector

    def driverVectorSet(self, race_vec):
        '''
        Given a set of data for a particular race
        Return an array: each vector in the array is a data set for a driver in the race
        '''
        driver_array= []
        drivers= race_vec[17]

        # Copy common data points
        copy_vec= []
        for i in range(14): # first 14 (i.e. 0-13) features same for all drivers of same race
            copy_vec.append(race_vec[i])

        # set team/driver pairs
        driver_team= {}
        for team in race_vec[16]: #team/driver dictionary from race_vec
            for driver in race_vec[16][team]: # each driver in that team
                driver_team[driver]= team

        # Go through each driver in the race, and make a vector for them
        for driver in drivers:
            driver_vec= copy_vec.copy()

            # driver's team current championship standing
            team= driver_team[driver]
            team_standings= race_vec[15][team]
            driver_vec.append(team_standings) # driver_vec[14]

            # driver's current championship standing
            driver_standings= race_vec[18][driver]
            driver_vec.append(driver_standings) # driver_vec[15]

            # driver's qualifying/starting position
            quali= race_vec[19][driver]
            driver_vec.append(quali) # driver_vec[16]

            # driver stats
            driver_vec.append(race_vec[20][driver]) # driver_vec[17]
            driver_vec.append(race_vec[21][driver]) # driver_vec[18]
            driver_vec.append(race_vec[22][driver]) # driver_vec[19]
            driver_vec.append(race_vec[23][driver]) # driver_vec[20]
            driver_vec.append(race_vec[24][driver]) # driver_vec[21]
            driver_vec.append(race_vec[25][driver]) # driver_vec[22]
            driver_vec.append(race_vec[26][driver]) # driver_vec[23]
            driver_vec.append(race_vec[27][driver]) # driver_vec[24]
            
            # finishing position
            driver_vec.append(race_vec[28][driver]) # driver_vec[25]

            # add with other driver vectors
            driver_array.append(driver_vec)
        
        return driver_array

    def trainingData(self):
        '''
        Returns a list of data used for training a model
        and an associated list of the varaibles not needed for training, but useful

        Features:
            RaceId       : int
            CircuitId    : int

            Round Num    : int
            
            Year         : int See Notes.1.
            Month        : int See Notes.1.
            Day          : int See Notes.1.
            
            Hour         : int; Default: 13 \ local time
            Min          : int; Default: 0  /

            Latitude     : float
            Longitude    : float
            Altitude     : float

            Tempature    : float ( celsius ) \ 
            Humidity     : float ( g/kg )     > Weather Data Avg Race (2 Hours)
            Precip       : float ( mm )      /
            
            Constr List  : list
            Constr Stand : dict

            Driver List  : list
            Driver Stand : dict

            Qualifying   : dict Default: See Notes.2.
            Laps Led Szn : dict
            Laps Led Tot : dict
            Laps Driven  : dict
            Tot Fastest  : dict
            Tot Race Srt : dict
            Tot Points   : dict
            Tot Wins     : dict
            Szn Wins     : dict


        Notes:
        1.  Day and Month decided by using round number divided by 22 (the usual # of rounds in a season)
            The season is from March to November (i.e. 3 <= month <= 11)
            Then take fraction from round#/22 and multiple by season length (8 months * 30.5 days per month)
            Then divide total days (rounded to int) by 30.5 for months and remainder is days (rounded to int)
            Quick check to make sure that correct number of day for that month (30 days in month 4,6,9,11)
            Every race has an associated year
        2.  Pre 1995 no qualifying data, using finishing order of last race.
            For first race of season, go in oder of last season standings, then driverId number
            First race in data, 1950, in driverId order
        '''
        print("Making Training Set...")

        for race in self.data_list['races']:
            try:
                race_vec= self.raceVectorSet(race[0])
                driver_arr= self.driverVectorSet(race_vec)
                for vec in  driver_arr:
                    self.training_data_set.append(vec)
            except:
                print("Something went wrong with raceId: " + race[0])
        
        return self.training_data_set

class raceData:
    '''
    :type fileName: str

    Takes in CSV file name and returns array with the data from the file
    CSV data files from Ergast.com/mrd/db
    '''


    def __init__(self, fileName):
        '''
        :type fileName: str
        
        Possible File:
        '''
        self.__files = [
        'circuits',
        'constructor_results',
        'constructor_standings',
        'constructors',
        'driver_standings',
        'drivers',
        'lap_times',
        'pit_stops',
        'qualifying',
        'races',
        'results',
        'seasons',
        'status']

        self.__csvData= []

        # Check type
        assert isinstance(fileName,str), "Argument \'fileName\' Must Be Type Str..."
        fileName = fileName.lower()
        self.__fileName = fileName
        
        # Check Valid Input
        assert (self.__files.count(fileName) > 0), "Arguemnt \'fileName\' Must Be A Valid File Name..."

        print('Reading \'%s.csv\'... ' % self.__fileName)
        cwd = os.path.dirname(__file__)
        with open(os.path.join(cwd,'historicRaceData',self.__fileName +'.csv'), newline='') as csvFile:
            data = csv.DictReader(csvFile)
            for row in data:
                self.__csvData.append(dict(row))
        print('Finished')
        

    def getName(self):
        '''
        :rtype str:

        Returns the file name of the data
        '''
        return self.__fileName

    def getRawData(self):
        '''
        :rtype dict:

        Returns a dictionary of the CSV file
        '''
        return self.__csvData

    def getData(self):
        '''
        :rtype list[list[]]:

        Returns organized list of the data based on the file
        Check schemas.txt for more details on types
        '''
        switcher = {
            1:  self.__circuits,
            2:  self.__constructor_results,
            3:  self.__constructor_standings,
            4:  self.__constructors,
            5:  self.__driver_standings,
            6:  self.__drivers,
            7:  self.__lap_times,
            8:  self.__pit_stops,
            9:  self.__qualifying,
            10: self.__races,
            11: self.__results,
            12: self.__seasons,
            13: self.__status
        }
        index = self.__files.index(self.__fileName)
        orgData = switcher.get(index+1, lambda: "Invalid File")
        return orgData()

    def __circuits(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [circuitId1,circuitRef1,name1,location1,country1,lat1,lng1],
            [circuitId2,circuitRef2,name2,location2,country2,lat2,lng2],
            . . . ]
        '''
        circuitsList= []
        for track in self.__csvData:
            circuitsList.append([track['circuitId'],track['circuitRef'],track['name'],
                                track['location'],track['country'],track['lat'],track['lng'],track['alt']])
        return circuitsList

    def __constructor_results(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [raceId1,constructorId1,points1,status1],
            . . . ]
        '''
        resultsList= []
        for row in self.__csvData:
            resultsList.append([row['raceId'],row['constructorId'],row['points'],row['status']])
        return resultsList

    def __constructor_standings(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [raceId1,constructorId1,points1,position1,wins1],
            . . . ]
        '''
        standingsList= []
        for row in self.__csvData:
            standingsList.append([row['raceId'],row['constructorId'],row['points'],
                                  row['position'], row['wins']])
        return standingsList

    def __constructors(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [constructorId1,constructorRef1,name1,nationality1],
            . . . ]
        '''
        constructorsList= []
        for row in self.__csvData:
            constructorsList.append([row['constructorId'],row['constructorRef'],row['name'],
                                  row['nationality']])
        return constructorsList

    def __driver_standings(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [raceId1,driverId1,points1,position1,win1],
            . . . ]
        '''
        standingsList= []
        for row in self.__csvData:
            standingsList.append([row['raceId'],row['driverId'],row['points'],
                                  row['position'], row['wins']])
        return standingsList

    def __drivers(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [driverId1,driverRef1,number1,code1,forename1,surname1,dob1,nationality1],
            . . . ]
        '''
        driversList= []
        for row in self.__csvData:
            driversList.append([row['driverId'],row['driverRef'],row['number'],row['code'],
                                row['dob'],row['nationality']])
        return driversList

    def __lap_times(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [raceId1,driverId1,lap1,position1,time1,milliseconds1],
            . . . ]
        '''
        timesList= []
        for lap in self.__csvData:
            timesList.append([lap['raceId'],lap['driverId'],lap['lap'],lap['position'],
                              lap['time'],lap['milliseconds']])
        return timesList

    def __pit_stops(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [raceId1,driverId1,stop1,lap1,time1,duration1,milliseconds1],
            . . . ]
        '''
        timesList= []
        for pit in self.__csvData:
            timesList.append([pit['raceId'],pit['driverId'],pit['stop'],pit['lap'],
                              pit['time'],pit['duration'],pit['milliseconds']])
        return timesList

    def __qualifying(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [raceId1,driverId1,constructorId1,number1,position1],
            . . . ]
        '''
        qualiList= []
        for row in self.__csvData:
            qualiList.append([row['raceId'],row['driverId'],row['constructorId'],row['number'],
                              row['position']])
        return qualiList

    def __races(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [raceId1, year1,round1,circuitId1,name1,date1,time1],
            . . . ]
        '''
        raceList= []
        for row in self.__csvData:
            raceList.append([row['raceId'],row['year'],row['round'],row['circuitId'],
                              row['name'],row['date'],row['time']])
        return raceList

    def __results(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [raceId1, driverId1,constructorId1,number1,grid1,positionOrder1,points1,laps1,time1,
             milliseconds1,fastestLap1,rank1,fastestLapTime1,fastestLapSpeed1,statusId1],
            . . . ]
        '''
        resultsList= []
        for row in self.__csvData:
            resultsList.append([row['raceId'],row['driverId'],row['constructorId'],row['number'],
                              row['grid'],row['positionOrder'],row['points'],row['laps'],row['time'],
                              row['milliseconds'],row['fastestLap'],row['rank'],
                              row['fastestLapTime'],row['fastestLapSpeed'],row['statusId']])
        return resultsList

    def __seasons(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
        [   [year1],
            . . . ]
        '''
        seasonList= []
        for row in self.__csvData:
            seasonList.append([row['year']])
        return seasonList

    def __status(self):
        '''
        :rtype list[list[]]:

        Returns a list of the form:
         [   [statusId1,status],
            . . . ]
        '''
        statusList= []
        for row in self.__csvData:
            statusList.append([row['statusId'],row['status']])
        return statusList
    
def testRaceData():
    '''
    Test functionality of reading race data from each excel file
    '''
    files = [
    'circuits',
    'constructor_results',
    'constructor_standings',
    'constructors',
    'driver_standings',
    'drivers',
    'lap_times',
    'pit_stops',
    'qualifying',
    'races',
    'results',
    'seasons',
    'status']

    print('---Start Test---')
    for item in files:
        print(item)
        try:
            circuitData = raceData(item)
            print(circuitData.getData()[0])
        except:
            print('----------------ERROR----------------')
            pass
    print('---End Test---')

def getWeatherData(lat,lng,date):
    '''
    :type lat: float
    :type lng: float
    :type data: dict
    :rtype list: weather values at each hour interval
    '''
    # Wather hour intervals
    d= 2 #hours
    # Date start time
    secc=       str(date['second']) if int(date['second']) > int('9') else '0' + str(date['second'])
    minn=       str(date['minute']) if int(date['minute']) > int('9') else '0' + str(date['minute'])
    hour_start= str(date['hour'])   if int(date['hour'])   > int('9') else '0' + str(date['hour'])
    hour_end=   str(int(date['hour'])+d) if int(date['hour'])+d > int('9') else '0' + str(int(date['hour'])+d)
    dayy=       str(date['day'])    if int(date['day'])    > int('9') else '0' + str(date['day'])
    monn=       str(date['month'])  if int(date['month'])  > int('9') else '0' + str(date['month'])
    days=       str(date['year']) + "-" + monn  + "-" + dayy

    time_start= hour_start + ":" + minn + ":" + secc
    time_end= hour_end + ":" + minn + ":" + secc

    # Params
    params= {
        "startDateTime"    : days + "T" + time_start,
        "aggregateHours"   : "1",
        "location"         : str(lat) + "," + str(lng),
        "endDateTime"      : days + "T" + time_end,
        "unitGroup"        : "metric",
        "contentType"      : "json",
        "shortColumnNames" : "0"} # 1 : short names; 0 : long names
    # Headers
    headers= {
        'x-rapidapi-key'   : "whoops",
        'x-rapidapi-host'  : "visual-crossing-weather.p.rapidapi.com"
    }

    url = "https://visual-crossing-weather.p.rapidapi.com/history"
    test= queryData(url,headers=headers,params=params).json()["locations"][str(lat) + "," + str(lng)]["values"]
    #jprint(test)
    return test

def queryData(url, **kwargs):
        '''
        :type address: str
        :type kwards: dictionaries: headers, params
        :rtype int:

        Queries server and tracks response
        -------------------------
        Possible status values
        
        200: Good Return
        301: Redirected
        400: Bad Request
        401: Not Authenticated
        403: Forbidden
        404: Not Found
        503: Not Ready
        
        200s: Successfull
        400s/500s: Error
        ------------------------
        '''
        print("\nQuerying: ")
        headers= kwargs['headers']
        params= kwargs['params']
        response = requests.request("GET", url, headers=headers, params=params)
        print(response.url)
        t = time.gmtime(time.time())
        status = response.status_code
        print("%2d:%2d:%2d %i-%i-%i :: status:%i\n" % (t.tm_hour,t.tm_min,t.tm_sec,t.tm_mday,t.tm_mon,t.tm_year,status))
        return response

def jprint(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)

if __name__ == "__main__":
    main()
