import pymongo as pm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

client = pm.MongoClient('bigdatadb.polito.it',
                        ssl=True,
                        authSource = 'carsharing',
                        tlsAllowInvalidCertificates=True)
db = client['carsharing'] #Choose the DB to use
db.authenticate('ictts', 'Ictts16!')#, mechanism='MONGODB-CR') #authentication

# start = datetime(2017,11,1,0,0,0) #1 novembre 2017 unixtime: 1509494400 #1 novembre 2017
# end = datetime(2017,11,30,23,59,59) #30 novembre 2017 unixtime: 1512086399 #30 novembre 2017

# print(start.weekday())

# cities = ["Milano", "Calgary", "Amsterdam"]
# collections = ["PermanentBookings", "PermanentParkings"]
# for c in cities:
#    for collection in collections:
#       duration = db.get_collection(collection).aggregate(
#          [
#             { "$match" : {"$and": [{"city": c }, {"init_date": {"$gte":start}}, {"final_date": {"$lte":end}}]} },
#             {
#                "$project": {
#                   "_id": 1,
#                   "city": 1,
#                   "duration": { "$divide": [ { "$subtract": ["$final_time", "$init_time"] }, 60 ] },
#                }
#             },
#             { "$group": {
#                   "_id": "$duration",
#                   "tot_rentals": {"$sum": 1}
#             }
#             },
#             { "$sort": {"_id": 1} }
#          ]
#       )

#       value_id = []
#       value_totrentals = []
#       for i in list(duration):
#          value_id.append(i["_id"])
#          value_totrentals.append(i["tot_rentals"])

#       tot_totrentals = sum(value_totrentals)

#       cumulative_totrentals = [value_totrentals[0]]
#       temp = 0
#       for i in range(1,len(value_totrentals)):
#          temp = cumulative_totrentals[i-1] + value_totrentals[i]
#          cumulative_totrentals.append(temp)

#       cumulative_totrentals = np.divide(cumulative_totrentals, tot_totrentals)

#       plt.plot(value_id,cumulative_totrentals)
#       plt.xscale("log")
#       plt.xlabel("Duration [min]")
#       plt.ylabel("CDF")

#    plt.title(c)
#    plt.grid()
#    plt.legend(collections)
#    plt.show()

start = datetime(2018,1,1,0,0,0) #1 novembre 2017 unixtime: 1509494400 #1 novembre 2017
end = datetime(2018,1,30,23,59,59)
days = [1,2,3,4,5,6,7]
days_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
#collections = ["PermanentBookings", "PermanentParkings"]
#collections = ["PermanentBookings"]
# for collection in collections:
#    for day in days:
#       duration = db.get_collection(collection).aggregate(
#                [
#                   { "$match" : {"$and": [ { "city": "Torino" },
#                                           { "init_date": { "$gte":start } },
#                                           { "final_date": { "$lte":end } },
#                                        ]
#                               } 
#                   },
#                   {
#                      "$project": {
#                         "_id": 1,
#                         "city": 1,
#                         "weekday": { "$isoDayOfWeek": "$init_date" },
#                         "duration": { "$divide": [ { "$subtract": ["$final_time", "$init_time"] }, 60 ] },
#                      }
#                   },
#                   { "$match" : {"weekday": day}},
#                   { "$group": {
#                         "_id": "$duration",
#                         "tot_rentals": {"$sum": 1}
#                   }
#                   },
#                   { "$sort": {"_id": 1} }
#                ]
#             )

#       value_id = []
#       value_totrentals = []
#       for i in list(duration):
#          value_id.append(i["_id"])
#          value_totrentals.append(i["tot_rentals"])

#       tot_totrentals = sum(value_totrentals)

#       cumulative_totrentals = [value_totrentals[0]]
#       temp = 0
#       for i in range(1,len(value_totrentals)):
#          temp = cumulative_totrentals[i-1] + value_totrentals[i]
#          cumulative_totrentals.append(temp)

#       cumulative_totrentals = np.divide(cumulative_totrentals, tot_totrentals)

#       plt.plot(value_id,cumulative_totrentals)
#       plt.xscale("log")
#       plt.xlabel("Duration [min]")
#       plt.ylabel("CDF")

#    plt.title("CDF for weekday")
#    plt.grid()
#    plt.legend(days_labels)
#    plt.show()

#%% Point 4
# for collection in collections:
#    cars_per_hour = db.get_collection(collection).aggregate(
#          [
#             { "$match" : {"$and": [ { "city": "Milano" },
#                                     { "init_date": { "$gte":start } },
#                                     { "final_date": { "$lte":end } },
#                                  ]
#                         } 
#             },
#             { "$project": {
#                   "_id": 1,
#                   "city": 1,
#                   "date_parts": { "$dateToParts": { "date": "$init_date" } },
#                }
#             },
#             { "$group": {
#                   "_id": {
#                   "day": "$date_parts.day",
#                   "hour": "$date_parts.hour"
#                   },
#                   "tot_rentals": {"$sum": 1}
#             }
#             },
#             { "$sort": {"_id": 1} }
#          ]
#       )
      
#    value_id = []
#    value_totrentals = []
#    value_id_2 = []
#    value_totrentals_2 = []

#    for i in list(cars_per_hour):
#       if i["_id"]["day"] < 3:
#          value_id.append(i["_id"]["day"] + i["_id"]["hour"]/24)
#          value_totrentals.append(i["tot_rentals"])
#       elif i["_id"]["day"] > 12:
#          value_id_2.append(i["_id"]["day"] + i["_id"]["hour"]/24)
#          value_totrentals_2.append(i["tot_rentals"])
      
#    plt.plot(value_id,value_totrentals, color="red",label='unfiltered')
#    plt.plot(value_id_2,value_totrentals_2, color='red')

cars_per_hour_filtered = db.get_collection("PermanentBookings").aggregate(
      [
         { "$match" : {"$and": [ { "city": "Milano" },
                                 { "init_date": { "$gte": start } },
                                 { "final_date": { "$lte": end } },
                              ]
                     } 
         },
         { "$project": {
               "_id": 1,
               "city": 1,
               "moved": { "$ne": [
                     {"$arrayElemAt": [ "$origin_destination.coordinates", 0]},
                     {"$arrayElemAt": [ "$origin_destination.coordinates", 1]} 
                  ]
               },
               "duration": { "$divide": [ { "$subtract": ["$final_time", "$init_time"] }, 60 ] },
               "date_parts": { "$dateToParts": { "date": "$init_date" } },
            }
         },
         { "$match" : { "$and": [ { "duration": { "$gte": 3 } },
                                    { "duration": { "$lte": 180 } },
                                    { "moved": True }
                                 ] 
                        }
         },
         { "$group": {
               "_id": {
                  "day": "$date_parts.day",
                  "month":"$date_parts.month",
                  "hour": "$date_parts.hour"
               },
               "tot_rentals": {"$sum": 1}
         }
         },
         { "$sort": {"_id": 1} }
      ]
   )
cars_per_hour_filtered = list(cars_per_hour_filtered)
 
   # value_id = []
   # value_totrentals = []
   # value_id_2 = []
   # value_totrentals_2 = []
   
   # for i in list(cars_per_hour_filtered):
   #    if i["_id"]["day"]<3:
   #       value_id.append(i["_id"]["day"] + i["_id"]["hour"]/24)
   #       value_totrentals.append(i["tot_rentals"])

   #    elif i["_id"]["day"]>12:
   #       value_id_2.append(i["_id"]["day"] + i["_id"]["hour"]/24)
   #       value_totrentals_2.append(i["tot_rentals"])

   # plt.plot(value_id,value_totrentals,color='blue',label='filtered')
   # plt.plot(value_id_2,value_totrentals_2,color='blue')
   
   # plt.xticks(np.arange(1, 32, step=1))
   # plt.xlabel("Days of November 2017")
   # plt.ylabel("Number of " + collection)
   # plt.grid()
   # plt.legend()
   # plt.title("Number of " + collection + " per hours")
   # plt.show()


# #%% Point 5

# for collection in collections:
#    avg_duration_per_day = db.get_collection(collection).aggregate(
#          [
#             { "$match" : {"$and": [ { "city": "Calgary" },
#                                     { "init_date": { "$gte":start } },
#                                     { "final_date": { "$lte":end } },
#                                  ]
#                         } 
#             },
#             { "$project": {
#                   "_id": 1,
#                   "city": 1,
#                   # "moved": { "$ne": [
#                   #       {"$arrayElemAt": [ "$origin_destination.coordinates", 0]},
#                   #       {"$arrayElemAt": [ "$origin_destination.coordinates", 1]} 
#                   #    ]
#                   # },
                  
#                   "moved": { "$cond": [ 
#                      { "$ne": ["$origin_destination", "$undefined"] },     
#                      { "$ne": [
#                         {"$arrayElemAt": [ "$origin_destination.coordinates", 0]},
#                         {"$arrayElemAt": [ "$origin_destination.coordinates", 1]} 
#                      ]},
#                      True
#                   ]},
#                   "duration": { "$divide": [ { "$subtract": ["$final_time", "$init_time"] }, 60 ] },
#                   "date_parts": { "$dateToParts": { "date": "$init_date" } },
#                }
#             },
#             { "$match" : { "$and": [ { "duration": { "$gte": 3 } },
#                                      { "duration": { "$lte": 180 } },
#                                      { "moved": True }
#                                     ] 
#                          }
#             },
#             { "$group": {
#                   "_id": "$date_parts.day",
#                   "avg_duration": {"$avg": "$duration"},
#                   "std_dev": {"$stdDevPop": "$duration"}
#             }
#             },
#             { "$sort": {"_id": 1} }
#          ]
#       )

#    value_id = []
#    value_duration = []
#    value_std = []
   
#    value_id_2 = []
#    value_duration_2 = []
#    value_std_2 = []
   
   
#    for i in list(avg_duration_per_day):
#       if i["_id"]<3:
#          value_id.append(i["_id"])
#          value_duration.append(i["avg_duration"])
#          value_std.append(i["std_dev"])
#       elif i["_id"]>12:
#          value_id_2.append(i["_id"])
#          value_duration_2.append(i["avg_duration"])
#          value_std_2.append(i["std_dev"])


#    plt.figure()         
#    plt.plot(value_id,value_duration,color='blue',label='avg duration')
#    plt.plot(value_id_2,value_duration_2,color='blue')
#    plt.plot(value_id,value_std,color='red',label='standard deviation')
#    plt.plot(value_id_2,value_std_2,color='red')

#    median_duration_per_day = db.get_collection(collection).aggregate(
#          [
#             { "$match" : {"$and": [ { "city": "Calgary" },
#                                     { "init_date": { "$gte":start } },
#                                     { "final_date": { "$lte":end } },
#                                  ]
#                         } 
#             },
#             { "$project": {
#                   "_id": 1,
#                   "city": 1,
#                   # "moved": { "$ne": [
#                   #       {"$arrayElemAt": [ "$origin_destination.coordinates", 0]},
#                   #       {"$arrayElemAt": [ "$origin_destination.coordinates", 1]} 
#                   #    ]
#                   # },
#                   "moved": { "$cond": [ 
#                      { "$ne": ["$origin_destination", "$undefined"] },
#                      { "$ne": [
#                         {"$arrayElemAt": [ "$origin_destination.coordinates", 0]},
#                         {"$arrayElemAt": [ "$origin_destination.coordinates", 1]} 
#                      ]},
#                      True,
#                   ]},
#                   "duration": { "$divide": [ { "$subtract": ["$final_time", "$init_time"] }, 60 ] },
#                   "date_parts": { "$dateToParts": { "date": "$init_date" } }
#                }
#             },
#             { "$match" : { "$and": [ { "duration": { "$gte": 3 } },
#                                      { "duration": { "$lte": 180 } },
#                                      { "moved": True }
#                                     ] 
#                          }
#             },
#             { "$group": {
#                   "_id": "$date_parts.day",
#                   "count": { "$sum": 1 },
#                   "values": { 
#                      "$push": "$duration" #creates a list with the duration values for each rental, grouped by day
#                   }
#                }
#             },
#             { 
#                "$unwind": "$values"  #creates a document per each element in the list containing the durations
#             },
#             { "$sort": {"values": 1} },
#             { 
#                "$project": { 
#                   "_id": 1,
#                   "count": 1, 
#                   "values": 1, 
#                   "midpoint": { 
#                      "$divide": [
#                         { "$subtract": [ "$count", 1 ] }, 2  # find the index of the median value of the durations list (it can be a float)
#                      ] 
#                   }
#                }
#             },
#             {
#                "$project": {
#                   "_id": 1,
#                   "count": 1,
#                   "values": 1,
#                   "midpoint": 1,
#                   "high": { 
#                      "$ceil": "$midpoint"    #round the index to the highest value (to avoid that midpoint is a float)
#                   },
#                   "low": {
#                      "$floor": "$midpoint"   #round the index to the lowest value (to avoid that midpoint is a float)
#                   }
#                }
#             },
#             { 
#                "$group": {
#                   "_id": "$_id",
#                   "values": {
#                      "$push": "$values"   #we push again the values of durations inside lists
#                   }, 
#                   "high": {
#                      "$avg": "$high"
#                   },
#                   "low": {
#                      "$avg": "$low"
#                   }
#                }
#             },
#             {
#                "$project": {
#                   "_id": 1,
#                   "beginValue": {
#                      "$arrayElemAt": ["$values" , "$high"]  #we take the element in the array at the position of the index rounded to the highest value
#                   } ,
#                   "endValue": {
#                      "$arrayElemAt": ["$values" , "$low"]   #we take the element in the array at the position of the index rounded to the lowest value
#                   },
#                   "85_percentile": {"$arrayElemAt": ["$values", {"$floor": {"$multiply": [0.85, {"$size": "$values"}]}}]}
#                }
#             },
#             {
#                "$project": {
#                   "_id": 1,
#                   "median": {
#                      "$avg": ["$beginValue" , "$endValue"]  #we compute the average between the 2 central values, in case the number of elements is odd, the 2 values will be the same
#                   },
#                   "85_percentile": 1      
#                }
#             },
#             { "$sort": {"_id": 1} }
#          ]
#       )

   

#    value_median = []
#    value_perc = []
#    value_median_2 = []
#    value_perc_2 = []
   
#    for i in list(median_duration_per_day):
#       if i["_id"]<3:
#          value_median.append(i["median"])
#          value_perc.append(i["85_percentile"])
#       elif i["_id"]>12:
#          value_median_2.append(i["median"])
#          value_perc_2.append(i["85_percentile"])


#    plt.plot(value_id,value_median,color='green',label='median')
#    plt.plot(value_id_2,value_median_2,color='green')
#    plt.plot(value_id, value_perc, color='orange',label='85th percentile')
#    plt.plot(value_id_2, value_perc_2, color='orange')

#    plt.xticks(np.arange(1, 31, step=1))
#    plt.xlabel("Days of November 2017")
#    plt.ylabel("Avg duration for " + collection)
#    plt.grid()
#    plt.legend(ncol=2)
#    plt.title("Avg duration for " + collection + " per day")
#    plt.show()

# #%% Point 6
# #Consider one city of your collection and check the position of the cars when parked, and compute
# #the density of cars during different hours of the day.




