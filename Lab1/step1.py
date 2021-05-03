#%%
import pymongo as pm
from datetime import datetime

client = pm.MongoClient('bigdatadb.polito.it',
                        ssl=True,
                        authSource = 'carsharing',
                        tlsAllowInvalidCertificates=True)
db = client['carsharing'] #Choose the DB to use
db.authenticate('ictts', 'Ictts16!')# mechanism='MONGODB-CR') #authentication
# Bookings_collection = db['PermanentBookings'] # Collection for Car2go to use

#%% Step1: How many documents are present in each collection?
collectionName = ['ActiveBookings','ActiveParkings','PermanentBookings',
                  'PermanentParkings','enjoy_ActiveBookings',
                  'enjoy_ActiveParkings','enjoy_PermanentBookings',
                  'enjoy_PermanentParkings']

for i in collectionName:
    collection=db.get_collection(i)
    print(i + ": " + str(collection.estimated_document_count()))

#%% Step1: For which cities the system is collecting data?
print('Cities Car2Go: ' + 
      str(db.get_collection('PermanentBookings').distinct("city")))
print('Cities Enjoy: ' + 
      str(db.get_collection('enjoy_PermanentBookings').distinct("city")))

#%% Step1: When the collection started? When the collection ended?
timestamps_sorted_car2go = db.get_collection('PermanentBookings').find().sort(
    'init_time', pm.DESCENDING).distinct('init_time')
start_car2go=timestamps_sorted_car2go[-1]
end_car2go=timestamps_sorted_car2go[0]

timestamps_sorted_enjoy = db.get_collection('enjoy_PermanentBookings').find().sort(
    'init_time', pm.DESCENDING).distinct('init_time')
start_enjoy=timestamps_sorted_enjoy[-1]
end_enjoy=timestamps_sorted_enjoy[0]

ts_car2go = int(start_car2go) # initial time
te_car2go = int(end_car2go) #ending time

ts_enjoy = int(start_enjoy) # initial time
te_enjoy = int(end_enjoy) #ending time

print("UTC time of the first car collected in car2go")
print(datetime.utcfromtimestamp(ts_car2go).strftime('%Y-%m-%d %H:%M:%S'))

print("UTC time of the last car collected in car2go")
print(datetime.utcfromtimestamp(te_car2go).strftime('%Y-%m-%d %H:%M:%S'))

less = db.get_collection('PermanentBookings').find_one({'init_time':start_car2go})
date = less.get('init_date')
city_car2go = less.get('city')
print("Local time of the first car collected for car2go in " + str(city_car2go) + ": ")
print(date)

print("UTC time of the first car collected in enjoy")
print(datetime.utcfromtimestamp(ts_enjoy).strftime('%Y-%m-%d %H:%M:%S'))
print("UTC time of the last car collected in enjoy")
print(datetime.utcfromtimestamp(te_enjoy).strftime('%Y-%m-%d %H:%M:%S'))

less_enjoy = db.get_collection('enjoy_PermanentBookings').find_one({'init_time':start_enjoy})
city_enjoy = less_enjoy.get('city')
date_enjoy = less_enjoy.get('init_date')
print("Local time of the first car collected for enjoy in " + str(city_enjoy) + ": ")
print(date_enjoy)

#%% Step1: How many cars are available in each city?
cities = ['Milano','Calgary','Amsterdam']
start = datetime(2017,11,1,0,0,0) #1 novembre 2017 unixtime: 1509494400 #1 novembre 2017
end = datetime(2017,11,30,23,59,59) #30 novembre 2017 unixtime: 1512086399 #30 novembre 2017
for c in cities:
    car=db.PermanentBookings.distinct("plate", {"city": c})
    print("Cars in " + c + ": " + str(len(car)))
#%% Step1: How many bookings have been recorded on the November 2017 in each city?
    bookings = db.PermanentBookings.count_documents(
        {"$and": [
                {"city":c}, 
                {"init_date": {"$gte":start} },
                {"final_date": {"$lte":end} }
                ]
            }
        )
    print("Booked cars in november 2017 in " + c + " are: " + str(bookings))
#%% Step1: How many bookings have also the alternative transportation modes recorded in each city?
    alternative = db.PermanentBookings.count_documents(
                    {"$and": [ {"city":c},
                    {"$or":  [ {"walking.distance":{"$ne":-1} },
                               {"driving.distance":{"$ne":-1} },
                               {"public_transport.distance":{"$ne":-1} } 
                               ]
                     }
                    ]
                     }
                    )
    print("Alternative transportation mode for " + c + ": " + str(alternative))

#%% Prova
# h=[]
# for c in cities: 
#     s = db.get_collection('PermanentBookings').aggregate(
#         [
#             { "$match" : {"$and": [ { "city": c },
#                                     { "init_date": { "$gte":start } },
#                                     { "final_date": { "$lte":end } },
#                                  ]
#                         }
#              }, 
#             { "$project": {
#                   "_id": 1,
#                   "city": 1,
#                }
#             },
#             { "$group": {
#                   "_id": "$city",
#                   "count": { "$sum": 1 }
#                }
#             },
#         ])
#     h.append(list(s))
    

             