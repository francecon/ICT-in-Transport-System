# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:11:13 2020

@author: Francesco Conforte
"""
import pymongo as pm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def myround(x, base=5):
    return base * round(x/base)

client = pm.MongoClient('bigdatadb.polito.it',
                        ssl=True,
                        authSource = 'carsharing',
                        tlsAllowInvalidCertificates=True)
db = client['carsharing'] #Choose the DB to use
db.authenticate('ictts', 'Ictts16!')# mechanism='MONGODB-CR'



start = datetime(2017,11,1,0,0,0) #1 novembre 2017 
end = datetime(2017,11,30,23,59,59)
alternative_transp_filtered = db.get_collection('PermanentBookings').aggregate(
[
  { "$match" : {"$and": [ { "city": "Milano" },
                          { "init_date": { "$gte": start } },
                          { "final_date": { "$lte": end } },
                          { "$or":  [ {"walking.duration":{"$ne":-1} },
                                      {"driving.duration":{"$ne":-1} },
                                      {"public_transport.duration":{"$ne":-1} } 
                                      ]
                            }
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
        "duration": { "$divide":[{"$subtract":["$final_time","$init_time"]},60]},
        "alternative_duration_tr":{"$divide":["$public_transport.duration",60]}
      }
  },
  { "$match" : { "$and": [ { "duration": { "$gte": 3 } },
                            { "duration": { "$lte": 180 } },
                            { "moved": True },
                            { "alternative_duration_tr": {"$gte": 0} }
                          ] 
                }
  },
  { "$sort": {"alternative_duration_tr": 1} }
]
  )


alt_tra_filt = list(alternative_transp_filtered)

alternative_duration_pt = []
for ad in alt_tra_filt:
    alternative_duration_pt.append(ad["alternative_duration_tr"])

max_ = myround(alternative_duration_pt[-1],5)
x = np.arange(0,max_+1,step=5)
plt.figure()
plt.grid()
plt.xlabel("Public Transport Duration [min]")
plt.ylabel("Number of rentals")
plt.title("Number of rentals VS alternative public transport duration")
n, bins, patches = plt.hist(alternative_duration_pt, bins=x)

#%% Walking
alternative_transp_filtered = db.get_collection('PermanentBookings').aggregate(
[
  { "$match" : {"$and": [ { "city": "Milano" },
                          { "init_date": { "$gte": start } },
                          { "final_date": { "$lte": end } },
                          { "$or":  [ {"walking.duration":{"$ne":-1} },
                                      {"driving.duration":{"$ne":-1} },
                                      {"public_transport.duration":{"$ne":-1} } 
                                      ]
                            }
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
        "duration": { "$divide":[{"$subtract":["$final_time","$init_time"]},60]},
        "alternative_duration_tr":{"$divide":["$walking.duration",60] }
      }
  },
  { "$match" : { "$and": [ { "duration": { "$gte": 3 } },
                            { "duration": { "$lte": 180 } },
                            { "moved": True },
                            { "alternative_duration_tr": {"$gte": 1} }
                          ] 
                }
  },
  { "$sort": {"alternative_duration_tr": 1} }
]
  )


alt_tra_filt = list(alternative_transp_filtered)

alternative_duration_walking = []
for ad in alt_tra_filt:
    alternative_duration_walking.append(ad["alternative_duration_tr"])

max_ = myround(alternative_duration_walking[-1],5)
x = np.arange(0,max_+1,step=5)
plt.figure()
plt.grid()
plt.xlabel("Walking Duration [min]")
plt.ylabel("Number of rentals")
plt.title("Number of rentals VS alternative walking duration")
plt.hist(alternative_duration_walking, bins=x)




#%%Driving

alternative_transp_filtered = db.get_collection('PermanentBookings').aggregate(
[
  { "$match" : {"$and": [ { "city": "Milano" },
                          { "init_date": { "$gte": start } },
                          { "final_date": { "$lte": end } },
                          { "$or":  [ {"walking.duration":{"$ne":-1} },
                                      {"driving.duration":{"$ne":-1} },
                                      {"public_transport.duration":{"$ne":-1} } 
                                      ]
                            }
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
        "duration": { "$divide":[{"$subtract":["$final_time","$init_time"]},60]},
        "alternative_duration_tr":{"$divide":["$driving.duration",60]}
      }
  },
  { "$match" : { "$and": [ { "duration": { "$gte": 3 } },
                            { "duration": { "$lte": 180 } },
                            { "moved": True },
                            { "alternative_duration_tr": {"$gte": 1} }
                          ] 
                }
  },
  { "$sort": {"alternative_duration_tr": 1} }
]
  )


alt_tra_filt = list(alternative_transp_filtered)

alternative_duration_driving = []
for ad in alt_tra_filt:
    alternative_duration_driving.append(ad["alternative_duration_tr"])

max_ = myround(alternative_duration_driving[-1],5)
x = np.arange(0,max_+1,step=5)
plt.figure()
plt.grid()
plt.xlabel("Driving Duration [min]")
plt.ylabel("Number of rentals")
plt.title("Number of rentals VS alternative driving duration")
plt.hist(alternative_duration_driving, bins=x)

















