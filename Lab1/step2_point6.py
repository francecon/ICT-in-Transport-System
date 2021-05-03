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


start = datetime(2017,11,1,0,0,0) #1 novembre 2017 unixtime: 1509494400 #1 novembre 2017
end = datetime(2017,11,1,23,59,59)

parkings_per_hour = db.get_collection("PermanentParkings").aggregate(
        [
        { "$match" : {"$and": [ { "city": "Milano" },
                                { "init_date": { "$gte":start } },
                                { "final_date": { "$lte":end } },
                                ]
                    } 
        },
        { "$project": {
                "_id": 1,
                "long": {"$arrayElemAt": [ "$loc.coordinates", 0]},
                "lat": {"$arrayElemAt": [ "$loc.coordinates", 1]},
                "date_parts": { "$dateToParts": { "date": "$init_date" } },
                "dayofweek": {"$dayOfWeek": "$init_date" },
            }
        },
        { "$match" : {"$and": [ { "date_parts.hour": { "$gte":8 } },
                                { "date_parts.hour": { "$lte":19 } },
                                {"dayofweek": {"$gte":1} },
                                {"dayofweek": {"$gte":5} },
                                 ]
                        }
        },
        { "$group": {
                  "_id": {
                    "day": "$dayofweek",
                    "hour": "$date_parts.hour"
                    },
                  }
        },
        ]
    )

with open("./working_days8-19.csv", "w") as pks:
    pks.write("latitude,longitude")
    for park in (list(parkings_per_hour)):
        pks.write(f"\n{park['lat']},{park['long']}")


