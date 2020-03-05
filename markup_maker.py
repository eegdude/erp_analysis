import os
import pathlib
import csv

import folders

p = pathlib.Path(folders.data_folder)
users = [str(a) for a in range(24, 37)]

with open(folders.markup_path, 'w', newline='') as csvfile:
    fieldnames = ['user','order','folder','reg','targets','ignore_events_id', 'general_valence', 'rare_valence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    
    for user in users:
        l = list(os.walk(p/user))
        order = {
            'faces': None,
            'facesnoise': None,
            'letters': None,
            'noise': None,
             }

        for folder in l:
            if sum([True for a in folder[2] if 'npy' in a ]):
                fname = [a for a in folder[2] if 'npy' in a ][0]
                time = fname.split('__')[1].split('_')[1] + fname.split('__')[1].split('_')[2]
                name = fname.split('_')[2]
                order[name] = int(time)
        order = [k for k, v in sorted(order.items(), key=lambda item: item[1])]
        
        writer.writerow({'user':user,
                         'order':order.index('faces'),
                         'folder':'fcs',
                         'reg':'faces',
                         'targets':[0, 5, 36, 41, 21, 38, 14, 4, 20, 17, 13, 18, 2, 8, 4, 14, 2, 4, 37],
                         'ignore_events_id':[],
                         'general_valence':None,
                         'rare_valence':None,
                         })
        
        writer.writerow({'user':user,
                         'order':order.index('facesnoise'),
                         'folder':'fn',
                         'reg':'facesnoise',
                         'targets':[0, 5, 36, 41, 21, 38, 14, 4, 20, 17, 13, 18, 2, 8, 4, 14, 2, 4, 37],
                         'ignore_events_id':[],
                         'general_valence':None,
                         'rare_valence':None,
                         })

        writer.writerow({'user':user,
                         'order':order.index('letters'),
                         'folder':'ltrs',
                         'reg':'letters',
                         'targets':[0, 5, 36, 41, 21, 38, 14, 4, 20, 17, 13, 18, 2, 8, 4, 14, 2, 4, 37],
                         'ignore_events_id':[],
                         'general_valence':None,
                         'rare_valence':None,
                         })

        writer.writerow({'user':user,
                         'order':order.index('noise'),
                         'folder':'ns',
                         'reg':'noise',
                         'targets':[0, 5, 36, 41, 21, 38, 14, 4, 20, 17, 13, 18, 2, 8, 4, 14, 2, 4, 37],
                         'ignore_events_id':[],
                         'general_valence':None,
                         'rare_valence':None,
                         })