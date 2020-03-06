import os
import pathlib
import csv

import folders

reg_dict = {'rare_neutral': {
                'folder':'r_n',
                'reg':'rare_neutral',
                'targets':[0, 5, 29, 35, 14, 2, 11, 8, 14, 8, 2, 13, 15, 0, 19, 7, 13, 11, 13, 6, 8, 2, 0, 11],
                'ignore_events_id':[],
                'general_valence':'happy',
                'rare_valence':'neutral',
                },
            'all_happy': {'folder':'a_h',
                        'reg':'all_happy',
                        'targets':[0, 5, 29, 35, 14, 1, 8, 13, 8, 14, 18, 19, 17, 20, 12, 4, 14, 19, 0, 19, 8, 13, 14],
                        'ignore_events_id':[],
                        'general_valence':'happy',
                        'rare_valence':None,
                        },
            'all_neutral': {'folder':'a_n',
                        'reg':'all_neutral',
                        'targets':[0, 5, 29, 35, 14, 1, 17, 13, 14, 2, 7, 2, 13, 14, 18, 19, 17, 8, 2, 19, 8, 13, 14],
                        'ignore_events_id':[],
                        'general_valence':'neutral',
                        'rare_valence':None,
                        },
            'rare_happy': {'folder':'r_h',
                        'reg':'rare_happy',
                        'targets':[0, 5, 29, 35, 14, 2, 13, 12, 15, 17, 4, 7, 4, 14, 18, 8, 21, 8, 18, 0, 19, 8, 13, 14],
                        'ignore_events_id':[],
                        'general_valence':'neutral',
                        'rare_valence':'happy',
                        },
            'letters_table': {'folder':'a_pixel',
                        'reg':'letters_table',
                        'targets':[0, 5, 29, 35, 14, 0, 14, 19, 8, 5, 20, 14, 3, 0, 12, 4, 14, 19, 0, 11, 8, 18, 19],
                        'ignore_events_id':[],
                        'general_valence':None,
                        'rare_valence':None,
                        }
}

p = pathlib.Path(folders.data_folder)
users = []
for a in os.listdir(p):
    try:
        float(a)
        users.append(a)
    except ValueError:
        pass


with open(folders.markup_path, 'w', newline='') as csvfile:
    fieldnames = ['user','order','folder','reg','targets','ignore_events_id', 'general_valence', 'rare_valence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    
    for user in users:
        l = list(os.walk(p/user))

        order = {
             }

        for folder in l:
            if sum([True for a in folder[2] if 'npy' in a ]):
                fname = folder[2][0]
                time = fname.split('__'[0])[-5]+ fname.split('__'[0])[-4]
                name = fname.split('_'[0])[1] + '_'+  fname.split('_'[0])[2]
                order[name] = int(time)
        order = [k for k, v in sorted(order.items(), key=lambda item: item[1])]
        for reg in order[::-1]: # changed order so first loaded file had all features
            line = {'user':user,
                    'order':order.index(reg)}
            line.update(reg_dict[reg])
            writer.writerow(line)