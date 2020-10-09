import warnings
warnings.filterwarnings("ignore")
from nilmtk.api import API
from nilmtk.disaggregate import Hart85, CO, Mean, FHMMExact
from nilmtk_contrib.disaggregate import DAE, DSC, AFHMM, AFHMM_SAC, Seq2Point, Seq2Seq, RNN, WindowGRU

redd = {
    'power':
        {
            'mains': ['apparent', 'active'],
            'appliance': ['apparent', 'active']
        },
    'sample_rate': 60,

    'appliances': ['fridge', 'microwave', 'light', 'sockets', 'dish washer', 'washer dryer'],
    'methods': {
        # "Hart85": Hart85({}),  #还有问题
        "CO": CO({}),
        'Mean': Mean({}),
        # "AFHMM": AFHMM({'default_num_states': 2}),    #还有问题
        # "AFHMM_SAC": AFHMM_SAC({'default_num_states': 2}),   #还有问题
        # 'DAE': DAE({'n_epochs': 5, 'batch_size': 32}),
        # 'DSC': DSC({}),   #还有问题
        # "FHMMExact": FHMMExact({'num_of_states': 2}),
        # 'RNN':RNN({'n_epochs':5,'batch_size':32}),
        # 'Seq2Point':Seq2Point({'n_epochs':5,'batch_size':32}),
        # 'Seq2Seq': Seq2Seq({'n_epochs': 5, 'batch_size': 32}),
        # 'WindowGRU':WindowGRU({'n_epochs':5,'batch_size':32}),
    },
    'train': {
        'datasets': {
            'redd': {
                'path': 'C:/Users/davwang/Desktop/nilmtk-contrib/dataset/redd.hdf5',
                'buildings': {
                    2: {
                        'start_time': '2011-04-17',
                        'end_time': '2011-04-27'
                    },
                }

            }
        }
    },
    'test': {
        'datasets': {
            'redd': {
                'path': 'C:/Users/davwang/Desktop/nilmtk-contrib/dataset/redd.hdf5',
                'buildings': {
                    3: {
                        'start_time': '2011-04-28',
                        'end_time': '2011-05-17'
                    },
                }
            }
        },
        'metrics': ['rmse', 'mae', 'f1score']
    }
}

api_res = API(redd)

api_res.errors

api_res.errors_keys
