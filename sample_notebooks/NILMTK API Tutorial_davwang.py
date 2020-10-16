import warnings

warnings.filterwarnings("ignore")
from nilmtk.api import API
from nilmtk.disaggregate import Hart85, CO, Mean, FHMMExact
from nilmtk_contrib.disaggregate import DAE, DSC, AFHMM, AFHMM_SAC, Seq2Point, Seq2Seq, RNN, WindowGRU, ModelTestS2P, ModelTestS2S

redd = {
    'power':
        {
            'mains': ['apparent', 'active'],
            'appliance': ['apparent', 'active']
        },
    'sample_rate': 60,

    # 'appliances': ['fridge', 'microwave', 'light', 'sockets', 'dish washer', 'washer dryer'],
    # 'appliances': ['fridge','microwave'],
    'appliances': ['fridge'],
    'methods': {
        # "Hart85": Hart85({}),  #还有问题
        # "CO": CO({}),
        # 'Mean': Mean({}),
        # "AFHMM": AFHMM({'default_num_states': 2}),    #还有问题
        # "AFHMM_SAC": AFHMM_SAC({'default_num_states': 2}),   #还有问题
        # 'DAE': DAE({'n_epochs': 5, 'batch_size': 32}),
        # 'DSC': DSC({}),   #还有问题
        # "FHMMExact": FHMMExact({'num_of_states': 2}),
        # 'RNN':RNN({'n_epochs':5,'batch_size':32}),
        #  'Seq2Point':Seq2Point({'n_epochs':5,'batch_size':32}),
        # 'Seq2Seq': Seq2Seq({'n_epochs': 5, 'batch_size': 32}),
        # 'WindowGRU':WindowGRU({'n_epochs':5,'batch_size':32}),
        # 'ModelTestS2P': ModelTestS2P({}),
        'ModelTestS2S': ModelTestS2S({}),
    },
    'train': {
        'datasets': {
            'redd': {
                'path': 'C:/Users/davwang/Desktop/redd.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-17',
                        'end_time': '2011-04-22'
                    },
                }

            }
        }
    },
    'test': {
        'datasets': {
            'redd': {
                'path': 'C:/Users/davwang/Desktop/redd.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-18',
                        'end_time': '2011-04-22'
                    },
                }
            }
        },
        # 'metrics': ['rmse','mae','f1score']
        'metrics': ['rmse']
    }
}

ukdale = {
    'power':
        {
            'mains': ['apparent', 'active'],
            'appliance': ['apparent', 'active']
        },
    'sample_rate': 60,

    'appliances': ['fridge'],
    'methods': {
        # "Hart85": Hart85({}),  #还有问题
        # "CO": CO({}),
        'Mean': Mean({}),
        # "AFHMM": AFHMM({'default_num_states': 2}),    #还有问题
        # "AFHMM_SAC": AFHMM_SAC({'default_num_states': 2}),   #还有问题
        'DAE': DAE({'n_epochs': 5, 'batch_size': 32}),
        # 'DSC': DSC({}),   #还有问题
        # "FHMMExact": FHMMExact({'num_of_states': 2}),
        # 'RNN':RNN({'n_epochs':5,'batch_size':32}),
        #  'Seq2Point':Seq2Point({'n_epochs':5,'batch_size':32}),
        # 'Seq2Seq': Seq2Seq({'n_epochs': 5, 'batch_size': 32}),
        # 'WindowGRU':WindowGRU({'n_epochs':5,'batch_size':32}),
    },
    'train': {
        'datasets': {
            'UKDALE': {
                'path': 'C:/Users/davwang/Desktop/ukdale.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2016-01-05',
                        'end_time': '2016-03-05'
                    },
                }

            }
        }
    },
    'test': {
        'datasets': {
            'UKDALE': {
                'path': 'C:/Users/davwang/Desktop/ukdale.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2016-01-05',
                        'end_time': '2016-01-15'
                    },
                }
            }
        },
        # 'metrics': ['rmse', 'mae', 'f1score', 'relative_error']
        'metrics': ['rmse']
    }
}


ampds2redd = {
    'power':
        {
            'mains': ['apparent', 'active'],
            'appliance': ['apparent', 'active']
        },
    'sample_rate': 60,

    'appliances': ['fridge'],
    'methods': {
        # "Hart85": Hart85({}),  #还有问题
        # "CO": CO({}),
        'Mean': Mean({}),
        # "AFHMM": AFHMM({'default_num_states': 2}),    #还有问题
        # "AFHMM_SAC": AFHMM_SAC({'default_num_states': 2}),   #还有问题
        # 'DAE': DAE({'n_epochs': 5, 'batch_size': 32}),
        # 'DSC': DSC({}),   #还有问题
        # "FHMMExact": FHMMExact({'num_of_states': 2}),
        # 'RNN':RNN({'n_epochs':5,'batch_size':32}),
        #  'Seq2Point':Seq2Point({'n_epochs':5,'batch_size':32}),
        'Seq2Seq': Seq2Seq({'n_epochs': 5, 'batch_size': 32}),
        # 'WindowGRU':WindowGRU({'n_epochs':5,'batch_size':32}),
    },
    'train': {
        'datasets': {
            'ampds': {
                'path': 'C:/Users/davwang/Desktop/ampds.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2013-01-05',
                        'end_time': '2013-01-06'
                    },
                }

            }
        }
    },
    'test': {
        'datasets': {
            'REDD': {
                'path': 'C:/Users/davwang/Desktop/redd.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-17',
                        'end_time': '2011-04-27'
                    }
                }
            }
        },
        'metrics': ['mae', 'rmse']
    }
    # 'test': {
    #     'datasets': {
    #         'ampds': {
    #             'path': 'C:/Users/davwang/Desktop/ampds.hdf5',
    #             'buildings': {
    #                 1: {
    #                     'start_time': '2013-01-05',
    #                     'end_time': '2013-01-08'
    #                 },
    #             }
    #         }
    #     },
    #     # 'metrics': ['rmse', 'mae', 'f1score', 'relative_error']
    #     'metrics': ['rmse']
    # }
}

ukdale2redd = {
    'power': {'mains': ['apparent', 'active'], 'appliance': ['apparent', 'active']},
    'sample_rate': 60,
    # 'appliances': ['washing machine', 'fridge'],
    'appliances': ['fridge','microwave'],
    'artificial_aggregate': True,
    'chunksize': 20000,
    'DROP_ALL_NANS': False,
    'methods': {
        # "Mean": Mean({}),
        # "FHMMExact": FHMMExact({'num_of_states': 2}),
        'CO': CO({}),
        'DAE': DAE({'n_epochs': 5, 'batch_size': 32}),
    },
    'train': {
        'datasets': {
            'UKDALE': {
                'path': 'C:/Users/davwang/Desktop/ukdale.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2017-01-05',
                        'end_time': '2017-03-05'
                    },
                }
            },
        }
    },
    'test': {
        'datasets': {
            'REDD': {
                'path': 'C:/Users/davwang/Desktop/redd.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-17',
                        'end_time': '2011-04-27'
                    }
                }
            }
        },
        'metrics': ['mae', 'rmse']
    }
}

greend2redd = {
    'power':
        {
            'mains': ['apparent', 'active'],
            'appliance': ['apparent', 'active']
        },
    'sample_rate': 60,

    'appliances': ['fridge'],
    'methods': {
        # "Hart85": Hart85({}),  #还有问题
        # "CO": CO({}),
        'Mean': Mean({}),
        # "AFHMM": AFHMM({'default_num_states': 2}),    #还有问题
        # "AFHMM_SAC": AFHMM_SAC({'default_num_states': 2}),   #还有问题
        'DAE': DAE({'n_epochs': 5, 'batch_size': 32}),
        # 'DSC': DSC({}),   #还有问题
        # "FHMMExact": FHMMExact({'num_of_states': 2}),
        # 'RNN':RNN({'n_epochs':5,'batch_size':32}),
        #  'Seq2Point':Seq2Point({'n_epochs':5,'batch_size':32}),
        # 'Seq2Seq': Seq2Seq({'n_epochs': 5, 'batch_size': 32}),
        # 'WindowGRU':WindowGRU({'n_epochs':5,'batch_size':32}),
    },
    'train': {
        'datasets': {
            'greend2redd': {
                'path': 'C:/Users/davwang/Desktop/greend.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2014-01-05',
                        'end_time': '2014-01-06'
                    },
                }

            }
        }
    },
    'test': {
        'datasets': {
            'REDD': {
                'path': 'C:/Users/davwang/Desktop/redd.hdf5',
                'buildings': {
                    1: {
                        'start_time': '2011-04-17',
                        'end_time': '2011-04-27'
                    }
                }
            }
        },
        'metrics': ['mae', 'rmse','f1score']
    }
    # 'test': {
    #     'datasets': {
    #         'ampds': {
    #             'path': 'C:/Users/davwang/Desktop/ampds.hdf5',
    #             'buildings': {
    #                 1: {
    #                     'start_time': '2013-01-05',
    #                     'end_time': '2013-01-08'
    #                 },
    #             }
    #         }
    #     },
    #     # 'metrics': ['rmse', 'mae', 'f1score', 'relative_error']
    #     'metrics': ['rmse']
    # }
}

api_res = API(redd)             # tested passed
# api_res = API(ukdale)           # tested passed
# api_res = API(ampds2redd)       # tested passed
# api_res = API(ukdale2redd)      # tested passed



api_res.errors

api_res.errors_keys
