import warnings

warnings.filterwarnings('ignore')

from nilmtk.dataset_converters.redd import convert_redd

# filename = "/Users/user/PycharmProjects/honors2/redd.hdf5"

filename = "C:\\Users\\davwang\\Desktop\\nilmtk\\nilmtk\\dataset_converters\\redd\\redd.hdf5"

convert_redd.convert_redd("C:\\Users\\davwang\\Desktop\\low_freq", filename)
