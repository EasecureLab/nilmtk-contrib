# IMPORTS=======================================================================
from pylab import rcParams
import matplotlib.pyplot as plt

import nilmtk as ntk
import nilmtk.disaggregate as ntkd
import nilmtk.metrics as ntkm

rcParams['figure.figsize'] = (14, 6)
plt.style.use('ggplot')

# CONSTANTS=====================================================================
h5_path = '/home/t7/Dropbox/Documents/TUDelft/Thesis/Datasets/DRED/DRED.h5'
h5_path = r'C:\Users\davwang\Desktop\nilmtk\nilmtk\dataset_converters\dred\DRED.h5'

# Load Data=====================================================================
dred = ntk.DataSet(h5_path)
# dred.set_window(start=None, end='2015-07-10 00:00:00')

elec = dred.buildings[1].elec
mains = elec.mains()

# Train==========================================================================
co = ntk.disaggregate.CombinatorialOptimisation()
co.train(elec)

# Disaggregate====================================================================
output = ntk.HDFDataStore(h5_path + 'outputDRED.h5', 'w')
co.disaggregate(mains, output)
output.close()

# Metrics==========================================================================
disag = ntk.DataSet(h5_path + 'outputDRED.h5')
disag_elec = disag.buildings[1].elec

f1 = ntk.metrics.f1_score(disag_elec, elec)
