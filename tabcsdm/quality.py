from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
import pandas as pd
import numpy as np
import pdb
from sdv.datasets.demo import download_demo, get_available_demos
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer

name = 'par10' 
method = 'syn'

train_path = f'dataset/{name}/train.csv'
train = pd.read_csv(train_path)
#syn.replace(-1,np.nan)
syn_path = f'dataset/{name}/{method}.csv'
syn = pd.read_csv(syn_path)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train)

quality_report = evaluate_quality(
    train,
    syn,
    metadata)


pdb.set_trace()
from sdv.evaluation.single_table import get_column_pair_plot

fig = get_column_pair_plot(
    real_data=train,
    synthetic_data=syn,
    metadata=metadata,
    column_names=["Credit Score","Annual Income"]
    )
    
fig.show()
pdb.set_trace()
quality_report.get_details(property_name='Column Shapes')
quality_report.get_details(property_name='Column Pair Trends')