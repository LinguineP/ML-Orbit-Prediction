import pandas as pd

"""
A utility script for adding a missing last value after generating nn inputs

"""

def getPhi(ts,gpDF):
    phi=None
    
    row = gpDF[gpDF.iloc[:, 0] == ts]

    
    perturbation_phi_value = row["pertrubation_phi"].iloc[0] if not row.empty else None
    
    return perturbation_phi_value


inputs_path="/home/pavle/op-ml/nnInputs/inputsFull.csv"
gp_path="/home/pavle/op-ml/data/exogenous/gravity_pertrubations_out.csv"

inputDF=pd.read_csv(inputs_path)
gpDF=pd.read_csv(gp_path)



for index, row in inputDF.iterrows():
        phi = getPhi(row['timeStamp'],gpDF)
        inputDF.at[index, 'perturbation_phi'] = phi
        


inputDF.drop('pertrubation_phi',axis=1,inplace=True)
inputDF.to_csv('inputcomplete.csv', index=False)   