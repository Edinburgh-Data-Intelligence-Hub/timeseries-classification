from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SEED

from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def fluo_scaler(data, feature_range = (0,1)): 
    #takes a list of series as input, log them and scale them minmax to feature_range
    global_min = np.min([np.min(np.log(serie)) for serie in data])
    global_max = np.max([np.max(np.log(serie)) for serie in data])
    data_scaled = []
    for serie in data:
        serie = np.log(serie)
        serie = (serie - global_min) / (global_max - global_min)
        serie = serie * (feature_range[1] - feature_range[0]) + feature_range[0]
        data_scaled.append(serie)
    return(data_scaled)

def all_projection_truncated(
    liste_x,
    variable: str = 'size',
    scaling: bool = False
):
    if variable == 'size':
        all_proj_trunc = np.array([x[0][24:96] for x in liste_x]) #24:96 = first 6 hours of antibiotic exposure (hour 2 to 8)
        if scaling:
            all_proj_trunc = TimeSeriesScalerMeanVariance().fit_transform(all_proj_trunc)
    elif variable == 'sos':
        all_proj_trunc = np.array([x[1][24:96] for x in liste_x])
        if scaling:
            all_proj_trunc = np.array(fluo_scaler([s for s in all_proj_trunc]))
    else:
        raise ValueError("variable must be 'size' or 'sos'")
    
    return all_proj_trunc
    
def process_track_data(
    df_tracks, 
    dataset_name_list,
    dataset_type = "classifier_training" #or "autoencoder_training"
):
    liste_x=[]
    liste_y=[]
    
    for exp_name in dataset_name_list:
        medium,treatment,replicate = exp_name.split('_')
        
        frame = df_tracks.loc(axis=0)[:,:,medium,treatment,replicate]
        size_array = np.array([list(frame['FeretMax'].T[k]) for k in frame['FeretMax'].T.keys()])
        cyclefate_array = np.array([list(frame['cellcycle_fate'].T[k]) for k in frame['cellcycle_fate'].T.keys()])
        sos_array = np.array([list(frame['MeanIntensity_gfp'].T[k]) for k in frame['MeanIntensity_gfp'].T.keys()])
        
        for i in range(size_array.shape[0]):
            size_serie =  size_array[i]
            ##fill nans missing timepoints with the mean of the previous and next timepoints
            for j in range(1,len(size_serie)-1):
                if np.isnan(size_serie[j]):
                    size_serie[j] = np.mean([size_serie[j-1],size_serie[j+1]])
            
            cyclefate_serie = cyclefate_array[i]
            ##fill nans missing timepoints with the next fate timepoint
            for j in range(len(cyclefate_serie)-1):
                if cyclefate_serie[j] == 'nan':
                    cyclefate_serie[j] = cyclefate_serie[j+1]
            
            sos_serie = sos_array[i]
            for j in range(1,len(sos_serie)-1):
                if np.isnan(sos_serie[j]):
                    sos_serie[j] = np.mean([sos_serie[j-1],sos_serie[j+1]])
            
            t_death = len(cyclefate_serie)
            if np.where(cyclefate_serie != 'alive')[0].shape[0] > 0:
                t_death = np.where(cyclefate_serie != 'alive')[0][0]
            
            first_nan_idx = len(size_serie)
            if np.where(np.isnan(size_serie))[0].shape[0] > 0:
                first_nan_idx = np.where(np.isnan(size_serie))[0][0]    
            
            if dataset_type == "autoencoder_training":
                usable_data = size_serie[:min(t_death,288)] # use data until cell death or 288 frames (24h), whichever comes first
                if len(usable_data) > 72: #only keep tracks with at least 6 hours of data
                    liste_x.append(usable_data)
                    liste_y.append((medium,treatment,replicate,t_death))
            
            elif dataset_type == "classifier_training":
                usable_data = (size_serie[:min(first_nan_idx,168)], sos_serie[:min(first_nan_idx,168)]) #use data until first nan or 168 frames (14h, whichever comes first)
                
                if len(usable_data[0]) > 72+24: #only keep tracks with at least 8 hours of data (24 frames of control + 72 frames of antibiotic exposure)
                    liste_x.append(usable_data)
                    liste_y.append((medium,treatment,replicate,t_death))
    
    if len(liste_x) == 0:
        raise ValueError("No usable data found. Please check the dataset and criteria.")
    return liste_x, liste_y


def main(
    input_path: Path = RAW_DATA_DIR / "growth_antibiotic_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR 
): 
    scaling = False
    variable = 'size' #or 'size', 'sos',  size to be used as sos is experimental
    task = 'ciptet' #or cip, tet or ciptet
    used_media = ['glu']
    condition_label = {'control':0,'cip':1,'tet':1,'ciptet':1}
    dataset_type = "classifier_training"

    print(task, dataset_type)

    df = pd.read_pickle(input_path)

    df_tracks= df.pivot(
        values=[
            'GrowthRateSize',
            'GrowthRateLength',
            'GrowthRateFeretMax',
            'GrowthRateFeretMaxSliding', 
            'InterdivisionTimes',
            'DivisionRate',
            'DivisionRate_filtered',
            'TrackLength',
            'TrackLength_filtered',
            'Size',
            'SizeAtBirthSize',
            'FeretMax',
            'SizeAtBirthFeretMax',
            'MaxLength',
            'SpineLength',
            'SizeAtBirthLength',
            'SpineWidth',
            'MeanIntensity_mch',
            'MeanIntensity_gfp',
            'Maxgfp',
            'BacteriaLineage',
            'NextDivisionFrame',
            'PreviousDivisionFrame',
            'TrackHeadIndices',
            'Prev',
            'Next',
            'Idx',
            'Frame',
            'Indices',
            'PositionIdx',
            'cellcycle_fate',
        ], 
    index=[
        'Position',
        'ParentTrackHeadIndices',
        'Medium',
        'Treatment',
        'RepeatID',
        'RepeatDate',
        'fate',
        'DeathSubtype',
    ],
    columns='Time')


    #### Creating dataset and splitting it into train and test sets then data augmentation
    dataset_name_list = [
        'gly_control_1', 
        'gly_control_2', 
        'gly_control_3', 
        'gly_cip_1', 
        'gly_cip_2', 
        'gly_tet_1', 
        'gly_tet_2', 
        'gly_tet_3', 
        'gly_ciptet_1', 
        'gly_ciptet_2', 
        'glu_control_1', 
        'glu_control_2', 
        'glu_cip_1', 
        'glu_cip_2', 
        'glu_tet_1', 
        'glu_tet_2', 
        'glu_ciptet_1', 
        'glu_ciptet_2', 
        'gluaa_control_1', 
        'gluaa_control_2', 
        'gluaa_cip_1', 
        'gluaa_cip_2', 
        'gluaa_cip_3', 
        'gluaa_tet_1', 
        'gluaa_tet_2', 
        'gluaa_ciptet_1', 
        'gluaa_ciptet_2'
    ]

    liste_x, liste_y = process_track_data( 
        df_tracks=df_tracks, 
        dataset_name_list=dataset_name_list, 
        dataset_type=dataset_type
    ) 

    all_proj_trunc = all_projection_truncated(
        liste_x=liste_x,
        variable=variable,
        scaling=scaling
    )
    
    if task in condition_label.keys():
        X = [x for x,y in zip(all_proj_trunc,liste_y) if (y[1]==task or y[1]=='control') and y[0] in used_media]   
        y = [condition_label[y[1]] for y in liste_y if (y[1]==task or y[1]=='control') and y[0] in used_media]
    else:
        raise ValueError("task must be 'cip', 'tet' or 'ciptet'")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=SEED
    )
    
    #### SAVE PROCESSED DATA 
    tag = (
        f"{task}_scaled"
        if scaling
        else f"{task}"
    )

    # Full data
    with open(PROCESSED_DATA_DIR/f'X_{tag}_full.pkl', 'wb') as f:
        pickle.dump(X, f)
    f.close()

    with open(PROCESSED_DATA_DIR/f'y_{tag}_full.pkl', 'wb') as f:
        pickle.dump(y, f)
    f.close()

    # Train data
    with open(PROCESSED_DATA_DIR/f'X_{tag}_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    f.close()

    with open(PROCESSED_DATA_DIR/f'y_{tag}_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    f.close()

    # Test data
    with open(PROCESSED_DATA_DIR/f'X_{tag}_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    f.close()

    with open(PROCESSED_DATA_DIR/f'y_{tag}_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    f.close()

    ###### autoencoder training dataset creation
    dataset_type = "autoencoder_training"
    scaling = True
    dataset_name_list_controls = [
        'gly_control_1',
        'gly_control_2',
        'gly_control_3',
        'glu_control_1',
        'glu_control_2',
        'gluaa_control_1',
        'gluaa_control_2'
    ]
    liste_x_autoencoder, liste_y_autoencoder = process_track_data(
        df_tracks=df_tracks, 
        dataset_name_list=dataset_name_list_controls, 
        dataset_type=dataset_type
    )
    
    X_train_preaug, X_test_preaug, y_train_preaug, y_test_preaug = train_test_split(
        liste_x_autoencoder, 
        liste_y_autoencoder, 
        test_size=0.1, 
        random_state=SEED, 
        shuffle=True
    )

    #### data augmentation
    ##### in X_train we change each sequence to series of size 72 with a roling window of 6
    liste_x_train = []
    liste_y_train = []
    for i in range(len(X_train_preaug)):
        usable_data = X_train_preaug[i]
        for j in range(0, len(usable_data) - 72 + 1, 6):
            liste_x_train.append(usable_data[j:j+72])
            liste_y_train.append(y_train_preaug[i])
    X_train = np.array(liste_x_train)
    if scaling:
        X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
    y_train = np.array(liste_y_train)

    # In X_test we cut each sequence into unique 72-long sequences
    liste_x_test = []
    liste_y_test = []
    for i in range(len(X_test_preaug)):
        usable_data = X_test_preaug[i]
        for j in range(0, len(usable_data) - 72 + 1, 72):
            liste_x_test.append(usable_data[j:j + 72])
            liste_y_test.append(y_test_preaug[i])
    X_test = np.array(liste_x_test)
    if scaling:
        X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)
    y_test = np.array(liste_y_test)

    # Save autoencoder training data
    with open(PROCESSED_DATA_DIR/'X_autoencoder_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    f.close()

    with open(PROCESSED_DATA_DIR/'y_autoencoder_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    f.close()

    with open(PROCESSED_DATA_DIR/'X_autoencoder_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    f.close()

    with open(PROCESSED_DATA_DIR/'y_autoencoder_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    f.close()

if __name__ == "__main__":
    main()
