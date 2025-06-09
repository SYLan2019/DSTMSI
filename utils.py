import zipfile
import os
import scipy.sparse as sp
import scipy.io
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
import joblib
from math import radians, cos, sin, asin, sqrt
from scipy.optimize import linprog
import time


def load_metr_la_rdata():
    if(not os.path.isfile("data/metr/adj_mat.npy")or not os.path.isfile("data/metr/node_values.npy")):
        with zipfile.ZipFile("data/metr/METR-LA.zip",'r') as zip_ref:
            zip_ref.extractall("data/metr/")

    A=np.load("data/metr/adj_mat.npy")
    X=np.load("data/metr/node_values.npy") #tnf
    X=X.transpose((1,2,0))
    X=X.astype(np.float32)
    return A,X

def load_chengdu_data():
    assert os.path.isfile('data/chengdu/dataset.npy')
    assert os.path.isfile('data/chengdu/matrix.npy')
    A=np.load("data/chengdu/matrix.npy")
    X=np.load("data/chengdu/dataset.npy")   #tnf
    X=X.transpose((1,2,0))  #nft
    X=X.astype(np.float32)
    return A,X


def load_shenzhen_data():
    assert os.path.isfile('data/shenzhen/dataset.npy')
    assert os.path.isfile('data/shenzhen/matrix.npy')
    A=np.load("data/shenzhen/matrix.npy")
    X=np.load("data/shenzhen/dataset.npy")   #tnf
    X=X.transpose((1,2,0))  #nft
    X=X.astype(np.float32)
    return A,X

def load_crowd_bm_data():
    assert os.path.isfile('data/crowd_bm/dataset.npy')
    assert os.path.isfile('data/crowd_bm/matrix.npy')
    A=np.load("data/crowd_bm/matrix.npy")
    X=np.load("data/crowd_bm/dataset.npy")   #tnf
    X=X.transpose((1,2,0))  #nft
    X=X.astype(np.float32)
    return A,X

def load_pems08_data(): #17856,170,3
    assert os.path.isfile('data/pems08/pems08.npz')
    assert os.path.isfile('data/pems08/pems08.csv')
    transfer_set = np.load('data/pems08/pems08.npz')['data']
    distance_df = pd.read_csv('data/pems08/pems08.csv', dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.001
    dist_mx = np.zeros((170, 170), dtype=np.float32)
    dist_mx[:] = np.inf
    for row in distance_df.values:
            dist_mx[int(row[0]), int(row[1])] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    # adj_mx[adj_mx < normalized_k] = 0
    A_new = adj_mx
    return transfer_set,A_new

def load_pems04_data(): #16992,307,3
    assert os.path.isfile('data/pems04/pems04.npz')
    assert os.path.isfile('data/pems04/pems04.csv')
    transfer_set = np.load('data/pems04/pems04.npz')['data']
    distance_df = pd.read_csv('data/pems04/pems04.csv', dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1
    dist_mx = np.zeros((307, 307), dtype=np.float32)
    dist_mx[:] = np.inf
    for row in distance_df.values:
            dist_mx[int(row[0]), int(row[1])] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    # adj_mx[adj_mx < normalized_k] = 0
    A_new = adj_mx
    return transfer_set,A_new

def load_pemsbay_data():
    assert os.path.isfile('data/pemsbay/pems-bay.h5')
    assert os.path.isfile('data/pemsbay/distances_bay_2017.csv')
    df = pd.read_hdf('data/pemsbay/pems-bay.h5')
    transfer_set = df.values
    distance_df = pd.read_csv('data/pemsbay/distances_bay_2017.csv', dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1
    dist_mx = np.zeros((325, 325), dtype=np.float32)
    dist_mx[:] = np.inf
    sensor_ids = df.columns.values.tolist()
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # adj_mx[adj_mx < normalized_k] = 0
    A = adj_mx
    X = transfer_set.transpose()
    return X,A # n,timesteps

def load_beijing_data():
    assert os.path.isfile('data/beijing/Beijing.h5')
    assert os.path.isfile('data/beijing/distance.npy')
    df = pd.read_hdf('data/beijing/Beijing.h5')    #8760，36
    X = df.values.astype(np.float32).T  #36,8760
    dist_mx = np.load('data/beijing/distance.npy')
    normalized_k = 0.1
    std = np.std(dist_mx)
    A = np.exp(-dist_mx ** 2 / std ** 2)
    A[A < normalized_k] = 0
    return X,A # n,timesteps

def load_bjair_data():
    stations_data = pd.read_excel('data/bjair/stations.xlsx').to_numpy()
    stations_data = stations_data[np.lexsort(stations_data[:, ::-1].T)]
    stations_list = stations_data[:, 0]
    G = nx.Graph()
    G.add_nodes_from(stations_list)
    for i in range(0, len(stations_list)):
        for j in range(i + 1, len(stations_list)):
            G.add_edge(stations_list[i], stations_list[j],
                       weight=haversine(stations_data[i][1], stations_data[i][2], stations_data[j][1],
                                             stations_data[j][2]))
    A = nx.adjacency_matrix(G).todense()

    # Gaussian kernel
    original_A = A
    A = np.exp(- 0.5 * (original_A / np.std(original_A, axis=1, keepdims=True)) ** 2)

    # here we use interpolate
    aq_data = []
    for path in  ['data/bjair/beijing_201802_201803_aq.csv', 'data/bjair/beijing_17_18_aq.csv']:
        aq_data.append(pd.read_csv(path))
    # interpolate missing values
    beijing_aq = pd.concat(aq_data).sort_values(
        by=['stationId', 'utc_time']).drop(columns=['stationId', 'utc_time'])
    missing_index = beijing_aq.isna().to_numpy().reshape((len(stations_list), -1, 6))
    # beijing_aq = beijing_aq.interpolate('linear', axis=0).ffill().bfill().fillna(0)
    beijing_aq = beijing_aq.fillna(0)

    beijing_aq = beijing_aq.to_numpy().reshape((len(stations_list), -1, 6))
    beijing_time = pd.concat(aq_data).sort_values(by=['stationId', 'utc_time'])['utc_time'].unique().tolist()

    # PM 2.5
    beijing_aq = beijing_aq[..., 0]
    missing_index = missing_index[..., 0]

    return beijing_aq,A,missing_index   #[35,10298][35,35]

def load_AQI_data():
    df = pd.read_hdf('data/AQI/full437.h5', key='/pm25')    #8760,437
    stations= pd.read_hdf('data/AQI/full437.h5', key='/stations')   #437,3
    transfer_set = df.values
    stations=stations.values
    A=np.zeros([stations.shape[0],stations.shape[0]])
    for i in range(0,stations.shape[0]):
        for j in range(i+1,stations.shape[0]):
            A[i,j]=haversine(stations[i][1],stations[i][0],stations[j][1],stations[j][0])
    A=A+A.T
    original_A = A
    A = np.exp(- 0.5 * (original_A / np.std(original_A, axis=1, keepdims=True)) ** 2)
    k=0.1
    A[A<k]=0
    return transfer_set,A


def generate_ushcn_data():
    pos = []
    Utensor = np.zeros((1218, 120, 12, 2))
    Omissing = np.ones((1218, 120, 12, 2))
    with open("data/ushcn/Ulocation", "r") as f:
        loc = 0
        for line in f.readlines():
            poname = line[0:11]
            pos.append(line[13:30])
            with open("data/ushcn/ushcn.v2.5.5.20191231/" + poname + ".FLs.52j.prcp", "r") as fp:
                temp = 0
                for linep in fp.readlines():
                    if int(linep[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linep[17 + 9 * i:22 + 9 * i]
                            p_temp = int(str_temp)
                            if p_temp == -9999:
                                Omissing[loc, temp, i, 0] = 0
                            else:
                                Utensor[loc, temp, i, 0] = p_temp
                        temp = temp + 1
            with open("data/ushcn/ushcn.v2.5.5.20191231/" + poname + ".FLs.52j.tavg", "r") as ft:
                temp = 0
                for linet in ft.readlines():
                    if int(linet[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linet[17 + 9 * i:22 + 9 * i]
                            t_temp = int(str_temp)
                            if t_temp == -9999:
                                Omissing[loc, temp, i, 1] = 0
                            else:
                                Utensor[loc, temp, i, 1] = t_temp
                        temp = temp + 1
            loc = loc + 1

    latlon = np.loadtxt("data/ushcn/latlon.csv", delimiter=",")
    sim = np.zeros((1218, 1218))

    for i in range(1218):
        for j in range(1218):
            sim[i, j] = haversine(latlon[i, 1], latlon[i, 0], latlon[j, 1], latlon[j, 0])  # RBF
    sim = np.exp(-sim / 10000 / 10)

    joblib.dump(Utensor, 'data/ushcn/Utensor.joblib')
    joblib.dump(Omissing, 'data/ushcn/Omissing.joblib')
    joblib.dump(sim, 'data/ushcn/sim.joblib')
def load_udata():
    if (not os.path.isfile("data/ushcn/Utensor.joblib")
            or not os.path.isfile("data/ushcn/sim.joblib")):
        with zipfile.ZipFile("data/ushcn/ushcn.v2.5.5.20191231.zip", 'r') as zip_ref:
            zip_ref.extractall("data/ushcn/ushcn.v2.5.5.20191231/")
        generate_ushcn_data()
    X = joblib.load('data/ushcn/Utensor.joblib')
    A = joblib.load('data/ushcn/sim.joblib')
    Omissing = joblib.load('data/ushcn/Omissing.joblib')
    X = X.astype(np.float32)
    return A,X,Omissing

def load_sedata():
    assert os.path.isfile('data/sedata/A.mat')
    assert os.path.isfile('data/sedata/mat.csv')
    A_mat = scipy.io.loadmat('data/sedata/A.mat')
    A = A_mat['A']
    X = pd.read_csv('data/sedata/mat.csv',index_col=0)
    X = X.to_numpy()
    return A,X

def calculate_random_walk_matrix(adj_mx):
    adj_mx=sp.coo_matrix(adj_mx)
    d=np.array(adj_mx.sum(1))
    d_inv=np.power(d,-1).flatten()
    d_inv[np.isinf(d_inv)]=0.
    d_mat_inv=sp.diags(d_inv)
    random_walk_mx=d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def haversine( lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r
