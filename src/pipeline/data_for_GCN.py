import matplotlib.pyplot as plt

from src.pipeline.graph import Grapher
import torch
import scipy.sparse
import torch_geometric.data
import networkx as nx
import numpy as np
import os 
import random

def from_networkx(G):
    """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    # G = nx.from_dict_of_lists(graph_dict)
    # layout = nx.kamada_kawai_layout(G)
    # nx.draw(G, layout, with_labels=True)
    # # # plt.savefig(plot_path, format="PNG")
    # plt.title('Graph bef')
    # plt.show()
    # print(nx.info(G, n=None)
    #       ,"mahdi")

    G = nx.convert_node_labels_to_integers(G)

    # print(nx.info(G, n=None)
    #       , "mahdi2")

    # G = nx.from_dict_of_lists(graph_dict)
    # layout = nx.kamada_kawai_layout(G)
    # nx.draw(G, layout, with_labels=True)
    # # # plt.savefig(plot_path, format="PNG")
    # plt.title('Graph ')
    # plt.show()
    # print(list(G.edges), "xx")
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    # print(edge_index)
    data = {}
    # print((G.nodes(data=True)),"mahdi")
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):

        for key, value in feat_dict.items():
            # print("mahdiimad")
            data[key] = [value] if i == 0 else data[key] + [value]
        # print(i, feat_dict, "iii")
    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass 

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    # print(data, "mm")
    return data

def get_data():
    """
    returns one big graph with unconnected graphs with the following:
    - x (Tensor, optional) ??? Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
    - edge_index (LongTensor, optional) ??? Graph connectivity in COO format with shape [2, num_edges]. (default: None)
    - edge_attr (Tensor, optional) ??? Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
    - y (Tensor, optional) ??? Graph or node targets with arbitrary shape. (default: None)
    - validation mask, training mask and testing mask 
    """
    path = "../../data/raw2/box/"
    l=os.listdir(path)
    files=[x.split('.')[0] for x in l]
    # print(files, "bef")
    files.sort()

    all_files = files[1:]
    # print(all_files)
    list_of_graphs = []

    r"""to create train,test,val data"""
    files = all_files.copy()
    random.shuffle(files)

    r"""Resulting in 500 receipts for training, 63 receipts for validation, and 63 for testing."""
    training,testing,validating = files[1:15],files[15:],files[0]


    for file in all_files:
        print("xxxxxxxxxxxxxxxxxxx" , file)
        connect = Grapher(file)

        G,_,_ = connect.graph_formation()
        # draw
        # G = nx.from_dict_of_lists(graph_dict)
        # layout = nx.kamada_kawai_layout(G)
        # nx.draw(G, layout, with_labels=True)
        # # plt.savefig(plot_path, format="PNG")
        # plt.title('Graph ')
        # plt.show()
        # print(G, "G")
        df = connect.relative_distance()
        # print(df)
        individual_data = from_networkx(G)

        feature_cols = ['rd_b', 'rd_r', 'rd_t', 'rd_l','line_number',\
                'n_upper', 'n_alpha', 'n_spaces', 'n_numeric','n_special']

        features = torch.tensor(df[feature_cols].values.astype(np.float32))
        # print(features.shape, "sh")
        for col in df.columns:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass

        df['labels'] = df['labels'].fillna('undefined')
        df.loc[df['labels'] == 'name', 'num_labels'] = 1
        df.loc[df['labels'] == 'id', 'num_labels'] = 2
        # df.loc[df['labels'] == 'invoice', 'num_labels'] = 3
        # df.loc[df['labels'] == 'date', 'num_labels'] = 4
        # df.loc[df['labels'] == 'total', 'num_labels'] = 5
        df.loc[df['labels'] == 'undefined', 'num_labels'] = 3
        # print(df['num_labels'].isnull().values.any(), df )
        assert df['num_labels'].isnull().values.any() == False, f'labeling error! Invalid label(s) present in {file}.csv'
        labels = torch.tensor(df['num_labels'].values.astype(np.int64))
        # print(labels,"mahd")
        text = df['Object'].values

        individual_data.x = features
        individual_data.y = labels
        individual_data.text = text

        # print(individual_data.x,"mahd")
        r"""Create masks"""
        if file in training:
            individual_data.train_mask = torch.tensor([True] * df.shape[0])
            individual_data.val_mask = torch.tensor([False] * df.shape[0])
            individual_data.test_mask = torch.tensor([False] * df.shape[0])


        elif file in validating:
            individual_data.train_mask = torch.tensor([False] * df.shape[0])
            individual_data.val_mask = torch.tensor([True] * df.shape[0])
            individual_data.test_mask = torch.tensor([False] * df.shape[0])
        else:
            individual_data.train_mask = torch.tensor([False] * df.shape[0])
            individual_data.val_mask = torch.tensor([False] * df.shape[0])
            individual_data.test_mask = torch.tensor([True] * df.shape[0])
        # print(individual_data.train_mask, "mahdi")

        print(f'{file} ---> Success')
        g = torch_geometric.utils.to_networkx(individual_data, to_undirected=True)
        nx.draw(g)
        list_of_graphs.append(individual_data)
    
    data = torch_geometric.data.Batch.from_data_list(list_of_graphs)
    data.edge_attr = None 

    save_path = "../../data/processed/"  
    torch.save(data, save_path +'data_withtexts2.dataset')
    print('Data is saved!')

if __name__ == "__main__":
    get_data()
