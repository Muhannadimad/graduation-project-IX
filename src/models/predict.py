import torch
# from src.models.final_model import Net
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch.nn import Parameter
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
data_path = "../../data/processed/" + "data_withtexts_predict.dataset"

data = torch.load(data_path)

print(data,"data")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if "GCN" == 'GCN':

            # cached = True is for transductive learning
            self.conv1 = GCNConv(data.x.shape[1], 16, cached=True)
            self.conv2 = GCNConv(16, 32, cached=True)
            self.conv3 = GCNConv(32, 64, cached=True)
            self.conv4 = GCNConv(64, 3, cached=True)
        elif "ChebConv1" == 'ChebConv':
            self.conv1 = ChebConv(data.x.shape[1], 16, K=3)
            self.conv2 = ChebConv(16, 32, K=3)
            self.conv3 = ChebConv(32, 64, K=3)
            self.conv4 = ChebConv(64, 3, K=3)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)

model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    # print(checkpoint['state_dict'], "mahdi")
    # for i in checkpoint['state_dict']:
        # i = i.unsqueeze(0)
    model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

if __name__ == '__main__':
    checkpoint_fpath = "../../saved_model/checkpoint.pt"


    model, optimizer, start_epoch  = load_ckp(checkpoint_fpath,model,optimizer)
    model.eval()
    print("model123  ",model)
    logits, accs = model(), []
    # mask = data('train_mask', 'val_mask', 'test_mask')
    # print(data.val_mask,"mmmm")
    pred = logits[data.val_mask].max(1)[1]
    print(pred, "pred")



