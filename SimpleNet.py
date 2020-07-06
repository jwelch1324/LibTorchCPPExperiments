import torch
import torch.nn.functional as F
import torchvision as tv

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784,64)
        self.fc2 = torch.nn.Linear(64,32)
        self.fc3 = torch.nn.Linear(32,10)

    def forward(self,x):
        x = F.relu(self.fc1(x.reshape(x.shape[0],784)))
        x = F.dropout(x,0.5,self.training)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x),1)
        return x

if __name__ == "__main__":
    net = Net()
    net.cuda()

    data_loader = torch.utils.data.dataloader.DataLoader(
        tv.datasets.MNIST("/media/ramdisk/data",True,transform=tv.transforms.ToTensor(),download=True),256
    )

    optim = torch.optim.SGD(net.parameters(),0.01)

    for i in range(10):
        batch_idx = 0
        
        for batch in data_loader:
            optim.zero_grad()
            s,t = batch
            pred = net(s.cuda())
            loss = F.nll_loss(pred,t.cuda())
            loss.backward()

            optim.step()

            batch_idx += 1

            if (batch_idx % 100 == 0):
                print(f"Epoch: {i} | Batch {batch_idx} | Loss {loss.item()}")
                torch.save(net,'net.pkl')
