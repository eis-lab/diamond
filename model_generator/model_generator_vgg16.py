import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet
import io
# model definition
from torchvision import transforms

import torchvision

model = torchvision.models.vgg16(pretrained=True).cuda()
k = torch.rand(1,3,224,224).cuda()
modellist = list(model.children())
print(modellist)
before = list(modellist[0].children())
avgpool = modellist[1]
after = list(modellist[2].children())


class ClientModel(torch.nn.Module):
    # define model elements
    def __init__(self):
        super(ClientModel, self).__init__()
        self.layers = torch.nn.ModuleList([before[0], before[1],
                                           before[2], before[3],
                                           before[4], before[5],
                                           before[6], before[7],
                                           before[8], before[9],
                                           before[10], before[11],
                                           before[12], before[13],
                                           before[14], before[15],
                                           before[16], before[17],
                                           before[18], before[19],
                                           before[20], before[21],
                                           before[22], before[23],
                                           before[24], before[25],
                                           before[26], before[27],
                                           before[28], before[29],
                                           before[30]
                                          , avgpool,torch.nn.Flatten(),
                                           after[0], after[1],
                                           after[2], after[3],
                                           after[4], after[5],
                                           after[6]])

    # forward propagate input
    def forward(self, X,k):
        # X = self.layers[0](X)
        for i, a in enumerate(self.layers):
            # if i > k:
            #     X = a(X)
            if k == i:
                return X
            X = a(X)

        return X


class ServerModel(torch.nn.Module):
    # define model elements
    def __init__(self):
        super(ServerModel, self).__init__()
        self.layers = torch.nn.ModuleList([before[0], before[1],
                                           before[2], before[3],
                                           before[4], before[5],
                                           before[6], before[7],
                                           before[8], before[9],
                                           before[10], before[11],
                                           before[12], before[13],
                                           before[14], before[15],
                                           before[16], before[17],
                                           before[18], before[19],
                                           before[20], before[21],
                                           before[22], before[23],
                                           before[24], before[25],
                                           before[26], before[27],
                                           before[28], before[29],
                                           before[30]
                                              , avgpool,torch.nn.Flatten(),
                                           after[0], after[1],
                                           after[2], after[3],
                                           after[4], after[5],
                                           after[6]])
    # forward propagate input
    def forward(self, X,k):
        # X = self.layers[0](X)
        if k == 0:
            for i, a in enumerate(self.layers):
                X = a(X)
        else:
            for i, a in enumerate(self.layers):
                #X = a(X)
                if i >= k:
                    X = a(X)
            # if k == i:
            #     return X

        return X

clientModel = ClientModel().cuda()
clientModel.eval()
serverModel = ServerModel().cuda()
serverModel.eval()

# tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
#                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = tfms(Image.open('../material/retriever.jpg')).unsqueeze(0).cuda()

# Load ImageNet class names
labels_map = json.load(open('../material/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
clientModelScript = torch.jit.script(clientModel).cuda()
serverModelScript = torch.jit.script(serverModel).cuda()

torch.jit.save(clientModelScript, 'client_vgg16.pt')
torch.jit.save(serverModelScript, 'server_vgg16.pt')


with torch.no_grad():
    res2 = clientModelScript.forward(img, torch.tensor(7.0))
    print(np.shape(res2))
    outputs = serverModelScript.forward(res2, torch.tensor(7.0))

# Print predictions
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))

shapes = []

for k in range(len(serverModel.layers)):
    with torch.no_grad():
        res2 = clientModelScript.forward(img, torch.tensor(k))
        print(np.shape(res2), k)
        shapes.append(np.shape(res2))
        outputs = serverModelScript.forward(res2, torch.tensor(k))
    print('-----')
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
       prob = torch.softmax(outputs, dim=1)[0, idx].item()
       print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))

P = shapes

def vecmul(l):
    ret = 1
    for i in l:
        ret *= i
    return ret
#
# d1 = vecmul(shapes[0])
# current_size = d1
# can  = [x for x in range(1,len(shapes)-1)]
# for i, l in enumerate(shapes[1:],1):
#     dl = vecmul(l)
#     if d1 < dl:
#         print("bigger than input", l)
#         print(i)
#         can.remove(i)
#     else:
#         if dl == current_size:
#             print(i)
#             can.remove(i)
#         else:
#             current_size = dl
# print(len(shapes))
# can.insert(0,0)
# candidated = np.subtract(can,0)

a = [vecmul(x) for x in P]
print(a)
import matplotlib.pyplot as plt
plt.bar(range(len(a)),a)
plt.title("vgg16")
plt.show()
p = []
p.append(0)
app = a[0]
for i, d in enumerate(a[1:],1):
    if d>a[0]:
        pass
    else:
        if app != d :
            p.append(i)
            print("insert ",d,app, i)
            app = d
        else:
            pass
print(p)
p = [x for x in range(0, len(shapes))]

for i,c in enumerate(p):
    if len(shapes[c]) == 4:
        print("std::vector<int64_t> shape{}{{{},{},{},{}}};".format(i+1, shapes[c][0],shapes[c][1],shapes[c][2],shapes[c][3]))
    else:
        print("std::vector<int64_t> shape{}{{{},{}}};".format(i+1, shapes[c][0],shapes[c][1]))

for i,c in enumerate(p):
    print("shapes.push_back(shape{});".format(i+1))
for i,c in enumerate(p):
    print("index_set.push_back((float){});".format(c))
