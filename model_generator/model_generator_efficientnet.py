import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

target_models = ["b{}".format(x) for x in range(0, 7)]

for target_model in target_models:
    model = EfficientNet.from_pretrained('efficientnet-{}'.format(target_model)).cuda()
    model.set_swish(memory_efficient=False)
    k = torch.rand(1,3,224,224).cuda()
    modellist = list(model.children())
    MBConvs = list(modellist[2].children())

    module_list= []
    module_list.append(modellist[0])
    module_list.append(modellist[1])
    for m in MBConvs:
        module_list.append(m)
    module_list.append(modellist[3])
    module_list.append(modellist[4])
    module_list.append(modellist[5])
    # module_list.append(modellist[5])

    module_list.append(torch.nn.Flatten())
    module_list.append(modellist[6])
    module_list.append(modellist[7])
    module_list.append(modellist[8])

    class ClientModel(torch.nn.Module):
        # define model elements
        def __init__(self):
            super(ClientModel, self).__init__()
            self.layers = torch.nn.ModuleList(module_list)
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
            self.layers = torch.nn.ModuleList(module_list)
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

    tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open('../material/retriever.jpg')).unsqueeze(0).cuda()

    # Load ImageNet class names
    labels_map = json.load(open('../material/labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify
    clientModelScript = torch.jit.script(clientModel).cuda()
    serverModelScript = torch.jit.script(serverModel).cuda()

    torch.jit.save(clientModelScript, 'client_{}.pt'.format(target_model))
    torch.jit.save(serverModelScript, 'server_{}.pt'.format(target_model))


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
    print("a",a)
    import matplotlib.pyplot as plt
    plt.bar(range(len(a)),a)
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

    print(target_model, len(shapes))

    p = [x for x in range(0, len(shapes))]
    print(p)
    print(shapes)
    muls = []
    fp = open("{}_shape".format(target_model), "w")
    for i,c in enumerate(p):
        if len(shapes[c]) == 4:
            fp.write("{} {} {} {}\n".format(shapes[c][0],shapes[c][1],shapes[c][2],shapes[c][3]))
            print("std::vector<int64_t> shape{}{{{},{},{},{}}};".format(i+1, shapes[c][0],shapes[c][1],shapes[c][2],shapes[c][3]))
            muls.append(shapes[c][0]*shapes[c][1]*shapes[c][2]*shapes[c][3])
        else:
            fp.write("{} {}\n".format(shapes[c][0],shapes[c][1]))

            print("std::vector<int64_t> shape{}{{{},{}}};".format(i+1, shapes[c][0],shapes[c][1]))
            muls.append(shapes[c][0] * shapes[c][1])
    fp.close()
    for i,c in enumerate(p):
        print("shapes.push_back(shape{});".format(i+1))
    for i,c in enumerate(p):
        print("index_set.push_back((float){});".format(c))

    print(muls)
    print(len(muls))
