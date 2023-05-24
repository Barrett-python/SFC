import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50
from sklearn.cluster import KMeans


class Net(nn.Module):

    def __init__(self, num_cls=21):
        super(Net, self).__init__()

        self.num_cls = num_cls

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)
        self.classifier = nn.Conv2d(2048, self.num_cls-1, 1, bias=False)

        self.f9 = torch.nn.Conv2d(3+2048+128, 2048, 1, bias=False)

        self.in_channels = 2048
        self.final_in_channels = 128
        self.lenk = 128
        self.inter_channels = 128
        self.Q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.K = nn.Linear(in_features = self.lenk, out_features = self.inter_channels)
        self.V = nn.Linear(in_features = self.lenk, out_features = self.inter_channels)
        self.aggre = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.concat_project = nn.Conv2d(in_channels=self.in_channels+self.inter_channels, out_channels=self.final_in_channels, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(in_channels=768, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.feat_dim = 128
        self.queue_len = 10
        self.momentum = 0.99
        for i in range(0, 20):
            self.register_buffer("queue" + str(i), torch.randn(self.queue_len, self.feat_dim))
            self.register_buffer("queue_ptr" + str(i), torch.zeros(1, dtype=torch.long))

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.f9, self.side1, self.side2, self.side3, self.side4, 
                                          self.Q, self.K, self.V, self.aggre, self.concat_project
                                          ])

    def PCM(self, cam, f, valid_mask):
        n,c,h,w = f.size()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        f = self.f9(f)
        f = f.view(n,-1,h*w)
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)
        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        cam_rv = torch.matmul(cam_rv.view(n,-1,h*w), aff).view(n,-1,h,w)
        return cam_rv
    
    def Bank(self, sem_feature):
        feat_memory = getattr(self, "queue0")
        for k in range(1, 20):
            feat_memory = torch.cat((feat_memory, getattr(self, "queue" + str(k))), 0)
        query_projector = self.Q(sem_feature.clone().detach()) 
        # b, c, h, w = query_projector.shape 
        # query = query_projector.permute(0, 2, 3, 1).view(b, w*h, c) # 16, 1024, 128
        # key = self.K(feat_memory).permute(1, 0)  
        # Affinity = F.relu(torch.matmul(query, key))
        # Affinity = Affinity/(torch.sum(Affinity,dim=1,keepdim=True)+1e-5)
        # value = self.V(feat_memory)
        # out_feature = torch.matmul(Affinity, value).permute(0, 2, 1).view(b, c, h, w)
        # query_ = self.aggre(out_feature) 
        # refine_feature = self.concat_project(torch.cat((sem_feature, query_), 1))
        
        # sem_feature = torch.cat([sem_feature, query_projector], dim=1)
        
        return sem_feature, query_projector

        
    def prototype(self, norm_cam, feature, valid_mask):
        n,c,h,w = norm_cam.shape
        norm_cam[:,0] = norm_cam[:,0]*0.2
        seeds = torch.zeros((n,h,w,c)).cuda()
        belonging = norm_cam.argmax(1)
        seeds = seeds.scatter_(-1, belonging.view(n,h,w,1), 1).permute(0,3,1,2).contiguous()
        seeds = seeds * valid_mask # 4, 21, 32, 32

        n,c,h,w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        # crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1) #.clone().detach()  # seed:[n,21,1,h,w], feature:[n,1,c,h,w], crop_feature:[n,21,c,h,w]
        # prototype = F.adaptive_avg_pool2d(crop_feature.view(-1,c,h,w), (1,1)).view(n, self.num_cls, c, 1, 1) # prototypes:[n,21,c,1,1]        

        feat_memory = getattr(self, "queue0").unsqueeze(0)
        for k in range(1, 20):
            feat_memory = torch.cat((feat_memory, getattr(self, "queue" + str(k)).unsqueeze(0)), 0)
        prototype = torch.mean(feat_memory,1).unsqueeze(0)
        
        feature = self.conv(feature)
        prototype = prototype.unsqueeze(-1).unsqueeze(-1)
        if n == 1:
            prototype = F.normalize(prototype, dim=-1)

        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2)) # feature:[n,1,c,h,w], prototypes:[n,21,c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)

        norm_cam = F.relu(IS_cam) 
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)*valid_mask 

        return norm_cam
        



    def forward(self, x, valid_mask,my_label= None, epoch=None, index=None, train= None):
        
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        
        sem_feature = x4
        cam = self.classifier(x4)
        score = F.adaptive_avg_pool2d(cam, 1)
        norm_cam = F.relu(cam) 
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)*valid_mask
        orignal_cam = norm_cam

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                            F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                            F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                            F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        sem_feature, query_projector = self.Bank(sem_feature)
        norm_cam = self.PCM(norm_cam, torch.cat([F.interpolate(x, side3.shape[2:],mode='bilinear',align_corners=True), sem_feature], dim=1), valid_mask.clone())
        
        IS_cam = self.prototype(norm_cam.clone(), hie_fea.clone(), valid_mask.clone())

        if x.size()[0]!=1:
            for j in range(0, x.size()[0]):
                self._dequeue_and_enqueue(query_projector[j].clone().detach(), IS_cam[j], my_label[j], score[j])

        return {"score": score, "cam1": norm_cam, "cam2": IS_cam, "orignal_cam": orignal_cam}
    
    def _dequeue_and_enqueue(self, x, map, label, probs):
        map = F.softmax(map, dim = 0)
        for ind, cla in enumerate(label):
            if cla == 1:
                if(probs[ind] > 0.99):
                    mask = map[ind] > (torch.mean(map[ind]))
                    x = x * mask.float() 
                    embedding = x.reshape(x.shape[0], -1).sum(1)/mask.float().sum()
                    queue_i = getattr(self, "queue" + str(ind))
                    queue_ptr_i = getattr(self, "queue_ptr" + str(ind))
                    ptr = int(queue_ptr_i)
                    queue_i[ptr:ptr + 1] = queue_i[ptr:ptr + 1] * self.momentum + embedding * (1 - self.momentum)
                    ptr = (ptr + 1) % self.queue_len  # move pointer
                    queue_ptr_i[0] = ptr

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class CAM(Net):

    def __init__(self, num_cls):
        super(CAM, self).__init__(num_cls=num_cls)
        self.num_cls = num_cls

    def forward(self, x, label):

        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())        
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'), 
                              F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                              F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)

        cam = self.classifier(x4)
        cam = (cam[0] + cam[1].flip(-1)).unsqueeze(0)
        hie_fea = (hie_fea[0] + hie_fea[1].flip(-1)).unsqueeze(0)
        norm_cam = F.relu(cam)
        norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)

        x4, _ = self.Bank(x4)
        
        x = (x[0] + x[1].flip(-1)).unsqueeze(0)
        x4 = (x4[0] + x4[1].flip(-1)).unsqueeze(0)
        x3 = (x3[0] + x3[1].flip(-1)).unsqueeze(0)
        
        norm_cam = self.PCM(norm_cam, torch.cat([F.interpolate(x,side3.shape[2:],mode='bilinear',align_corners=True), x4], dim=1), label.unsqueeze(0).clone())
        IS_cam = self.prototype(norm_cam.clone(), hie_fea.clone(), label.unsqueeze(0).clone())

        return norm_cam[0], IS_cam[0]