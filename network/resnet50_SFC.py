import torch
import torch.nn as nn
import torch.nn.functional as F
from tool import torchutils
from network import resnet50



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

        self.f9 = torch.nn.Conv2d(3+2048, 2048, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier, self.f9, self.side1, self.side2, self.side3, self.side4])

    def PCM(self, cam, f):
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
    
    def prototype(self, norm_cam, feature, valid_mask):
        n,c,h,w = norm_cam.shape
        norm_cam[:,0] = norm_cam[:,0]*0.3
        seeds = torch.zeros((n,h,w,c)).cuda()
        belonging = norm_cam.argmax(1)
        seeds = seeds.scatter_(-1, belonging.view(n,h,w,1), 1).permute(0,3,1,2).contiguous()
        seeds = seeds * valid_mask # 4, 21, 32, 32

        n,c,h,w = feature.shape # hie 
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1) #.clone().detach()  # seed:[n,21,1,h,w], feature:[n,1,4c,h,w], crop_feature:[n,21,4c,h,w]
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1,c,h,w), (1,1)).view(n, self.num_cls, c, 1, 1) # prototypes:[n,21,c,1,1]        

        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2)) # feature:[n,1,4c,h,w], prototypes:[n,21,4c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam
        
    def forward(self, x, valid_mask, my_label= None, epoch=None, index=None, train= None):
        
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())
        
        # print(self.side1.weight)
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

        norm_cam = self.PCM(norm_cam, torch.cat([F.interpolate(x, side3.shape[2:],mode='bilinear',align_corners=True), sem_feature], dim=1))
        IS_cam = self.prototype(norm_cam.clone(), hie_fea.clone(), valid_mask.clone())

        return {"score": score, "cam1": norm_cam, "cam2": IS_cam, "orignal_cam": orignal_cam}
    
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
        orignal_cam = norm_cam
        x = (x[0] + x[1].flip(-1)).unsqueeze(0)
        x4 = (x4[0] + x4[1].flip(-1)).unsqueeze(0)
        x3 = (x3[0] + x3[1].flip(-1)).unsqueeze(0)
        
        norm_cam = self.PCM(norm_cam, torch.cat([F.interpolate(x,side3.shape[2:],mode='bilinear',align_corners=True), x4], dim=1))
        IS_cam = self.prototype(norm_cam.clone(), hie_fea.clone(), label.unsqueeze(0).clone())
        return norm_cam[0], IS_cam[0], orignal_cam[0]