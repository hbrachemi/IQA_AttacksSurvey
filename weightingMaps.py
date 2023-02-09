import torch
import torchvision
from architectures import *
from config import *


class VGG(torch.nn.Module):
    def __init__(self,weights="../pretrained/iqaModel_tid_vgg.pth"):
        super(VGG, self).__init__()
        
        # get the pretrained VGG16 network
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg = FC(self.vgg,2,1024,0.25,'vgg16')
        self.vgg.classifier.load_state_dict(torch.load(weights, map_location = device))
        self.vgg.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:30]
        
        # get the max pool of the features stem
        self.max_pool = self.vgg.features[30]
        
        # get the classifier of the vgg16
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
        
        
class RESNET(torch.nn.Module):
    def __init__(self,weights="../pretrained/iqaModel_tid_resnet.pth"):
        super(RESNET, self).__init__()
        
        # get the pretrained ResNet50 network
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = FC(self.resnet,2,1024,0.25,'resnet')
        self.resnet.fc.load_state_dict(torch.load(weights, map_location = device))
        
        # disect the network to access its last convolutional layer
        self.features_conv = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))
        
        # get the avg pool of the features stem
        self.avg_pool = self.resnet.avgpool
        
        # get the classifier of the resnet50 
        self.classifier = self.resnet.fc
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avg_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
