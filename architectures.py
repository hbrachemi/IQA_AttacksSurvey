from torchvision import models
import torch
import timm

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_my_state_dict(model, state_dict,exceptions):
 
        own_state = model.state_dict()
        for name, param in state_dict.items():
          try:
            param = param.data
            own_state[name].copy_(param)
          except:
            print('layer not copied: '+name)
            



def initialize_model(model_name, feature_extract, use_pretrained=True,channels = None, cnn_backbone = None):
    
    model_ft = None
    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if feature_extract:
          x = [torch.nn.Identity(num_ftrs,num_ftrs)]
          model_ft.fc = torch.nn.Sequential(*x)
        else:
          model_ft.fc = torch.nn.Linear(num_ftrs,1)
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[0].in_features
        if feature_extract:
          x = [torch.nn.Identity(num_ftrs,num_ftrs)]
          model_ft.classifier = torch.nn.Sequential(*x)
        else:
          model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,1)
    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model_ft.classifier[0].in_features
        if feature_extract:
          x = [torch.nn.Identity(num_ftrs,num_ftrs)]
          model_ft.classifier = torch.nn.Sequential(*x)
        else:
          model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,1)
    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model_ft.classifier[0].in_features
        if feature_extract:
          x = [torch.nn.Identity(num_ftrs,num_ftrs)]
          model_ft.classifier = torch.nn.Sequential(*x)
        else:
          model_ft.classifier[-1] = torch.nn.Linear(num_ftrs,1)
    
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = torch.nn.Conv2d(512, 1, kernel_size=(1,1), stride=(1,1))

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if feature_extract:
          x = [torch.nn.Identity(num_ftrs,num_ftrs)]
          model_ft.classifier = torch.nn.Sequential(*x)
        else:
          model_ft.classifier = torch.nn.Linear(num_ftrs, 1)

    elif model_name == "inception":
        """ Inception v3
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features

        if feature_extract:
          x = [torch.nn.Identity(num_ftrs,num_ftrs)]
          model_ft.fc = torch.nn.Sequential(*x)
        else:
          # Handle the auxilary net
          num_ftrs = model_ft.AuxLogits.fc.in_features
          model_ft.AuxLogits.fc = torch.nn.Linear(num_ftrs, 1)
          # Handle the primary net
          model_ft.fc = torch.nn.Linear(num_ftrs,1)
    elif model_name == "vit":
        """ VIsion Transformer
        """
        model_ft = ViT()
        if use_pretrained:
          model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)
          target_dict = model_ft.state_dict()

          new_keys = list(model_ft.state_dict().keys())
          for key,n_key in zip(target_dict.keys(), new_keys):
            target_dict = target_dict.copy()
            target_dict[n_key] = target_dict.pop(key)
          load_my_state_dict(model_ft,target_dict,'')

    elif model_name == "cvit":
        """ CNN and VIsion Transformer
        """
        model_ft = initialize_model(cnn_backbone,feature_extract, use_pretrained=True,channels = None)
        model_ft.avgpool= torch.nn.Identity()
        model_ft = torch.nn.Sequential(*(list(model_ft.children())[:-1]))
        model_vit = ViT(in_channels=channels,patch_size=1,img_w=20, img_h=20)
        try:
        	model_ft.fc = model_vit
        except:
                model_ft.classifier = model_vit
       
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft
    
def FC(model,fc_layers,hidden_units,dropOutRate,name):
    if name in ['vgg16','vgg19','vgg11']:
        num_ftrs = 512
        x=[torch.nn.Linear(num_ftrs,hidden_units)]
        for i in range(fc_layers):
            x.append(torch.nn.Linear(hidden_units,hidden_units,bias=True))
            x.append(torch.nn.ReLU(inplace=False))
            x.append(torch.nn.Dropout(dropOutRate,inplace=False))
        x.append(torch.nn.Linear(hidden_units,1,bias=True))
        x.append(torch.nn.ReLU(inplace=False))
        Fc=torch.nn.Sequential(*x)
        model.classifier=Fc    	
    elif name in ['resnet','inception']:
        num_ftrs = 2048
        x=[torch.nn.Linear(num_ftrs,hidden_units)]
        for i in range(fc_layers):
            x.append(torch.nn.Linear(hidden_units,hidden_units,bias=True))
            x.append(torch.nn.ReLU(inplace=False))
            x.append(torch.nn.Dropout(dropOutRate,inplace=False))
        x.append(torch.nn.Linear(hidden_units,1,bias=True))
        x.append(torch.nn.ReLU(inplace=False))
        Fc=torch.nn.Sequential(*x)
        model.fc=Fc
    else:
        num_ftrs = model.classifier[0].in_features
        x=[torch.nn.Linear(num_ftrs,hidden_units)]
        for i in range(fc_layers):
            x.append(torch.nn.Linear(hidden_units,hidden_units,bias=True))
            x.append(torch.nn.ReLU(inplace=False))
            x.append(torch.nn.Dropout(dropOutRate,inplace=False))
        x.append(torch.nn.Linear(hidden_units,1,bias=True))
        x.append(torch.nn.ReLU(inplace=False))
        Fc=torch.nn.Sequential(*x)
        model.classifier=Fc
        
        
    try:
          model.fc.train()
    except:
          try:
            model.classifier.train()
          except:
            print("no fc or classifier")
    
    return model
