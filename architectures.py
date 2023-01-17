from torchvision import models
import torch

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, feature_extract, use_pretrained=True):
    
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
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft
    
def FC(model,fc_layers,hidden_units,dropOutRate,name):
    if name == 'vgg16' or 'vgg19' or 'vgg11':
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
    elif (name=='resnet'):
        num_ftrs = model.fc.in_features
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
    return model
