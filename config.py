from torchvision import transforms
import torch 

normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transf = transforms.Compose([
        transforms.ToTensor(),
        normalize_imagenet
    ])
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
