import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import tqdm
import cv2

transform = transforms.Compose([
                               transforms.ToTensor(),
                                transforms.Normalize((0.9860190423175113, 0.9665174438361579, 0.9856502582125848), 
                                                     (0.07237375454268005, 0.1562765767163662, 0.07293897151056382))])


class mobilenet(object):
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = ['on_guard', 'overhand_stroke','underhand_stroke']
        self.transform = transform
        # set seed
        torch.manual_seed(46)
        if self.device == 'cuda':
            torch.cuda.manual_seed(46)
        
    def load_model(self,model_name):
        
        
        self.model = models.mobilenet_v3_small(pretrained=False)
        self.model.fc = nn.Linear(1000, 3)
        self.model.load_state_dict(torch.load(model_name)['model_state_dict'])
        self.model.to(self.device)
        
    def predict(self,img):
        self.model.eval()
        
        img = cv2.resize(img,(100,100))
        if self.transform is not None:
            img = self.transform(img)
        with torch.no_grad():
            images = img
            images = torch.unsqueeze(images, 0)
            images = images.to(self.device)
            # calculate outputs by running images through the network
            outputs = self.model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
        if predicted[0].item() < 3:
            return self.classes[predicted[0].item()]