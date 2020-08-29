import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from imagenet_class import idx2label
import torch.nn.functional as F

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert('RGB')
    loader = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = loader(image).float()
    # image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image

def build_model():
    model = resnet50(pretrained=True)
    model.eval()
    return model

def see_lablador_prob(model, image_name):
    with torch.no_grad():
        image = image_loader(image_name)
        out = F.softmax(model(image), dim=1).squeeze(0)
        return "{prob}% 확률로 Lablador 라고 생각됩니다. ㅠㅠ?".format(prob=int(out[208].item()*100))

def inference(model, image_name):
    with torch.no_grad():
        image = image_loader(image_name)
        out = F.softmax(model(image), dim=1).squeeze(0)
        label = idx2label[out.argmax(dim=0).item()]
        prob = int(out.max(dim=0)[0].item()*100)
        return label, prob, "{prob}% 확률로 '{label}' 이라고 생각됩니다. 맞나요?".format(prob=prob, label=label)