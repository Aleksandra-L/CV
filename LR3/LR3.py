import os
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import csv


def accuracy_top5(model):
    num_correct = 0
    model.eval()
    y = []
    with open('test_labels2.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter='\n')
        for row in readCSV:
            splitted = row[0].split(',')
            y.append(splitted[1])
    j = 1
    with torch.no_grad():
        folder_dir = "test_data"
        for images in os.listdir(folder_dir):
            if (images.endswith(".jpg")):
                img = Image.open(folder_dir + '/' + images)
                img_t = transform(img)
                batch_t = torch.unsqueeze(img_t, 0)
                model.eval()
                out = model(batch_t)
                sorted, indices = torch.sort(out, descending=True)
                percentage = F.softmax(out, dim=1)[0] * 100.0
                results = [(classes[i]) for i in indices[0][:5]]
                k = 0
                for i in results:
                    if y[j] == i:
                        num_correct += 1
                    k += 1
                j += 1
        print(f"Top5 Accuracy of the model: {float(num_correct)/float(len(y))*100:.2f}")


def accuracy_top1(model):
    num_correct = 0
    model.eval()
    y = []
    with open('test_labels2.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter='\n')
        for row in readCSV:
            splitted = row[0].split(',')
            y.append(splitted[1])
    j = 1
    with torch.no_grad():
        folder_dir = "test_data"
        for images in os.listdir(folder_dir):
            if (images.endswith(".jpg")):
                img = Image.open(folder_dir + '/' + images)
                img_t = transform(img)
                batch_t = torch.unsqueeze(img_t, 0)
                model.eval()
                out = model(batch_t)
                sorted, indices = torch.sort(out, descending=True)
                percentage = F.softmax(out, dim=1)[0] * 100.0
                results = [(classes[i]) for i in indices[0][:1]]
                for i in results:
                    if y[j] == i:
                        num_correct += 1
                    j += 1
        print(f"Top1 Accuracy of the model: {float(num_correct)/float(len(y))*100:.2f}")


def output(out):
    sorted, indices = torch.sort(out, descending=True)
    percentage = F.softmax(out, dim=1)[0] * 100.0
    results = [(classes[i], percentage[i].item()) for i in indices[0][:5]]
    print("\n5 наиболее вероятных классов")
    for i in range(5):
        print('{}: {:.4f}%'.format(results[i][0], results[i][1]))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

alexnet = models.alexnet(weights='IMAGENET1K_V1')
resnet = models.resnet50(weights='IMAGENET1K_V1')
inception = models.inception_v3(weights='IMAGENET1K_V1')

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

with open('imagenet_classes1.txt') as labels:
    classes = [i.strip() for i in labels.readlines()]

print('AlexNet')
accuracy_top1(alexnet)
accuracy_top5(alexnet)

print('ResNet50')
accuracy_top1(resnet)
accuracy_top5(resnet)

print('InceptionV3')
accuracy_top1(inception)
accuracy_top5(inception)

folder_dir = "animals\goose"
for images in os.listdir(folder_dir):
    if (images.endswith("6bb8056eaa.jpg")):
        img = Image.open(folder_dir + '/' + images)
        # print(images)
        # img.show()
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        alexnet.eval()
        out_alexnet = alexnet(batch_t)
        output(out_alexnet)

        resnet.eval()
        out_resnet = resnet(batch_t)
        output(out_resnet)

        inception.eval()
        out_inception = inception(batch_t)
        output(out_inception)