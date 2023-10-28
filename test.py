#newly added
import dlib
import numpy as np
import cv2
from torchvision.models import *
import torchvision.models as models
import pandas as pd
from glob import glob #for finding files recursively
from collections import defaultdict
from PIL import Image

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils
import torchvision.datasets as dset

from torch import optim
from torch.utils.data import DataLoader,Dataset

from torchvision.datasets import ImageFolder
from torch.autograd import Variable
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    

from facenet_pytorch import InceptionResnetV1
import torchvision.models as models

def align_face(image, detector, predictor):
        #newly added for dlib
        # Convert the PIL Image to a numpy array
        img_np = np.array(image)
        # Convert RGB to BGR
        img_np = img_np[:, :, ::-1].copy()

        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)
        for rect in dets:
            shape = predictor(gray, rect) #changed
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            aligned_face = dlib.get_face_chip(img_np, shape)
            return aligned_face


class trainingDataset(Dataset):
       def __init__(self, _data, _transform=None):
           self.data = _data #choose either train or val dataset to use
           self.transform = _transform

       def __len__(self):
           return len(self.data)#essential for choose the num of data in one epoch


        
       def __getitem__(self, index):
           try:
                   #newly added for dlib
               first_img_path = self.data[index][0]
               second_img_path = self.data[index][1]
               img1Opened = Image.open(first_img_path)
               img2Opened = Image.open(second_img_path)

               # Align the images
               img1Aligned = align_face(img1Opened, detector, predictor)
               img2Aligned = align_face(img2Opened, detector, predictor)

               if img1Aligned is None or img2Aligned is None:
                   # Handle the case where alignment fails (e.g., return placeholder images)
                   img1Aligned = img1Opened
                   img2Aligned = img2Opened
                # Transform the images into tensor format
               if self.transform is not None:
                   img1Aligned = Image.fromarray((np.array(img1Aligned) * 255).astype(np.uint8))
                   img2Aligned = Image.fromarray((np.array(img2Aligned) * 255).astype(np.uint8))
                   img1Aligned = self.transform(img1Aligned)
                   img2Aligned = self.transform(img2Aligned)

               return img1Aligned, img2Aligned, self.data[index][2]
           except Exception as e:
               print(f"Error loading item at index {index}: {e}")
               return None  # or some default value
class SiameseNetwork_ResNet(nn.Module):
    def __init__(self):
        super(SiameseNetwork_ResNet, self).__init__()

        model = models.resnet101(pretrained=True)
        for param in model.parameters():
          param.require_grad = False #change this to true

        self.cnn1 = torch.nn.Sequential(*(list(model.children())[:-1]))



        model.classifier = nn.Sequential(
            nn.Linear(4096,460),
            nn.ReLU(),
            nn.Linear(460, 230),
            nn.ReLU(),
            nn.Linear(230, 2)
        )
        self.fc = model.classifier

    def forward(self, input1, input2):
        output1 = self.cnn1(input1)
        output1 = output1.view(output1.size()[0], -1)
        output2 = self.cnn1(input2)
        output2 = output2.view(output2.size()[0], -1)

        output = torch.cat((output1, output2),1)
        output = self.fc(output)
        return output
        

class ContrastiveLoss(nn.Module):
     def __init__(self, margin=2.0):
          super(ContrastiveLoss, self).__init__()
          self.margin = margin

     def forward(self, output1, output2, label):
         euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
         return loss_contrastive
           
class testDataset(Dataset):

    def __init__(self,transform=None):
        self.test_df = pd.read_csv('sample_submission.csv')
        self.transform = transform

    def __getitem__(self,index):

        img0_path = self.test_df.iloc[index].img_pair.split("-")[0]
        img1_path = self.test_df.iloc[index].img_pair.split("-")[1]

        img0 = Image.open('test/'+img0_path)
        img1 = Image.open('test/'+img1_path)

        # Align the images using dlib
        img0Aligned = align_face(img0, detector, predictor)
        img1Aligned = align_face(img1, detector, predictor)

        # If alignment fails, use the original images
        if img0Aligned is None:
            img0Aligned = img0
        else:
            img0Aligned = Image.fromarray((np.array(img0Aligned) * 255).astype(np.uint8))

        if img1Aligned is None:
            img1Aligned = img1
        else:
            img1Aligned = Image.fromarray((np.array(img1Aligned) * 255).astype(np.uint8))

        if self.transform is not None:
            img0 = self.transform(img0Aligned)
            img1 = self.transform(img1Aligned)

        return img0, img1

    def __len__(self):
        return len(self.test_df)


def main():

    # model = models.vgg19(weights=VGG19_Weights.DEFAULT)
    # print(model)

    mdl = InceptionResnetV1(pretrained='vggface2', num_classes=2, classify=True)

    # test = torch.nn.Sequential(*(list(mdl.children())[:-1]))
    # print(test)


    relationshipsCSV = pd.read_csv("train_relationships.csv")
    train_images_folder = "train/"

    available_images = glob(train_images_folder + "*\*\*.jpg")
    '''
    Only for windows pc:
    '''
    for a in range(0, len(available_images)):
        available_images[a] = available_images[a].replace("\\\\", "\\")

    all_ppl = [x.split("\\")[-3] + "\\" + x.split("\\")[-2] for x in available_images] #all the peoplex
    #creating the training set
    personPathFile = defaultdict(list)
    for x in available_images:
        personPathFile[x.split("\\")[-3] + "\\" + x.split("\\")[-2]].append(x)


    #read from the csv to create a list of tuples
    relationships = relationshipsCSV
    relationship_pairs = [(row['p1'], row['p2']) for index, row in relationships.iterrows()] # Create a list of tuples
    relationship_pairs = [x for x in relationship_pairs if x[0] in all_ppl and x[1] in all_ppl] #data cleaning

    #converting into dictionary of person is related to this list of people
    relationshipDict = defaultdict(list)
    for item in relationship_pairs:
        relationshipDict[item[0]].append(item[1]) #you do not need to consider the item[1] because the CSV is already pretty clean


    #create more data by finding the 'negative pair'
    notRelationshipDict = defaultdict(list)
    set_all_ppl = set(all_ppl)
    count=0
    #for each key(person) in the relationship dictionary, find the non-relations
    for k,v in relationshipDict.items():
        notRelationshipDict[k]=list(set_all_ppl-set(v))



    # convert both dictionaries into a list of tuples with 1, 0 as labels
    # for each of the image, take the first image path stored in the dictionary made earlier
    # then take each of the path file in the item

    trainData = []
    for k,v in relationshipDict.items():
        for relation in v:
            trainData.append((personPathFile[k][0], personPathFile[relation][0],1))
            try:
                trainData.append((personPathFile[k][0], personPathFile[relation][1],1))
            except:
                pass

    positiveRelationsCount = len(trainData)
    NUM_DATA = 1
    for k,v in notRelationshipDict.items():
        count1 = 0
        count2 = 0
        for relations in v:
            if count1 <= NUM_DATA and k[:5]==relations[:5]:
                trainData.append((personPathFile[k][0], personPathFile[relations][0],0))
                count1 += 1
            elif count2 <= NUM_DATA and k[:5]!=relations[:5]:
                trainData.append((personPathFile[k][0], personPathFile[relations][0],0))
                count2 += 1
            elif count1>NUM_DATA and count2>NUM_DATA:
                break




    from sklearn.model_selection import train_test_split
    BATCH_SIZE=128
    IMG_SIZE=100
    NUM_WORKERS = 8


    trainSet, valSet = train_test_split(trainData, test_size=0.5, random_state=42)
    #TODO: Normalize the data

    #Training set and training loader
    trainset = trainingDataset(_data=trainSet,
                                            _transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                                                                        transforms.ToTensor()
                                                                        ]))

    valset = trainingDataset(_data=valSet,
                                            _transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                                                                        transforms.ToTensor()
                                                                        ]))


    trainloader = DataLoader(trainset,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            batch_size=BATCH_SIZE)  # Using a small batch size for debugging



    valloader = DataLoader(valset,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            batch_size=BATCH_SIZE)


    model = SiameseNetwork_ResNet()
    
    from PIL import Image

    net = SiameseNetwork_ResNet().cuda()
    criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss #TODO: change to BCELoss or BSEwithLogitLoss
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    import numpy as np

    counter = []
    iteration_number= 0

    BATCH_SIZE=128
    NUMBER_EPOCHS=20 #10 is already done --> total 30 epochs
    epoch_accuracies = np.array([])
    net.train()

    for epoch in range(0,NUMBER_EPOCHS):
        total_loss = 0
        for i, data in enumerate(trainloader,0):
            img0, img1 , labels = data
            img0, img1 , labels = img0.cuda(), img1.cuda() , labels.cuda() #if you need to add it to the GPU to make it run faster
            optimizer.zero_grad()
            outputs = net(img0,img1) #run the model
            loss = criterion(outputs,labels) #run it through the loss criterion
            loss.backward() #update the loss backward
            optimizer.step()
            total_loss += loss.item() #add the into the total loss


        #this is to check how well the model is training
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in valloader:
                img0, img1 , labels = data
                img0, img1 , labels = img0.cuda(), img1.cuda() , labels.cuda() #if you need add it to the GPU to make it run faster
                outputs = net(img0,img1) #get an output from the model
                _, predicted = torch.max(outputs.data, 1) #get the prediction
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        accuracy = 100 * correct_val / total_val
        epoch_accuracies = np.append(epoch_accuracies, accuracy)
        print(f"Epoch [{epoch+1}/{NUMBER_EPOCHS}]  Loss: {total_loss/len(trainloader)}, Accuracy: {accuracy}")
        '''
        # In case very large epochs are being run , we can save the path then resume training later
        # not sure whether this will work but it is an idea
        saved_path = "/content/drive/MyDrive/Trial_V_8.pth"
        torch.save(net, saved_path)
        '''

    best_accuracy = np.amax(epoch_accuracies)
    best_epoch = epoch_accuracies.argmax()
    print('END TRAINING: BEST EPOCH: ', best_epoch, ' WITH ACCURACY: ', best_accuracy)


    testset = testDataset(transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                                                                        transforms.ToTensor()
                                                                        ]))
    testloader = DataLoader(testset,
                            shuffle=False,
                            num_workers=0,
                            batch_size=1)#Both extra workers and batch size lead to data out of order, the submission.csv will be


    test_df = pd.read_csv('sample_submission.csv')
    predictions=[]
    net.eval()
    with torch.no_grad():
        for data in testloader:
            img0, img1 = data
            img0, img1 = img0.cuda(), img1.cuda()
            output = net(img0,img1)
            _, predicted = torch.max(output, 1)
            predictions = np.concatenate((predictions,predicted.cpu().numpy()),0)

    test_df['is_related'] = predictions
    test_df.to_csv("test_Results.csv", index=False) #submission.csv should be placed directly in current fold.
    test_df.head(50)#show the result to be committed

if __name__ == '__main__':
    main()
