from PIL import Image
import numpy as np
import pickle

def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

id_to_data={}
id_to_size={}

with open("./data/images.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,path=line.split(" ",1)
        image=Image.open("./data/images/"+path).convert('RGB')
        id_to_size[int(id)]=np.array(image,dtype=np.float32).shape[0:2]
        image=image.resize((224,224))
        image=np.array(image,dtype=np.float32)
        image=image/255
        image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
        id_to_data[int(id)]=image

id_to_data=np.array(list(id_to_data.values()))
id_to_size=np.array(list(id_to_size.values()))
f=open("./id_to_data","wb+")
pickle.dump(id_to_data,f,protocol=4)
f=open("./id_to_size","wb+")
pickle.dump(id_to_size,f,protocol=4)

id_to_box={}

with open("./data/bounding_boxes.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,box=line.split(" ",1)
        box=np.array([float(i) for i in box.split(" ")],dtype=np.float32)
        box[0]=box[0]/id_to_size[int(id)-1][1]*224
        box[1]=box[1]/id_to_size[int(id)-1][0]*224
        box[2]=box[2]/id_to_size[int(id)-1][1]*224
        box[3]=box[3]/id_to_size[int(id)-1][0]*224
        id_to_box[int(id)]=box
id_to_box=np.array(list(id_to_box.values()))
f=open("./id_to_box","wb+")
pickle.dump(id_to_box,f,protocol=4)
