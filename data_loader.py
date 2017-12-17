import numpy as np
import cv2

FOLDER_DATASET = "data/"
IMAGE_DATASET = "UCF101_images/"
class DataClass():

    def __init__(self, data_folder, image_folder, file_name, transform=None):
        self.transform = transform
        self.data_folder = data_folder
        self.image_folder = image_folder
        self.width = 320
        self.height = 240
        
        with open(self.data_folder + file_name) as f:
            content = f.readlines()
        
        self.data = np.asarray([
            [i.split(" ")[0], i.split(" ")[1], i.split(" ")[2].split("\n")[0]] for i in content])

    # Generate image files for the given batch of videos
    # return batch_size * longest_sequence * channels * height * width
    def generatebatch(self, meta_batch):

        folder = self.data_folder + self.image_folder
        batch_len = len(meta_batch)

        maximum_video_length = meta_batch[:,1].astype(int).max()        
        arr = []
        for batch_index, file in enumerate(meta_batch):
            
            filename = file[0]
            sequence_len = int(file[1])
            # generate transformation here if you want to
            current_image = []
            for i in range(0, sequence_len): #pad the beginning
                image = cv2.imread(folder + filename + "_" + str(i) + ".jpg")                
                # apply transformation here if you want to
#                 image = cv2.resize(image, (267,200), interpolation = cv2.INTER_AREA)

                image = image.transpose(2,0,1)
                current_image.append(image)
            
            #repeat image/reflection
            current_image = np.tile(current_image, (int(np.ceil(maximum_video_length/float(sequence_len))),1,1,1))
            
            #add it to the batch_array
            arr.append(current_image[:maximum_video_length])            
        return np.asarray(arr)        
        
    # Get a batch of given batch size
    def getbatch(self, batchsize):
        batch = np.random.choice(len(self.data), batchsize, replace=False)
        batch = self.data[batch]
        labels = batch[:,2].astype(int)
        final_batch = self.generatebatch(batch)
        return final_batch, labels
    
    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.data)

