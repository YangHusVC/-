from PIL import Image #2
import numpy as np
import os, pickle,glob
from libtiff import TIFF
 # to open a tiff file for reading:
 
IN_SIZE=200
NUM=20

#tif = TIFF.open(os.path.join(os.getcwd(),'data','trn_input_raw','25.tif'), mode='r')
 # to read an image in the currect TIFF directory and return it as numpy array:
#image_data = tif.read_image()
#print(image_data.shape)
for i in range(NUM):

    print(i)
    tif = TIFF.open(os.path.join(os.getcwd(),'data','trn_input_raw',str(i+1)+'.tif'), mode='r')
    image_data = tif.read_image()
    #print(image_data.shape)
    trn_data=np.pad(image_data,((0,max(IN_SIZE-image_data.shape[0],0)),(0,max(IN_SIZE-image_data.shape[1],0)),(0,0)),'constant',constant_values=(0,0))
    trn_data=trn_data[0:200,0:200,0:14]
    #print(trn_data)

    tif2 = TIFF.open(os.path.join(os.getcwd(),'data','trn_target_raw',str(i+1)+'.tif'), mode='r')
    annotation_data = tif2.read_image()
    anno_data=np.pad(annotation_data,((0,max(0,IN_SIZE-annotation_data.shape[0])),(0,max(0,IN_SIZE-annotation_data.shape[1]))),'constant',constant_values=(0,0))
    print(anno_data.shape)
    anno_data=anno_data[0:200,0:200]
    np.savez(os.path.join(os.getcwd(),'data','trn_input_proc',str(i)+'.npz'), data = trn_data, annotation=anno_data)
    print(anno_data)

npz_files = glob.glob(os.path.join(os.getcwd(),'data','trn_input_proc', '*.npz'))


records = []
for npz in npz_files:
    i = npz.split('/')[-1].split('.')[0]
    #meta = [m for m in meta_files if m.split('/')[-1].split('.')[0]==ii][0]
    records.append({'data':npz})

#shuffle(records)
pickle.dump(records, open(os.path.join(os.getcwd(),'data','trn_input_proc','records.pickle'),'wb'))

