import os
import numpy as np

def load_annoatation(p):
    #load annoatation from txt
    #txt: picname label xmin ymin xmax ymax
    #p:path of txt
    text_polys=[]
    text_label=[]
    if not os.path.exists(p):
        print('error: no such txt exist!')
        return np.array(text_polys, dtype=np.float32), np.array(text_label, dtype=np.str)
    with open(p,'r') as f:
        reader = f.readlines()
        for line in reader:
            line=line.strip()
            linelist=line.split(' ')
            text_label.append(linelist[0])
            xmin=linelist[1]
            ymin = linelist[2]
            xmax = linelist[3]
            ymax = linelist[4]
            text_polys.append([xmin,ymin,xmax,ymax])
    return np.array(text_polys, dtype=np.float32), np.array(text_label, dtype=np.str)
#    return text_polys,text_label


def compute_IOU(rec1,rec2):
    #rec: rectangle, [xmin ymin xmax ymax]
    #return IoU of rec1 and rec2
    width=max(0,min(rec1[2],rec2[2])-max(rec1[0],rec2[0]))
    hight=max(0,min(rec1[3],rec2[3])-max(rec1[1],rec2[1]))
    inter=width*hight
    union=(rec1[3]-rec1[1])*(rec1[2]-rec1[0])+(rec2[3]-rec2[1])*(rec2[2]-rec2[0])-inter
    return inter/union



dir_gt='./data/gt'#directory of gt
dir_pt='./data/pt'#directory of predict txt
num_correct=0
num_miss=0
num_error=0
for file in os.listdir(dir_gt):
    if not file.endswith(".txt"):
        continue
    path_gt=os.path.join(dir_gt,file)
    gt = load_annoatation(path_gt)
    path_pt=os.path.join(dir_pt,file)
    if not os.path.exists(path_pt):
        print('error!! such txt is no exist:{}'.format(file))
        num_miss=num_miss+len(gt[0])
        print('in this step , {} miss has been count with no correct!'.format(len(gt[0])))
        continue
    #print(file)
    pt=load_annoatation(path_pt)
    #gt=([[125,125,136,136],[111,111,136,136]],['s2','s1'])
    #pt=([[125,125,140,140],[100,100,136,136],[123.2,123.3,125,127]],['s2','s2','s1'])
    count=0
    match =np.zeros(100)
    for i in range (0,len(gt[0])):

        rec1=gt[0][i]
        for j in range (0,len(pt[0])):
            rec2=pt[0][j]

            #print(compute_IOU(rec1,rec2))
            if compute_IOU(rec1,rec2)>0.5 and gt[1][i]==pt[1][j] and match[j]==0:
                count=count+1
                match[j] = 1
                break



    tmp_num_correct=count
    tmp_num_error=len(pt[0])-count
    tmp_num_miss=len(gt[0])-count
    #print(tmp_num_miss)
    #print(tmp_num_correct,tmp_num_error,tmp_num_miss)
    num_miss=num_miss+tmp_num_miss
    num_error=num_error+tmp_num_error
    num_correct=num_correct+tmp_num_correct
print('correct: {}, error: {}, miss: {}'.format(num_correct,num_error,num_miss))
mAP=num_correct/(num_correct+num_error)
mAR=num_correct/(num_correct+num_miss)
F_measure=2*mAP*mAR/(mAP+mAR)

print(' mAP={}\n mAR={}\n F-measure={}'.format(mAP,mAR,F_measure))
