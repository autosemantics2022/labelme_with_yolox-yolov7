import sys
import os
import re
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QFileDialog, QComboBox,QGroupBox, QTextBrowser, QLineEdit, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
 
class Yolov7(QWidget):
 
   def __init__(self):
       super().__init__()
       self.start()
 

    #전체 위젯
   def start(self):
       grid1 = QGridLayout()
       grid1.addWidget(self.yolov7train(), 0, 0)
       grid1.addWidget(self.yolov7test(), 0, 1)
       grid1.addWidget(self.yolov7result(), 0, 2)
       self.setLayout(grid1)
       self.setWindowTitle('모델 학습')
       self.show()
    
    #Yolov7로 Train하는 위젯 부분
    #위젯의 첫번째 부분
   def yolov7train(self):
       yolov7train = QGroupBox('Yolov7 Train')

       #위젯 layout을 grid로 설정
       Yolov7TrainLayout = QGridLayout()
       Yolov7TrainLayout.addWidget(self.add2train2017(),1,0,1,1)
       Yolov7TrainLayout.addWidget(self.labelme2yolov7(),0,0,1,1)
       Yolov7TrainLayout.addWidget(self.select_transfer_group(),0,1,2,1)
       Yolov7TrainLayout.addWidget(self.show_model_group(),0,2,2,1)
       Yolov7TrainLayout.addWidget(self.practice_save_group(),0,3,2,1)
    
       yolov7train.setLayout(Yolov7TrainLayout)
       return yolov7train

    #Yolov7으로 추가 학습데이터를 predict하는 위젯 부분
    #위젯의 두번째 부분
   def yolov7test(self):
       yolov7test = QGroupBox('Yolov7 Predict')
       Yolov7TestLayout = QGridLayout()
       Yolov7TestLayout.addWidget(self.select_model_group(),0,0,1,1)
       Yolov7TestLayout.addWidget(self.model_ver_group(),1,0,1,1)
       Yolov7TestLayout.addWidget(self.select_exp_folder(),0,1,1,1)
       Yolov7TestLayout.addWidget(self.txt2json(),1,1,1,1)
       yolov7test.setLayout(Yolov7TestLayout)
       return yolov7test

    #Yolox로 학습되서 만들어진 모델의 성능 평가 위젯 부분
    #위젯의 세번째 부분
   def yolov7result(self):

       yolov7result = QGroupBox('Yolov7 Result')
       Yolov7ResultLayout = QGridLayout()
       Yolov7ResultLayout.addWidget(self.select_model_mea(),0,0,1,1)
       Yolov7ResultLayout.addWidget(self.save_group(),1,0,1,1)
       yolov7result.setLayout(Yolov7ResultLayout)
       return yolov7result


    #Yolov7 Train Start------------------------------------------------------------------------

   #파일 합치기 (resultmerge에 txt 생성 후 각각 train, val, labels로 이동)
   #위젯 구성 부분
   def add2train2017(self):
       add2train2017 = QGroupBox('resultmerge에 대한 labelme2yolov7')
       add = QPushButton('실행')
       self.add_now = QLabel('실행 전')
       vbox_add = QHBoxLayout()
       vbox_add.addWidget(add)
       add2train2017.setLayout(vbox_add)
       
       add.clicked.connect(self.moveresultmerge2train) #add 버튼을 누르면 movereulstmerge2train 함수가 실행됨

       return add2train2017

   #파일 합치기 (resultmerge에 txt 생성 후 각각 train, val, labels로 이동)
   #실제 실행 코드
   def moveresultmerge2train(self):
       self.add_now.setText('실행중입니다')
       os.system('python3 ~/Yolov7-Pytorch/labelme2yolov7seg.py --labelme_dataset_dir ./Yolov7-Pytorch/resultmerge --ouput_dataset_dir ./Yolov7-Pytorch/yolov7_seg_dataset --image_name dataset')
       os.system('cp ~/Yolov7-Pytorch/resultmerge/*.json ~/Yolov7-Pytorch/data/yolov7_seg_dataset/train/json')
       os.system('cp ~/Yolov7-Pytorch/resultmerge/*.jpg ~/Yolov7-Pytorch/data/yolov7_seg_dataset/train/images')
       os.system('cp ~/Yolov7-Pytorch/resultmerge/*.txt ~/Yolov7-Pytorch/data/yolov7_seg_dataset/train/labels')

   #labelme2yolov7 (train, val에 txt 생성 후 각각 train, val, labels로 이동)
   #위젯 구성 부분
   def labelme2yolov7(self):
       labelme2yolov7 = QGroupBox('train, val에 대한 labelme2yolov7')
       play = QPushButton('실행')
       vbox_2COCO = QHBoxLayout()
       vbox_2COCO.addWidget(play)
       labelme2yolov7.setLayout(vbox_2COCO)
        
       play.clicked.connect(self.playClicked) #play 버튼을 누르면 playClicked 함수가 실행됨

       return labelme2yolov7

   #labelme2yolov7 (train, val에 txt 생성 후 각각 train, val, labels로 이동)
   #실제 실행 코드
   def playClicked(self):
       os.system('python3 ~/Yolov7-Pytorch/labelme2yolov7seg.py --labelme_dataset_dir ./Yolov7-Pytorch/train --ouput_dataset_dir ./Yolov7-Pytorch/yolov7_seg_dataset --image_name dataset')
       os.system('python3 ~/Yolov7-Pytorch/labelme2yolov7seg.py --labelme_dataset_dir ./Yolov7-Pytorch/val --ouput_dataset_dir ./Yolov7-Pytorch/yolov7_seg_dataset --image_name dataset')
       os.system('cp ~/Yolov7-Pytorch/train/*.json ~/Yolov7-Pytorch/data/yolov7_seg_dataset/train/json')
       os.system('cp ~/Yolov7-Pytorch/train/*.jpg ~/Yolov7-Pytorch/data/yolov7_seg_dataset/train/images')
       os.system('cp ~/Yolov7-Pytorch/train/*.txt ~/Yolov7-Pytorch/data/yolov7_seg_dataset/train/labels')
       os.system('cp ~/Yolov7-Pytorch/val/*.json ~/Yolov7-Pytorch/data/yolov7_seg_dataset/val/json')
       os.system('cp ~/Yolov7-Pytorch/val/*.jpg ~/Yolov7-Pytorch/data/yolov7_seg_dataset/val/images')
       os.system('cp ~/Yolov7-Pytorch/val/*.txt ~/Yolov7-Pytorch/data/yolov7_seg_dataset/val/labels')
       

   #파라미터 설정에서 입력된 값이 제대로 설정되었으면 모델 부분에서 확인 가능
   #max epoch
   def TextFunction1(self):
       sender1 = self.sender()
       self.epoch2.setText(sender1.text())
   #batch size
   def TextFunction2(self):
       sender2 = self.sender()
       self.batch_size2.setText(sender2.text())
   #learning rate
   def TextFunction3(self):
       sender3 = self.sender()
       self.learning_rate2.setText(sender3.text())
   
  
   #모델을 선택해서 선택한 모델을 보여주는 함수
   def modelopen(self):
       global filename
       filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'ckpt(*.pth)') #ckpt만 선택할 수 있도록 함
       if filename[0]:
           self.backbone.setText(filename[0])   
      
  
    #파라미터 설정 버튼
    #위젯 구성 부분
   def select_transfer_group(self):
       select_transfer_group = QGroupBox('파라미터 설정')

       epoch = QLabel('max epoch :')
       batch_size = QLabel('batch size :')
       learning_rate = QLabel('learning rate :')
 
       self.information = QLabel('Parameter를 입력하고 엔터를 누르면, 값이 설정됩니다.')

       self.epoch_line = QLineEdit(self)
       self.batch_size_line = QLineEdit(self)
       self.learning_rate_line = QLineEdit(self)

       self.epoch_line.returnPressed.connect(self.TextFunction1) #epoch line의 값이 입력된 후 enter를 누르면 TextFunction1함수가 실행됨
       self.batch_size_line.returnPressed.connect(self.TextFunction2) #batch size의 값이 입력된 후 enter를 누르면 TextFunction2함수가 실행됨
       self.learning_rate_line.returnPressed.connect(self.TextFunction3) #learninr rate의 값이 입력된 후 enter를 누르면 TextFunction3함수가 실행됨

       #위젯 layout을 grid로 설정
       grid2 = QGridLayout()
       grid2.addWidget(epoch,0,0)
       grid2.addWidget(self.epoch_line,0,1)
       grid2.addWidget(batch_size,1,0)
       grid2.addWidget(self.batch_size_line,1,1)
       grid2.addWidget(learning_rate,2,0)
       grid2.addWidget(self.learning_rate_line,2,1)
 
       vbox_select_transfer = QVBoxLayout()
       vbox_select_transfer.addWidget(self.information)
       vbox_select_transfer.addLayout(grid2)
       select_transfer_group.setLayout(vbox_select_transfer)
 
       return select_transfer_group

   #전이학습 모델, 파라미터 위젯 부분에서 설정된 값들을 보여주는 위젯
   def show_model_group(self):
       show_model_group = QGroupBox('모델')
       bc = QLabel('backbone:')
       mo = QLabel('model:')
       ep = QLabel('max epoch:')
       ba = QLabel('batch size:')
       le = QLabel('learning rate:')

       self.backbone2 = QLabel('E-ELAN')
       self.model2 = QLabel('yolov7')
       self.epoch2 = QLabel('값')
       self.batch_size2 = QLabel('값')
       self.learning_rate2 = QLabel('값')

       #위젯 layout을 grid로 설정
       grid3 = QGridLayout()
       grid3.addWidget(bc,0,0)
       grid3.addWidget(self.backbone2,0,1)
       grid3.addWidget(mo,1,0)
       grid3.addWidget(self.model2,1,1)
       grid3.addWidget(ep,2,0)
       grid3.addWidget(self.epoch2,2,1)   
       grid3.addWidget(ba,3,0)
       grid3.addWidget(self.batch_size2,3,1)
       grid3.addWidget(le,4,0)
       grid3.addWidget(self.learning_rate2,4,1)

       show_model_group.setLayout(grid3)

       return show_model_group

   #사용자가 선택하고 입력한 파라미터에 맞춰서 코드 다시 수정하기
   def model_practice(self):
        #모델의 파라미터 설정은 file_path에 있는 코드를 수정함으로써 이루어질 수 있음
        file_path = "~/Yolov7-Pytorch/data/hyp.scratch-high.yaml"

        with open(file_path, "r") as f:
            lines = f.readlines()
        with open(file_path, "w") as f:
            for line1 in lines:
                f.write(line1.replace(line1,'lr0: '+self.learning_rate_line.text()))
                break
    
   #파라미터가 설정된 후 train을 시작시키는 위젯
   #위젯 구성 부분
   def practice_save_group(self):
       practice_save_group = QGroupBox('모델 학습 및 저장')
       self.practice = QPushButton(self)
       self.practice.setText('yolov7 모델 학습 실행')
       self.practice.clicked.connect(self.model_practice) #practice 버튼을 누르면 사용자가 입력한 토대로 모델의 파라미터를 수정하는 model_practice가 실행 됨
       self.now.setWordWrap(True)

       vbox_practicie_save = QVBoxLayout()
       vbox_practicie_save.addWidget(self.practice)
       practice_save_group.setLayout(vbox_practicie_save)
 
       self.practice.clicked.connect(self.practiceClicked)  #practice 버튼을 누르면 train을 시작하는 practcieClicked 함수가 실행이 됨
       
       return practice_save_group

   #파라미터가 설정된 후 train을 시작시키는 위젯
   #실제 train 시작 코드
   def practiceClicked(self):
       os.system('python3 ~/Yolov7-Pytorch/segment/train.py --data ~/Yolov7-Pytorch/data/custom.yaml --batch '+self.batch_size2.text()+' --weights ~/Yolov7-Pytorch/data/yolov7-seg.pt --cfg ~/Yolov7-Pytorch/data/yolov7-seg.yaml --epochs '+self.epoch_line.text()+' --name yolov7-seg --img 640 --hyp ~/Yolov7-Pytorch/data/hyp.scratch-high.yaml')
       

   #Yolov7 Train End------------------------------------------------------------------------
   

   #Yolov7 Predict Start------------------------------------------------------------------------
    
   #predict를 돌릴 모델을 선택해서 선택한 모델을 보여주는 함수
   def modelopen1(self):
        #global filename
        global filepath
        self.filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'ckpt(*.pt)') 
        self.filepath = self.filename[0]
        self.arr = self.filename[0].split('/')
        self.fileName = self.arr[-1]

        if self.filename[0]:
            self.label5.setText(self.filepath)

    
   #predict를 돌릴 모델 설정 
   def expopen(self):
        global filename1
        global filepath1
        self.filename1 = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.filepath1 = self.filename1
        
        self.label10.setText(self.filepath1)

   #predict된 이미지와 txt를 토대로 json 파일 만들어내기
   def makejson(self):
       edited_lines = []
       list = self.filepath1.split('/')
       filepathend = list[3]+'/'+list[4]+'/'+list[5]+'/'+list[6]+'/'
       with open('Yolov7-Pytorch/txt2json.py','r',encoding='UTF8') as f:
           lines = f.readlines()
       for line in lines:
           # 조건에 따라 원하는 대로 line을 수정
           if 'file_path = ' in line:
               edited_lines.append('file_path = "'+filepathend+'labels/" \n')
           elif 'file_path2 = ' in line:
               edited_lines.append('file_path2 = "'+filepathend+'" \n')
           else:
               edited_lines.append(line)
       with open('Yolov7-Pytorch/txt2json.py', 'w') as f:
           f.writelines(edited_lines)
       os.system('python3 Yolov7-Pytorch/txt2json.py')
       os.system('cp -r ~/Yolov7-Pytorch/test2017/* ~/Yolov7-Pytorch/resultmerge') #predict된 학습데이터가 다시 전체 학습데이터로 추가되어야 하므로 segment 픽셀처리가 되지 않은 원본의 이미지가 필요 
     #resultmerge에 img+json이 있음, 사용자가 여기에서 polygon 수정 가능
    

     #적용 모델 및 파라미터 설정 
   def select_model_group(self):
        select_model_group = QGroupBox('적용 모델 설정')
        model = QPushButton(self) 
        model.setText('적용 모델 선택')
        model.clicked.connect(self.modelopen1)
        self.label5 = QLabel('적용 모델')

        vbox_select_model = QVBoxLayout()
        vbox_select_model.addWidget(model)
        vbox_select_model.addWidget(self.label5)
        select_model_group.setLayout(vbox_select_model)

        return select_model_group

    #모델 검증 실행
    #위젯 구성 부분
   def model_ver_group(self):
        model_ver_group = QGroupBox('predict 실행')
        ver = QPushButton(self)
        ver.setText('실행')

        vbox_ver = QHBoxLayout()
        vbox_ver.addWidget(ver)
        model_ver_group.setLayout(vbox_ver)

        ver.clicked.connect(self.demoPlay)

        return model_ver_group
    
    #exp 폴더 선택
    #위젯 구성 부분
   def select_exp_folder(self):
        select_exp_folder = QGroupBox('predict된 exp 선택')
        model = QPushButton(self) 
        model.setText('predict된 exp 선택')
        model.clicked.connect(self.expopen)
        self.label10 = QLabel('적용 폴더')

        vbox_select_model = QVBoxLayout()
        vbox_select_model.addWidget(model)
        vbox_select_model.addWidget(self.label10)
        select_exp_folder.setLayout(vbox_select_model)

        return select_exp_folder

    #predict된 이미자와 txt파일을 토대로 json파일 만들기
    #위젯 구성 부분
   def txt2json(self):
        txt2json = QGroupBox('txt+img -> json')
        ver = QPushButton(self)
        ver.setText('실행')

        vbox_ver = QHBoxLayout()
        vbox_ver.addWidget(ver)
        txt2json.setLayout(vbox_ver)

        ver.clicked.connect(self.makejson)  #ver 클릭시 json을 만들어주는 makejosn함수가 실행됨
        return txt2json


    #모델 검증 실행
    #실제 코드 부분
   def demoPlay(self):
        os.system('python3 ~/Yolov7-Pytorch/segment/predict.py --weights "Yolov7-Pytorch/runs/train-seg/yolov7-seg19/weights/best.pt" --source "Yolov7-Pytorch/test2017" --conf 0.05 --save-txt')

       
    #model을 선택하는 함수
   def modelopen(self):        
        self.filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'ckpt(*.pth)') 
        self.arr = self.filename[0].split('/')
        self.fileName = self.arr[-1]

        if self.filename[0]:
            self.label6.setText(self.fileName)


   #Yolov7 Predict End------------------------------------------------------------------------


   #Yolov7 Result Start------------------------------------------------------------------------
  
   #성능 측정할 모델을 선택해서 선택한 모델을 보여주는 함수
   #위젯 구성 부분
   def select_model_mea(self):
        select_model_mea = QGroupBox('성능 측정 모델 설정')
        model_mea = QPushButton(self) 
        model_mea.setText('성능 측정 모델 선택')
        model_mea.clicked.connect(self.ptopen)  #model_mea 버튼이 눌리면 성능측정을 위한 모델을 선택할 수 있는 ptopen 함수가 실행된다
        self.label6 = QLabel('성능 측정 모델')

        vbox_model_mea = QVBoxLayout()
        vbox_model_mea.addWidget(model_mea)
        vbox_model_mea.addWidget(self.label6)
        select_model_mea.setLayout(vbox_model_mea)

        return select_model_mea

   #선택된 exp 폴더에 있는 best.pt의 결과를 보여줌 
   def ptopen(self):
        global filename2
        global filepath2
        self.filename2 = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.filepath2 = self.filename2
        
        self.label6.setText(self.filepath2)



    #모델학습 및 결과 저장
    #위젯 구성 부분
   def save_group(self):
        save_group = QGroupBox('성능 출력')
        save = QPushButton(self)
        save.setText('실행 후 결과 출력')
        self.output = QTextBrowser()

        vbox_save = QVBoxLayout()
        vbox_save.addWidget(save)
        save_group.setLayout(vbox_save)

        save.clicked.connect(self.showOutput)  #save 버튼이 눌리면 성능 측정의 모델의 결과를 txt로 저장하고 저장된 결과를 dialog로 불러와주는 showOutput 함수가 실행됨

        return save_group

    #실제 실행 코드
    #성능 측정의 모델의 결과를 가져와 dialog로 불러와줌
   def showOutput(self):
        self.dialog = QDialog()
        self.tb = QTextBrowser()
        lay = QHBoxLayout()
        lay.addWidget(self.tb)
        self.dialog.setLayout(lay)

        f = open(self.filename2+'/results.csv','r', encoding='utf-8')
        with f:
            data = f.read()
            self.tb.setText(data)

        self.dialog.setWindowTitle('성능출력창')
        self.dialog.resize(500,500)
        self.dialog.show()

