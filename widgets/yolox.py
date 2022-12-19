import sys
import os
import re
#pyqt로 위젯을 구성하였기에 필요한 class들을 import
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QFileDialog, QComboBox,QGroupBox, QTextBrowser, QLineEdit, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
 
class Yolox(QWidget):
 
   def __init__(self):
       super().__init__()
       self.start()
 
    #전체 위젯
   def start(self):
       grid1 = QGridLayout()
       grid1.addWidget(self.yoloxtrain(), 0, 0)
       grid1.addWidget(self.yoloxtest(), 0, 1)
       grid1.addWidget(self.yoloxresult(), 0, 2)
       self.setLayout(grid1)
       self.setWindowTitle('모델 학습')
       self.show()
    

    #Yolox로 Train하는 위젯 부분
    #위젯의 첫번째 부분
   def yoloxtrain(self):
       yoloxtrain = QGroupBox('Yolox Train')
       
       #위젯 layout을 grid로 설정
       YoloxTrainLayout = QGridLayout()
       YoloxTrainLayout.addWidget(self.add2train2017(),0,0,1,1)
       YoloxTrainLayout.addWidget(self.labelme2COCO(),1,0,1,1)
       YoloxTrainLayout.addWidget(self.select_transfer_group(),0,1,2,1)
       YoloxTrainLayout.addWidget(self.show_model_group(),0,2,2,1)
       YoloxTrainLayout.addWidget(self.practice_save_group(),0,3,2,1)
       
       yoloxtrain.setLayout(YoloxTrainLayout)
       return yoloxtrain

    #Yolox로 추가 학습데이터를 demo하는 위젯 부분
    #위젯의 두번째 부분
   def yoloxtest(self):
       yoloxtest = QGroupBox('Yolox Test')
       YoloxTestLayout = QGridLayout()
       YoloxTestLayout.addWidget(self.select_model_group(),0,0,1,1)
       YoloxTestLayout.addWidget(self.model_ver_group(),1,0,1,1)
       yoloxtest.setLayout(YoloxTestLayout)
       return yoloxtest


    #Yolox로 학습되서 만들어진 모델의 성능 평가 위젯 부분
    #위젯의 세번째 부분
   def yoloxresult(self):
       yoloxresult = QGroupBox('Yolox Result')
       YoloxResultLayout = QGridLayout()
       YoloxResultLayout.addWidget(self.select_model_mea(),0,0,1,1)
       YoloxResultLayout.addWidget(self.save_group(),1,0,1,1)
       yoloxresult.setLayout(YoloxResultLayout)
       return yoloxresult


    #Yolox Train Start------------------------------------------------------------------------
   
   #파일 합치기 (asset->train2017 폴더로)
   #위젯 구성 부분
   def add2train2017(self):
       add2train2017 = QGroupBox('asset->train2017')
       add = QPushButton('train2017로 이동 버튼')
       self.add_now = QLabel('실행 전')
       vbox_add = QHBoxLayout()
       vbox_add.addWidget(add)
       add2train2017.setLayout(vbox_add)
       
       add.clicked.connect(self.moveFile2train) #버튼을 누르면 moveFile2train 함수가 실행됨

       return add2train2017

    #파일 합치기 (asset->train2017 폴더로)
    #실제 실행 코드
   def moveFile2train(self):
       self.add_now.setText('실행중입니다')
       os.system('cp YOLOX/assets/* YOLOX/datasets/COCO/train2017/')
       self.add_now.setText('실행 전')

   #labelme2COCO
   #위젯 구성 부분
   def labelme2COCO(self):
       labelme2COCO = QGroupBox('labelme2COCO')
       play = QPushButton('실행')
       vbox_2COCO = QHBoxLayout()
       vbox_2COCO.addWidget(play)
       labelme2COCO.setLayout(vbox_2COCO)
        
       play.clicked.connect(self.playClicked) #버튼을 누르면 playClicked 함수가 실행됨

       return labelme2COCO
       
   #labelme2COCO
   #실제 실행 코드
   def playClicked(self):
       os.system('labelme2coco YOLOX/datasets/COCO/train2017/') #train2017폴더에 있는 img와 json에 대해서 labelme2coco가 실행됨
       os.system('labelme2coco YOLOX/datasets/COCO/val2017/') #val2017폴더에 있는 img와 json에 대해서 labelme2coco가 실행됨
       

    #전이학습 모델, 파라미터 설정에서 입력된 값이 제대로 설정되었으면 모델 부분에서 확인 가능
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
   
  
   #파일을 선택해서 선택한 파일을 보여주는 함수
   def fileopen(self):
       sender = self.sender()

       global folderpath
       folderpath = QFileDialog.getExistingDirectory()

       if folderpath:
           if sender.text() == "annotation":
               self.label1.setText(folderpath)
           if sender.text() == "train2017":
               self.label2.setText(folderpath)
           if sender.text() == "val2017":
               self.label3.setText(folderpath)

    
    #모델을 선택해서 선택한 모델을 보여주는 함수
   def modelopen(self):
       global filename
       filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'ckpt(*.pth)') #ckpt만 선택할 수 있도록 함
       if filename[0]:
           self.backbone.setText(filename[0])   
      
  
    #전이학습 모델, 파라미터 설정 버튼
    #위젯 구성 부분
   def select_transfer_group(self):
       select_transfer_group = QGroupBox('전이학습 모델, 파라미터 설정')

       model = QLabel('model :')
       epoch = QLabel('max epoch :')
       batch_size = QLabel('batch size :')
       learning_rate = QLabel('learning rate :')
 
       self.information = QLabel('Parameter를 입력하고 엔터를 누르면, 값이 설정됩니다.')

       #모델을 선택할 수 있는 dropbox
       self.model_select = QComboBox(self)
       self.model_select.addItem('yolox_s')
       self.model_select.addItem('yolox_m')
       self.model_select.addItem('yolox_l')
       self.model_select.addItem('yolox_x')
       self.epoch_line = QLineEdit(self)
       self.batch_size_line = QLineEdit(self)
       self.learning_rate_line = QLineEdit(self)

       self.model_select.activated[str].connect(self.clicked) #모델이 선택되면 오른쪽의 모델 위젯에서 확인할 수 있도록 함
       self.epoch_line.returnPressed.connect(self.TextFunction1) #epoch line의 값이 입력된 후 enter를 누르면 TextFunction1함수가 실행됨
       self.batch_size_line.returnPressed.connect(self.TextFunction2) #batch size의 값이 입력된 후 enter를 누르면 TextFunction2함수가 실행됨
       self.learning_rate_line.returnPressed.connect(self.TextFunction3) #learninr rate의 값이 입력된 후 enter를 누르면 TextFunction3함수가 실행됨
 
       #위젯 layout을 grid로 설정
       grid2 = QGridLayout()
       grid2.addWidget(model,0,0)
       grid2.addWidget(self.model_select,0,1)
       grid2.addWidget(epoch,1,0)
       grid2.addWidget(self.epoch_line,1,1)
       grid2.addWidget(batch_size,2,0)
       grid2.addWidget(self.batch_size_line,2,1)
       grid2.addWidget(learning_rate,3,0)
       grid2.addWidget(self.learning_rate_line,3,1)
 
       vbox_select_transfer = QVBoxLayout()
       vbox_select_transfer.addWidget(self.information)
       vbox_select_transfer.addLayout(grid2)
       select_transfer_group.setLayout(vbox_select_transfer)
 
       return select_transfer_group

    #모델을 선택할 수 있는 dropbox에서 모델이 선택되면 clicked 함수가 실행됨
   def clicked(self):
       text = str(self.model_select.currentText()) #선택된 모델의 이름이 text에 저장됨 
       self.model2.setText(text)

    #전이학습 모델, 파라미터 위젯 부분에서 설정된 값들을 보여주는 위젯
   def show_model_group(self):
       show_model_group = QGroupBox('모델')
       bc = QLabel('backbone:')
       mo = QLabel('model:')
       ep = QLabel('max epoch:')
       ba = QLabel('batch size:')
       le = QLabel('learning rate:')

       self.backbone2 = QLabel('darknet53')
       self.model2 = QLabel('값')
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

    #사용자가 선택하고 입력한 모델, 파라미터에 맞춰서 코드 다시 수정하기
   def model_practice(self):
        #모델의 파라미터 설정은 file_path에 있는 선택된 모델의 이름을 가지고 있는 파이썬 코드를 수정함으로써 이루어질 수 있음
        #ex) yolox_s 모델 설정 -> YOLOX/exps/default/yolox_s.py를 수정해야 함
        file_path = "YOLOX/exps/default/"

        with open(file_path + self.model2.text() + ".py", "r") as f:
            lines = f.readlines()
        #코드에 있는 모든 deafult 설정 초기화하기
        with open(file_path + self.model2.text() + ".py", "w") as f:
            for line1 in lines:
                line1 = re.sub(r"[a-z]", "", line1)
                line1 = re.sub(r"[A-Z]", "", line1)
                line1 = re.sub(r"[0-9]", "", line1)
                line1 = re.sub(r"[.]", "", line1)
                line1 = re.sub(r"[,]", "", line1)
                line1 = re.sub(r"[-]", "", line1)
                line1 = re.sub(r"[/]", "", line1)
                line1 = re.sub(r"[_]", "", line1)
                line1 = re.sub(r"[()]", "", line1)
                line1 = re.sub(r"[:]", "", line1)
                line1 = re.sub(r"[*]", "", line1)
                line1 = re.sub(r"[!]", "", line1)
                line1 = re.sub(r"[#]", "", line1)
                line1 = re.sub(r"[=]", "", line1)
                line1 = re.sub(r"[\[]", "", line1)
                line1 = re.sub(r"[\]]", "", line1)
                line1 = re.sub(r"[\"]", "", line1)
                line1 = re.sub(r"[\n]", "", line1)
                line1 = re.sub(r"[\t]", "", line1)
                line1 = line1.strip()
                f.write(line1)
            
            #사용자가 입력한 모델과 파라미터를 토대로 코드 쓰기
            data = "import os\n"
            f.write(data)
            data = "from yolox.exp import Exp as MyExp\n"
            f.write(data)
            data = "class Exp(MyExp):\n"
            f.write(data)
            data = "    def __init__(self):\n"
            f.write(data)
            data = "        super(Exp, self).__init__()\n"
            f.write(data)  

            #선택한 모델의 코드를 열어서 수정하기
            if (self.model2.text()+".py" == 'yolox_s.py'):
                data = "        self.depth = 0.33\n        self.width = 0.50\n"
                f.write(data)
            elif (self.model2.text()+".py" == 'yolox_m.py'):
                data = "        self.depth = 0.67\n        self.width = 0.75\n"
                f.write(data)
            elif (self.model2.text()+".py" == 'yolox_l.py'):
                data = "        self.depth = 1.0\n        self.width = 1.0\n"
                f.write(data)
            elif (self.model2.text()+".py" == 'yolox_x.py'):
                data = "        self.depth = 1.33\n        self.width = 1.25\n"
                f.write(data)
            data = "        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(\".\")[0]\n"
            f.write(data)
            data = "        self.num_classes = 12\n"
            f.write(data)
            data = "        self.max_epoch = "
            f.write(data)
            f.write(self.epoch_line.text())
            data = "\n        self.basic_lr_per_img = "
            f.write(data)
            f.write(self.learning_rate_line.text())
            f.write(" / 64.0\n")
            f.write("\n        self.hsv_prob=0.2\n")
            f.write("        self.translate = 0.3\n")
            f.write("        self.mosaic_scale = (0.8,1.6)\n")
            f.write("        self.flip_prob = 0.8\n")
            f.write("        self.mixup_scale = (0.8,2.0)\n")
        """      
        f.close()
        """ 

    #모델, 파라미터까지 입력된 후 train을 시작시키는 위젯
    #위젯 구성 부분
   def practice_save_group(self):
       practice_save_group = QGroupBox('모델 학습 및 저장')
       self.practice = QPushButton(self)
       self.practice.setText('yolox 모델 학습 실행')
       self.practice.clicked.connect(self.model_practice) #practice 버튼을 누르면 사용자가 입력한 토대로 모델의 파라미터를 수정하는 model_practice가 실행 됨
       self.now.setWordWrap(True)

       vbox_practicie_save = QVBoxLayout()
       vbox_practicie_save.addWidget(self.practice)
       practice_save_group.setLayout(vbox_practicie_save)

       self.practice.clicked.connect(self.practiceClicked) #practice 버튼을 누르면 train을 시작하는 practcieClicked 함수가 실행이 됨
       
       return practice_save_group

    #모델, 파라미터까지 입력된 후 train을 시작시키는 위젯
    #실제 train 시작 코드
   def practiceClicked(self):
       text = str(self.model_select.currentText())

       os.system('python3 YOLOX/tools/train.py -f YOLOX/exps/default/'+text+'.py -d 1 -b '+self.batch_size2.text()+' --fp16 -c YOLOX/'+text+'.pth')
    

   #Yolox Train End------------------------------------------------------------------------


   #Yolox Test Start------------------------------------------------------------------------
   
   def fileopen(self):
        sender = self.sender()
        
        global folderpath
        folderpath = QFileDialog.getExistingDirectory()
        
        if folderpath:
            self.label4.setText(folderpath)
    
    #demo를 돌릴 모델을 선택해서 선택한 모델을 보여주는 함수
   def modelopen(self):
        self.filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'ckpt(*.pth)') 
        self.arr = self.filename[0].split('/')
        self.fileName = self.arr[-1]

        if self.filename[0]:
            self.label5.setText(self.fileName)


     #demo를 돌릴 모델 설정 
   def select_model_group(self):
        select_model_group = QGroupBox('적용 모델 설정')
        model = QPushButton(self) 
        model.setText('적용 모델 선택')
        model.clicked.connect(self.modelopen) #model 버튼을 누르면 demo를 돌릴 모델을 선택하는 modelopen 함수가 실행된다
        self.label5 = QLabel('적용 모델')

        vbox_select_model = QVBoxLayout()
        vbox_select_model.addWidget(model)
        vbox_select_model.addWidget(self.label5)
        select_model_group.setLayout(vbox_select_model)

        return select_model_group

    #모델 demo 실행
   def model_ver_group(self):
        model_ver_group = QGroupBox('demo 실행')
        ver = QPushButton(self)
        ver.setText('실행')

        vbox_ver = QHBoxLayout()
        vbox_ver.addWidget(ver)
        model_ver_group.setLayout(vbox_ver)

        ver.clicked.connect(self.demoPlay) #버튼을 누르면 demoPlay 함수가 실행 됨

        return model_ver_group

    #모델 demo 실행
    #실제 실행 코드
   def demoPlay(self):
        #assets에 있는 이미지들을 선택한 모델로 demo를 돌린 결과가 img와 json으로 분리가 되서 json폴더의 json이 assets 폴더로 이동하는 labelme.py 코드 사용
        os.system('python YOLOX/tools/labelme.py image -n yolox-s -c YOLOX_outputs/yolox_s/'+self.fileName+' --path YOLOX/assets/ --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]')
        os.system('cp json/* YOLOX/assets/') #json 폴더에 생성된 assets 이미지의 json파일을 YOLOX/assets로 이동시키기
       

   #Yolox Test End------------------------------------------------------------------------


   #Yolox Result Start------------------------------------------------------------------------
    #성능 측정할 모델을 선택해서 선택한 모델을 보여주는 함수
   def modelopen1(self):        
        self.filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'ckpt(*.pth)') 
        self.arr = self.filename[0].split('/')
        self.fileName = self.arr[-1]

        if self.filename[0]:
            self.label6.setText(self.fileName)

    
  
     #성능 측정 모델 설정 
   def select_model_mea(self):
        select_model_mea = QGroupBox('성능 측정 모델 설정')
        model_mea = QPushButton(self) 
        model_mea.setText('성능 측정 모델 선택')
        model_mea.clicked.connect(self.modelopen1) #model_mea 버튼이 눌리면 성능측정을 위한 모델을 선택할 수 있는 modelopen1 함수가 실행된다
        self.label6 = QLabel('성능 측정 모델')

        vbox_model_mea = QVBoxLayout()
        vbox_model_mea.addWidget(model_mea)
        vbox_model_mea.addWidget(self.label6)
        select_model_mea.setLayout(vbox_model_mea)

        return select_model_mea


    #모델 성능의 결과 저장
    #위젯 구성 부분
   def save_group(self):
        save_group = QGroupBox('성능 출력')
        save = QPushButton(self)
        save.setText('실행 후 결과 출력')
        self.output = QTextBrowser()

        vbox_save = QVBoxLayout()
        vbox_save.addWidget(save)
        save_group.setLayout(vbox_save)

        save.clicked.connect(self.showOutput) #save 버튼이 눌리면 성능 측정의 모델의 결과를 txt로 저장하고 저장된 결과를 dialog로 불러와주는 showOutput 함수가 실행됨

        return save_group

    #모델 성능의 결과 저장
    #실제 실행 코드
    #성능 측정의 모델의 결과를 terminal에서 가져와 txt로 저장하고, 저장된 결과를 dialog로 불러와줌
   def showOutput(self):
        #모델의 성능을 측정해주는 eval 코드 실행
        os.system('python -m yolox.tools.eval -n yolox-s -c YOLOX_outputs/yolox_s/'+self.label6.text()+' -b 64 -d 0 --conf 0.001 [--fp16] [--fuse]')
    
        self.dialog = QDialog() #모델의 성능 측정 결과를 띄울 dialog 창 생성
        self.tb = QTextBrowser()
        lay = QHBoxLayout()
        lay.addWidget(self.tb)
        self.dialog.setLayout(lay)

        f = open('YOLOX/eval.txt','r', encoding='utf-8') #모델의 성능 측정 결과를 저장할 eval.txt 파일
        with f:
            data = f.read()
            self.tb.setText(data)

        self.dialog.setWindowTitle('성능출력창')
        self.dialog.resize(500,500)
        self.dialog.show()

    #Yolox Result End------------------------------------------------------------------------
