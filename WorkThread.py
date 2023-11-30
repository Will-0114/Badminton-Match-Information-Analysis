
from PyQt5.QtCore import QThread,pyqtSignal




class WorkThread(QThread):    
    signals = pyqtSignal(str)# 定義信號對象,傳遞值爲str類型，使用int，可以爲int類型
    
    def __init__(self,content):#向線程中傳遞參數，以便在run方法中使用
        super(WorkThread, self).__init__()       
        self.content = content
        
    def __del__(self):
        self.wait()
        
    def run(self):#重寫run方法
        print(self.content)
        self.signals.emit(self.content)#發射信號，str類型數據