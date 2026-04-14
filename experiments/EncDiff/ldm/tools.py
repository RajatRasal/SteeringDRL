import pandas as pd
import os

class Record():
    def __init__(self, plot_image, every_step_num=200) -> None:
        self.plot_image = plot_image
        self.every_step_num = every_step_num

    def init(self, path):
        if self.plot_image:
            out_dir = os.path.join(path,"record")
            if not os.path.exists(out_dir):
                os.mkdir(os.path.join(path,"record"))
        self.csv_file = path + f"/record.csv"
        self.data = None
        self.start = False

    def log_dict(self,log_dict, it):
        if it % self.every_step_num == 0:
            if self.start:
                # data = pd.read_csv(self.csv_file)
                self.data = pd.concat([self.data, pd.DataFrame([dict([(k,v.item() if type(v) is not int else v) for k,v in log_dict.items()])])])
            else:
                self.start  = True
                self.data = pd.DataFrame([dict([(k,v.item() if type(v) is not int else v) for k,v in log_dict.items()])])
            # print(self.data)
            self.data.to_csv(self.csv_file, index=False)

    
        


    
