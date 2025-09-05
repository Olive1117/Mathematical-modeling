import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.signal import  find_peaks
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

angel__alpha1=10*np.pi/180
angel__alpha2=15*np.pi/180

n__air=1.0

#导入数据
def find_data(file_path):
    df=pd.read_excel(file_path)
    wave_num=df.iloc[:,0].valves#波数
    ref_per = df.iloc[:, 1].valves#反射率

    sorted_idx=np.argsort(wave_num)
    wave_num=wave_num[sorted_idx]
    ref_per=ref_per[sorted_idx]
    return  wave_num,ref_per

wave_10,ref_10=find_data(('附件1.xlsx'))
wave_15,ref_15=find_data(('附件2.xlsx'))
#提取干涉明纹波数
def find__peaks(wave_num,ref_per):
    peaks,_=find_peaks(ref_per,height=0.5)
    peak_wave=wave_num[peaks]
    return peak_wave

peak_wave_10=find__peaks(wave_10,ref_10)
peak_wave_15=find__peaks(wave_15,ref_15)

def snell_law(angle1,n1,n2):
    angle2=np.arcsin(np.sin(angle1)*n1/n2)
    return angle2

def d_long(m,line_count,n_outer,angle2):#计算厚度公式
    wave_number=line_count/100
    wavelength=1/wave_number
    d=(wavelength*m)/(2*n_outer*np.cos(angle2))
    return d

def get_m(peak_wave):
    num1,num2



