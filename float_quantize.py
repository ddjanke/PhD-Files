
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats

def percent_error(x,y, absolute=True):
    if absolute:
        return torch.abs((x - y)/x)
    else:
        return x/torch.abs(x) * ((x - y)/x)

def closest_power_of_2(x, mode=None):
    x = x.data.clone().detach()
    sign = - 2 * (x < 0) + 1
    abs_x = torch.abs(x)
    #abs_x[abs_x < 2**-11] = 2**-11
    if mode:
        if mode == "ciel":
            return sign * 2**torch.ceil(torch.log2(abs_x))
        elif mode == "floor":
            return sign * 2**torch.floor(torch.log2(abs_x))
    else:
        #return sign * 2**torch.round(torch.log2(abs_x))
        lower, upper = 2**torch.floor(torch.log2(abs_x)), 2**torch.ceil(torch.log2(abs_x))
        result = lower.clone().detach()
        
        #error = percent_error(abs_x, lower), percent_error(abs_x, upper)
        error = percent_error(abs_x, lower)
        #result[error[0] > error[1]] = upper[error[0] > error[1]]
        result[torch.abs(error) > 1/3] = upper[torch.abs(error) > 1/3]
        return sign * result
    
def closest_quantization(n, N = 3):
    n = n.data.clone().detach()
    sign = - 2 * (n < 0) + 1
    abs_n = torch.abs(n)
    n = torch.abs(n)
    
    if N == 1:
        return sign * closest_power_of_2(n)
    
    n2 = 0
    for i in range(N-1):
        x = closest_power_of_2(n, mode="floor")
        n = n - x
        n2 += x
        n[n < 2**-11] = 2**-11
    n2 += closest_power_of_2(n)
    return sign * n2

def quantized_abs_error(n, N=2):
    x = closest_quantization(n,N)
    qae = torch.sum(torch.abs(n-x))
    return qae
    
if __name__ == "__main__":
    
    mpl.rcParams['font.family'] = "Times New Roman"
    
    x = torch.tensor(np.sort((np.random.rand(1000000) - 0.5) * 2 * 32))
    x[torch.abs(x) < 2**-11] = 2**-11
    y = closest_quantization(x,N=2)
    yi = torch.round(x)
    err = percent_error(x,y, absolute=False)
    erri = percent_error(x,yi, absolute=False)
    
    
    plt.figure(figsize=(3,3))
    plt.plot(x,y)
    plt.plot(x,x)
    plt.yticks([-32, -16, -8, -4, 0, 4, 8, 16, 32])
    plt.xticks([-32, -16, 0, 16, 32])
    plt.grid(True, axis='y')
    plt.ylabel("Quantized Value")
    plt.xlabel("Actual Value")
    figname = "QuantizedValues.png"
    #plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # Calculate the PDF for quantization
    ev = torch.tensor(np.linspace(-1,1,1000)/3)
    
    #fy = 4/3/(1-ev)**2
    #'''
    fy = ev.clone().detach()
    fy[fy > 0]  = (1/(1-fy)**2)[fy > 0]
    fy[fy <= 0] = (2/(1-fy)**2)[fy <= 0]
    #'''
    fy = (fy + torch.flip(fy, [0])) / 2
    
    # Plot histogram with calculated PDF
    plt.figure(figsize=(3,3))
    plt.hist(err, 400, density=True)
    #plt.hist(erri, 400, density=True)
    a = np.linspace(-0.3, 0.3, 100)
    b = scipy.stats.norm.pdf(a,0,0.08)
    plt.plot(a,b, color='coral')
    #plt.plot(ev,fy, color='red')
    #plt.xlim((-0.5,0.5))
    #plt.ylim((0,20))
    plt.ylabel("PDF $f_Y(y)$")
    plt.xlabel("% Error")
    plt.xticks([-0.4, -0.2, 0, 0.2, 0.4], labels=["$-40$","$-20$","$0$","$20$","$40$"])
    plt.xlim((-0.46, 0.46))
    figname = "QuantizeErrorPDF2bit.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # plot absolute error vs x
    #err = percent_error(x,y,True)
    #plt.scatter(x,err, s=0.25)
    #plt.show()
    
