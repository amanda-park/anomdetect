import pandas as pd

def logic_to_numeric(data):
    if isinstance(data, pd.DataFrame):
        data = data.applymap(lambda x: 1 if x==True else (0 if x==False else x))
    else:
        data = data.to_frame()
        data = data.applymap(lambda x: 1 if x==True else (0 if x==False else x))
    return data

def num_den_to_ratio(s,numerator,denominator):
    if isinstance(s, pd.DataFrame):
        s['ratio'] = s[numerator] / s[denominator]
    else:
        s = s.to_frame()
        s['ratio'] = s[numerator] / s[denominator]
        
    s = s['ratio']
    return s

if __name__ == '__main__':
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    num = [10,40,30,20,10,50,60,50,40,30,20,60,50,40,30,20,40]
    den = [110,430,290,210,120,510,590,530,410,310,190,650,510,420,310,220,421]
    
    data = pd.DataFrame({"Numerator":num,"Denominator":den})
    
    print(num_den_to_ratio(data,"Numerator","Denominator"))
    
    print(sum([1/2,1/3,1/6]))
    
def series_div(x,y):
    j=0
    for i in y:
        if i == 0:
            y[j] = 1
            x[j] = 0
        j+=1
    return x / y
    
    