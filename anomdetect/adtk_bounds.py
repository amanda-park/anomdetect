
import copy
import utils_ad
import math
import pandas as pd

import sys
sys.setrecursionlimit(10000)


class ADTK_Bounds:
    
    """This class creates the mathematical bounds that the ADTK package uses in order to determine if a point is an anomaly or not.
    """
    
    def __init__(self,adtk_obj, s):
        self._s = s
        self._adtk_obj = adtk_obj
        
    def univ_bounds(self,delta=.0001):
        """Calculate the univariate bounds for ADTK algorithms. 
        
        :param float delta: Required; default .0001. Offset to bounds.
        
        :returns: Pandas DataFrame with univariate bound violations.
        """
        
        main = self._adtk_obj.predict(self._s)
        main = utils_ad.logic_to_numeric(main)
        main.columns = ['anomaly_logic']
        upper = [0]*len(self._s)
        lower = [0]*len(self._s)
        i = 0
        for index in self._s.index:
            adtk_obj = copy.deepcopy(self._adtk_obj)
            up_temp_s = self._s.copy()
            up_temp_s = pd.DataFrame({'temp_s':up_temp_s})
            up_temp_s.set_index(self._s.index)
            up = 0
            down_temp_s = self._s.copy()
            down_temp_s = pd.DataFrame({'temp_s':down_temp_s})
            down_temp_s.set_index(self._s.index)
            down = 0
            if main.at[index,'anomaly_logic'] == 0:
                pass
            else:
                up_temp_s.at[index,'temp_s'] = up_temp_s.median()
                down_temp_s.at[index,'temp_s'] = down_temp_s.median()
                adtk_obj.fit_detect(up_temp_s)
            while up == 0:
                up_temp_s.at[index,'temp_s'] = up_temp_s.at[index,'temp_s'] + delta
                up_anoms = adtk_obj.predict(up_temp_s)
                up_anoms = utils_ad.logic_to_numeric(up_anoms)
                up_anoms.columns = ['anomaly_logic']
                up = up_anoms.at[index,'anomaly_logic']
                if math.isnan(up):
                    up = 1
            while down ==0:
                down_temp_s.at[index,'temp_s'] = down_temp_s.at[index,'temp_s'] - delta
                down_anoms = adtk_obj.predict(down_temp_s)
                down_anoms = utils_ad.logic_to_numeric(down_anoms)
                down_anoms.columns = ['anomaly_logic']
                down = down_anoms.at[index,'anomaly_logic']
                if math.isnan(down):
                    down = 1
            upper[i] = up_temp_s.at[index,'temp_s']-delta
            lower[i] = down_temp_s.at[index,'temp_s']+delta
            i+=1
        out = pd.DataFrame()
        out['Values'] = self._s.copy()
        out['UCL'] = upper
        out['LCL'] = lower
        out['Violation'] = utils_ad.logic_to_numeric(self._adtk_obj.predict(self._s))
        return out
        
    def ratio_bounds(self,numerator,denominator,delta=1):
        """Calculate the ratio bounds for ADTK algorithms. 
        
        :param float delta: Required; default 1. Offset to bounds.
        
        :returns: Pandas DataFrame with ratio bound violations.
        """
        main = self._adtk_obj.predict(self._s)
        main = utils_ad.logic_to_numeric(main)
        main.columns = ['anomaly_logic']
        upper = [0]*len(self._s)
        lower = [0]*len(self._s)
        i = 0
        for index in self._s.index:
            adtk_obj = copy.deepcopy(self._adtk_obj)
            up_temp_s = self._s.copy()
            up = 0
            down_temp_s = self._s.copy()
            down = 0
            if main.at[index,'anomaly_logic'] == 0:
                pass
            else:
                up_temp_s.at[index,numerator] = (up_temp_s.at[index,denominator]*up_temp_s[numerator].median())/up_temp_s[denominator].median()
                down_temp_s.at[index,numerator] = (down_temp_s.at[index,denominator]*down_temp_s[numerator].median())/down_temp_s[denominator].median()
                adtk_obj.fit_detect(up_temp_s)
            while up == 0:
                up_temp_s.at[index,numerator] = up_temp_s.at[index,numerator] + delta
                up_anoms = adtk_obj.predict(up_temp_s)
                up_anoms = utils_ad.logic_to_numeric(up_anoms)
                up_anoms.columns = ['anomaly_logic']
                up = up_anoms.at[index,'anomaly_logic']

            while down ==0:
                down_temp_s.at[index,numerator] = down_temp_s.at[index,numerator] - delta
                down_anoms = adtk_obj.predict(down_temp_s)
                down_anoms = utils_ad.logic_to_numeric(down_anoms)
                down_anoms.columns = ['anomaly_logic']
                down = down_anoms.at[index,'anomaly_logic']
            upper[i] = (up_temp_s.at[index,numerator]-delta)/up_temp_s.at[index,denominator]
            lower[i] = (down_temp_s.at[index,numerator]+delta)/down_temp_s.at[index,denominator]
            i+=1
        out = pd.DataFrame()
        out['Values'] = self._s[numerator]/self._s[denominator]
        out['UCL'] = upper
        out['LCL'] = lower
        out['Violation'] = utils_ad.logic_to_numeric(self._adtk_obj.predict(self._s))
        return out
        
        
if __name__ == '__main__':
    import pandas as pd
    import adtk.detector as ad
    from adtk.data import validate_series
    from sklearn.linear_model import LinearRegression
    import datetime
    #build data
    start_date = datetime.date(2019, 9, 30)
    number_of_days = 20
    d = []
    for day in range(number_of_days):
        a_date = (start_date + datetime.timedelta(days = day)).isoformat()
        d.append(a_date)    
    
    num = [10,10,11,9,10,11,10,10,12,10,10,10,9,11,10,11,10,11,10,11] #91
    den = [110,430,290,210,110,430,290,210,110,430,290,210,110,430,290,210,110,430,290,210]
    df = pd.DataFrame({"Numerator":num,"Denominator":den,"Date":d})
    #validate data
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    s = validate_series(df)
    pca_ad = ad.PcaAD(k=1)
    pca_ad.fit_detect(s)
    bounds = ADTK_Bounds(adtk_obj=pca_ad,s=s)
    print(bounds.ratio_bounds("Numerator","Denominator"))
    
    s = utils_ad.num_den_to_ratio(s,"Numerator","Denominator")
    quantile_ad = ad.QuantileAD(high=0.99, low=0.01)
    quantile_ad.fit_detect(s)
    bounds = ADTK_Bounds(adtk_obj=quantile_ad,s=s)
    print(bounds.univ_bounds())
    