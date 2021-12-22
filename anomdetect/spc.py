
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

class SPC:
    """"This class creates necessary functions for Statistical Process Control (SPC) charts. 
    
    Currently, only p-charts are developed due to their ubiquity with Quality Control in healthcare.
    
    The general workflow of the class is as follows:
        1) Fit data to a method (currently the option is p-chart) on a certain date range. This will represent the baseline data.
        2) Predict new anomalies based on the baseline fit. This can be done with predict().
    
    """
    def __init__(self,df):
        self._df = df
        self._chart = None
        self._numerator = None
        self._denominator = None
        self._n_df = None
        #p_chart specific params
        self._pbar = None
        
    def p_chart(self,numerator,denominator):
        """Runs the calculations necessary to create a p-chart on baseline data.
        
        :param str numerator: Required. The name of the numerator in the data frame fed into the class.
        
        :param str denominator: Required. The name of the denominator in the data frame fed into the class.
        
        :returns: DataFrame column specifying binary yes/no violations.
        """
        self._numerator = numerator
        self._denominator = denominator
        self._df['Values'] = self._df[numerator]/self._df[denominator]
        self._pbar = statistics.mean(self._df['Values'])
        pse = np.sqrt((self._pbar*(1-self._pbar))/(self._df[self._denominator]))
        df = self._df.copy()
        df['Violation'] = 0
        for index in df['Values'].index:
            ub = self._pbar+3*pse.at[index]
            lb = self._pbar-3*pse.at[index]
            if df.at[index,'Values'] > ub or df.at[index,'Values'] < lb:
                df.at[index, "Violation"] = 1
            else:
                df.at[index,"Violation"] = 0
        self._df['Violation'] = df['Violation']
        self._chart = 'p_chart()'
        return df['Violation']
    
    def predict(self,df):
        """Predicts anomalies depending on the baseline fit. 
        
        :param DataFrame df: DataFrame to predict new violations on.
        
        :returns: DataFrame column specifying binary yes/no violations.
        """
        
        self._n_df = df
        if self._chart == 'p_chart()':
            df['Values'] = df[self._numerator]/df[self._denominator]
            pse = np.sqrt((self._pbar*(1-self._pbar))/df[self._denominator])
            df['Violation'] = 0
            for index in df['Values'].index:
                ub = self._pbar+3*pse.at[index]
                lb = self._pbar-3*pse.at[index]
                if df.at[index,'Values'] > ub or df.at[index,'Values'] < lb:
                    df.at[index, "Violation"] = 1
                else:
                    df.at[index,"Violation"] = 0
        return df['Violation']
    
    def bounds(self,predict=False):
        """Creates bound for chosen control chart. 
        
        :param bool predict: Default False. Set True if the predict() function has been used.
        
        :returns: DataFrame with the following columns:
            * Values
            * UCL
            * LCL
            * Violation
        """
        if predict:
            df = self._n_df.copy()
        else:
            df = self._df.copy()
        if self._chart == 'p_chart()':
            # Plot p-chart
            df['UCL'] = self._pbar+3*(np.sqrt((self._pbar*(1-self._pbar))/(df[self._denominator])))
            df['LCL'] = self._pbar-3*(np.sqrt((self._pbar*(1-self._pbar))/(df[self._denominator])))
            df = df[['Values','UCL','LCL','Violation']]
            return df
        else:
            f = "no SPC chart was specified"
            return f
        
if __name__ == '__main__':
    num = [10,40,30,20,10,50,60,50,40,30,20,60,50,40,30,20,40]
    den = [110,430,290,210,120,510,590,530,410,310,190,650,510,420,310,220,421]
    n_num = [10,40,30,20,10,50,60,50,40,30,20,60,50,40,30,20,40,9999,9999,9999,9999,9999,9999]
    n_den = [110,430,290,210,120,510,590,530,410,310,190,650,510,420,310,220,421,10000,10000,10000,10000,10000,10000]
    
    df = pd.DataFrame({"Numerator":num,"Denominator":den})
    n_df = pd.DataFrame({"Numerator":n_num,"Denominator":n_den})
    
    spc = SPC(df)
    spc.p_chart("Numerator","Denominator")
    print(spc.predict(n_df))
    print(spc.bounds())
    print(spc.bounds(predict=True))
    
    