import pandas as pd
#import pickle as pkl

import adtk.detector as ad
from adtk.data import validate_series

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from spc import SPC
import utils_ad
from adtk_bounds import ADTK_Bounds

class Anomaly:

    """Class that allows for detecting anomalies through a variety of machine learning and control chart methodologies. Inspiration is from the ADTK library in Python, which can be found here - https://adtk.readthedocs.io/en/stable/
    """    

    def __init__(self, df, var_type = "univariate", numerator=None, denominator=None):
        self.df = df
        self.var_type = var_type #univariate, ratio
        self.date_col = None
        self.s = None
        self.numerator = numerator
        self.denominator = denominator
        if var_type == "ratio":
            self.median = utils_ad.series_div(self.df[self.numerator],self.df[self.denominator]).median()
        elif var_type == "univariate":
            self.median = self.df[self.df.columns].median()[0]

        self.method = [] #Stores string of the AD Method used
        self.proc = [] #Stores class call to AD Method

        self.bounds = [] #Stores bounds from AD Method
        
    def validate(self, date_col):
        """Validates inputs to the class are the approprite type.
        
        :param str chart: Required. Name of Date column in data frame.
        """
        self.date_col = date_col
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.set_index(self.date_col)
        self.s = validate_series(self.df)
        if isinstance(self.s, pd.DataFrame):
            self.s = self.s.squeeze() #Squeeze down into a numpy array
        else:
            self.s = self.s
        
    def spc(self, chart, test=True):
        """Runs an SPC chart based on the chosen chart type.
        
        :param str chart: Required. Current options: "p".
        
        :param bool test: Default True. Returns chart bounds for a given metric in order to validate its use and appropriateness.
        
        :returns: None.
        """
        if chart == "p":
            spc = SPC(self.df)
            spc.p_chart(self.numerator, self.denominator)
            if test:
                return spc.bounds()
            else:
                self.method.append('spc()')
                self.proc.append(spc)
                self.bounds.append(spc.bounds())
                return "Added: spc()"
        
    def ad_quantile(self,high=0.99, low=0.01, delta=.0001, test=True):
        """Fits an Anomaly Detection Quantile chart.
        
        :param float high: Required, default .99. Must be float between 0 and 1. Determines violation range for upper bound.
        
        :param float low: Required, default .01. Must be float between 0 and 1. Determines violation range for lower bound.
        
        :param float delta: Required, default .0001. Offset value for creating bounds.
        
        :param bool test: Default True. Returns chart bounds for a given metric in order to validate its use and appropriateness.
        
        :returns: Bounds if test = True, message validating ad_quantile() is added to class parameters if test = False.
        """
        quantile_ad = ad.QuantileAD(high=high, low=low)
        if self.var_type == "ratio":
            s = utils_ad.num_den_to_ratio(self.s,self.numerator,self.denominator)
            quantile_ad.fit_detect(s)
            bounds = ADTK_Bounds(adtk_obj=quantile_ad,s=s)
            bounds = bounds.univ_bounds(delta = delta) #Yes, univariate bounds are used here and not ratio 
            #Ratio var_type for ad_quantile treats the ratio as if it's univariate
            #Plots univariate bounds on z for z = numerator/denominator
        elif self.var_type == "univariate":
            quantile_ad.fit_detect(self.s)
            bounds = ADTK_Bounds(adtk_obj=quantile_ad,s=self.s)
            bounds = bounds.univ_bounds(delta=delta)
        else:
            return "No other var_types built at this time"
        if test:
            return bounds
        else:
            self.method.append('ad_quantile()')
            self.proc.append(quantile_ad)
            self.bounds.append(bounds)
            return "Added: ad_quantile()"

    def ad_seasonal(self,c=3.0, side="both", test=True):
        """Fits an Anomaly Detection Seasonal chart.
        
        :param float c: Default 3.0. Factor used to determine the bound of normal range based on historical interquartile range.
        
        :param str side: Default "both".
        - If "both", to detect anomalous positive and negative residuals;
        - If "positive", to only detect anomalous positive residuals;
        - If "negative", to only detect anomalous negative residuals.
        
        :param bool test: Default True. Returns chart bounds for a given metric in order to validate its use and appropriateness.
        
        :returns: Bounds if test = True, message validating ad_seasonal() is added to class parameters if test = False.
        """
        seasonal_ad = ad.SeasonalAD(c=c, side=side)
        if self.var_type == "ratio":
            s = utils_ad.num_den_to_ratio(self.s,self.numerator,self.denominator)
            seasonal_ad.fit_detect(s)
            bounds = ADTK_Bounds(adtk_obj=seasonal_ad,s=s)
            bounds = bounds.ratio_bounds(self.numerator,self.denominator)
        elif self.var_type == "univariate":
            seasonal_ad.fit_detect(self.s)
            bounds = ADTK_Bounds(adtk_obj=seasonal_ad,s=self.s)
            bounds = bounds.univ_bounds()
        else:
            return "No other var_types built at this time"
        if test:
            return bounds
        else:
            self.method.append('ad_seasonal()')
            self.proc.append(seasonal_ad)
            self.bounds.append(bounds)
            return "Added: ad_seasonal()"
        
    def ad_kmeans_high_dim(self, n_clusters=3, test=True):
        """Fits an Anomaly Detection K-Means Chart, which detects anomalies based on clustering of historical data.
        
        :param int n_clusters: Number of clusters to form. Default is 3.
        
        :param bool test: Default True. Returns chart bounds for a given metric in order to validate its use and appropriateness.
        
        :returns: Bounds if test = True, message validating ad_kmeans_high_dim() is added to class parameters if test = False.
        """
        min_cluster_detector = ad.MinClusterDetector(KMeans(n_clusters=n_clusters))
        min_cluster_detector.fit_detect(self.s)
        if self.var_type == "ratio":
            bounds = ADTK_Bounds(adtk_obj=min_cluster_detector,s=self.s)
            bounds = bounds.ratio_bounds(self.numerator,self.denominator)
        elif self.var_type == "univariate":
            return "Method does not support var_type: univariate"
        else:
            return "No other var_types built at this time"
        if test:
            return bounds
        else:
            self.method.append('ad_kmeans_high_dim()')
            self.proc.append(min_cluster_detector)
            self.bounds.append(bounds)
            return "Added: ad_kmeans_high_dim()"
        
    def ad_regression(self, c=3.0, test=True):
        """Fits an Anomaly Detection Regression Chart, which detects anomalies based on a regression relationship.
        
        :param float c: Default 3.0. Factor used to determine the bound of normal range based on historical interquartile range.
        
        :param bool test: Default True. Returns chart bounds for a given metric in order to validate its use and appropriateness.
        
        :returns: Bounds if test = True, message validating ad_regression() is added to class parameters if test = False.
        """
        regression_ad = ad.RegressionAD(regressor=LinearRegression(), target=self.numerator, c=c)
        regression_ad.fit_detect(self.s)
        if self.var_type == 'ratio':
            bounds = ADTK_Bounds(adtk_obj=regression_ad,s=self.s)
            bounds = bounds.ratio_bounds(self.numerator,self.denominator)
        elif self.var_type == "univariate":
            return "Mehtod does not support var_type: univariate"
        else:
            return "No other var_types built at this time"
        if test:
            return bounds
        else:
            self.method.append('ad_regression()')
            self.proc.append(regression_ad)
            self.bounds.append(bounds)
            return "Added: ad_regression()"
            
    def ad_pca(self, k=1, test=True):
        """Fits an Anomaly Detection Principal Component Analysis (PCA) Chart, which performs principal component analysis (PCA) to the multivariate time series (every time point is treated as a point in high-dimensional space), measures reconstruction error at every time point, and identifies a time point as anomalous when the recontruction error is beyond anomalously large.
        
        :param int k: Default 1. Number of principal components to use.
        
        :param bool test: Default True. Returns chart bounds for a given metric in order to validate its use and appropriateness.
        
        :returns: Bounds if test = True, message validating ad_pca() is added to class parameters if test = False.
        """
        pca_ad = ad.PcaAD(k=k)
        pca_ad.fit_detect(self.s)
        if self.var_type == 'ratio':
            bounds = ADTK_Bounds(adtk_obj=pca_ad,s=self.s)
            bounds = bounds.ratio_bounds(self.numerator,self.denominator)
        elif self.var_type == "univariate":
            return "Mehtod does not support var_type: univariate"
        else:
            return "No other var_types built at this time"
        if test:
            return bounds
        else:
            self.method.append('ad_pca()')
            self.proc.append(pca_ad)
            self.bounds.append(bounds)
            return "Added: ad_pca()"
            
    def assemble(self,weights=None):
        """Combine multiple anomaly detection algorithms based on a pre-provided weighting.
        
        :param list weights: Stores a list of weights to assign to each anomaly detection algorithm. Sum of values provided to weights must be equal to 1.
        
        :returns: concatenated DataFrame with combined AD predictions.
        """
        if len(self.bounds) == 1:
            concatenated = self.bounds[0]
            concatenated['Median'] = [self.median]*len(concatenated)
            return concatenated
        elif weights is None:
            weights = [1/len(self.bounds)]*len(self.bounds)
        elif sum(weights) != 1:
            raise "sum of object: weights must be equal to 1"
        else:
            i = 0
            for df in self.bounds:
                df = utils_ad.logic_to_numeric(df)
                df = df.apply(lambda x: x*weights[i])
                self.bounds[i] = df
                i+=1
            concatenated = pd.concat(self.bounds, axis=1)
            concatenated = concatenated.groupby(lambda x:x, axis=1).sum()
        concatenated['Median'] = [self.median]*len(concatenated)
        return concatenated
    
    def new_obs(self,df):
        """Applies previous fit of anomaly detection algorithms to new observations for control charts.
        
        :param DataFrame df: A data frame including new observations to be fit on.
        
        :returns: None.
        """
        self.df = df
        self.validate(self.date_col)
        self.bounds = []
        j = 0
        for i in self.method:
            if i == 'spc()':
                spc = self.proc[j]
                spc.predict(self.s)
                self.bounds.append(spc.bounds(predict=True))
                j+=1
            elif i == 'ad_quantile()':
                quantile_ad = self.proc[j]
                if self.var_type == 'ratio':
                    s = utils_ad.num_den_to_ratio(self.s,self.numerator,self.denominator)
                    bounds = ADTK_Bounds(adtk_obj=quantile_ad,s=s)
                    bounds = bounds.univ_bounds()
                    self.bounds.append(bounds)
                else:
                    print("No other var_types built at this time")
                j+=1
            elif i == 'ad_seasonal()':
                ad_seasonal = self.proc[j]
                if self.var_type == 'ratio':
                    s = utils_ad.num_den_to_ratio(self.s,self.numerator,self.denominator)
                    bounds = ADTK_Bounds(adtk_obj=ad_seasonal,s=s)
                    bounds = bounds.univ_bounds()
                    self.bounds.append(bounds)
                else:
                    print("No other var_types built at this time")
                j+=1
            elif i == 'ad_kmeans_high_dim()':
                min_cluster_detector = self.proc[j]
                if self.var_type == 'ratio':
                    bounds = ADTK_Bounds(adtk_obj=min_cluster_detector,s=self.s)
                    bounds = bounds.ratio_bounds(self.numerator,self.denominator)
                    self.bounds.append(bounds)
                else:
                    print("No other var_types built at this time")
                j+=1
            elif i == 'ad_regression()':
                regression_ad = self.proc[j]
                if self.var_type == 'ratio':
                    bounds = ADTK_Bounds(adtk_obj=regression_ad,s=self.s)
                    bounds = bounds.ratio_bounds(self.numerator,self.denominator)
                    self.bounds.append(bounds)
                else:
                    print("No other var_types built at this time")
                j+=1
            elif i == 'ad_pca()':
                pca_ad = self.proc[j]
                if self.var_type == 'ratio':
                    bounds = ADTK_Bounds(adtk_obj=pca_ad,s=self.s)
                    bounds = bounds.ratio_bounds(self.numerator,self.denominator)
                    self.bounds.append(bounds)
                else:
                    print("No other var_types built at this time")
                j+=1
            else:
                print("no other options ¯\_(ツ)_/¯")
                
                
if __name__ == '__main__':
    import datetime
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    start_date = datetime.date(2019, 9, 30)
    number_of_days = 17
    n_number_of_days = 23
    d = []
    for day in range(number_of_days):
        a_date = (start_date + datetime.timedelta(days = day)).isoformat()
        d.append(a_date)
    n_d = []
    for day in range(n_number_of_days):
        a_date = (start_date + datetime.timedelta(days = day)).isoformat()
        n_d.append(a_date)
    num = [10,40,30,20,10,50,60,50,40,30,20,60,50,40,30,20,40]
    den = [110,430,290,210,120,510,590,530,410,310,190,650,510,420,310,220,421]
    n_num = [10,40,30,20,10,50,60,50,40,30,20,60,50,40,30,20,40,9999,9999,9999,9999,9999,9999]
    n_den = [110,430,290,210,120,510,590,530,410,310,190,650,510,420,310,220,421,10000,10000,10000,10000,10000,10000]
    
    hdvch = pd.DataFrame({"Numerator":num, "Denominator":den, "Date":d})
    n_hdvch = pd.DataFrame({"Numerator":n_num, "Denominator":n_den,"Date":n_d})
    #instantiate anomaly
    ad_hdvch = Anomaly(hdvch,var_type="ratio",numerator="Numerator",denominator="Denominator")
    #validate step is required
    ad_hdvch.validate('Date')
    #add p-chart
    print(ad_hdvch.spc("p",test=False))
    #add ad_regression
    print(ad_hdvch.ad_regression(c=6,test=False))
    #add ad_quantile
    print(ad_hdvch.ad_quantile(test=False))
    #new observations
    ad_hdvch.new_obs(df=n_hdvch)
    #test to see if it works
    print(ad_hdvch.assemble(weights=[1,0,0]))
    #with open('ad_hdvch.pickle', 'wb') as handle:
    #    pkl.dump(ad_hdvch, handle, protocol=pkl.HIGHEST_PROTOCOL)
        
    #with open('ad_hdvch.pickle', 'rb') as handle:
    #    ad_saved = pkl.load(handle)
    

    #print(ad_hdvch.s)