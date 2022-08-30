class DataCleaning:
    def __init__(self):
        pass
    #Function to drop columns with zero values
    def drop_rows(self, df,col1,col2):
        df_new= df.drop(df[(df[col1] == 0) & (df[col2] == 0)].index)
        return df_new
    # Calculating skewness of each column 
    def calculating_skewness(self,df):
        df.skew(axis='index', skipna=True)
   #Function to drop columns missing descriptive data that is unpredictable
    def drop_missing(self,df,col):
        df_dropped= df.dropna(subset=[col])
        return df_dropped.shape

    