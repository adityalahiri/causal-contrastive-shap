from src.adult import Adult
from src.compass import Compass
from src.helpers import *
#suppress warnings
import warnings
warnings.filterwarnings("ignore")

df=load_compass_data()
original_df,df = label_enocde(df)
clf = train_model(df,kind="compass")

compass_ob = Compass(clf,df,original_df)
idx_1,idx_2,all_shap = compass_ob.run_pair()
compass_ob.plot(idx_1,idx_2,all_shap)


df=load_adult_income_dataset()
original_df,df = label_enocde(df)
clf = train_model(df,kind="adult")

adult_ob = Adult(clf,df,original_df)
idx_1,idx_2,all_shap = adult_ob.run_pair()
adult_ob.plot(idx_1,idx_2,all_shap)