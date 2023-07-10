#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:\prac\ml\hotel_bookings.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.drop(['agent','company'],axis=1,inplace=True)


# In[7]:


df['country'].fillna(df['country'].value_counts().index[0],inplace=True)


# In[8]:


df.isnull().sum()


# In[9]:


filter1=(df['children']==0) & (df['adults']==0) & (df['babies']==0)


# In[10]:


df[filter1]


# In[11]:


data=df[~filter1]


# In[12]:


data.shape


# In[13]:


data['is_canceled'].unique()


# In[16]:


data[data['is_canceled']==0]['country'].value_counts()/75011


# In[15]:


len(data[data['is_canceled']==0])


# In[17]:


country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns=['country','no_of_guests']
country_wise_data


# In[18]:


get_ipython().system('pip install plotly')


# In[19]:


get_ipython().system('pip install chart_studio')


# In[20]:


import plotly
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs ,init_notebook_mode ,plot ,iplot
init_notebook_mode(connected=True)


# In[21]:


import plotly.express as px


# In[22]:


map_guest=px.choropleth(country_wise_data,
             locations=country_wise_data['country'],
             color=country_wise_data['no_of_guests'],
              hover_name=country_wise_data['country'],
              title='home country of guests'
             )


# In[23]:


map_guest.show()


# In[24]:


data2=data[data['is_canceled']==0]


# In[25]:


data2.columns


# In[26]:


plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',y='adr' ,hue='hotel',data=data2)

plt.title('Price of room types per night and person')
plt.xlabel('room types')
plt.ylabel('price( EUR)')


# In[27]:


data['hotel'].unique()


# In[28]:


data_resort=data[(data['hotel']=='Resort Hotel') & (data['is_canceled']==0)]
data_city = data[(data['hotel']=='City Hotel') & (data['is_canceled']==0)]


# In[29]:


rush_resort=data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns=['month','no_of_guests']
rush_resort


# In[30]:


rush_city=data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns=['month','no_of_guests']
rush_city


# In[31]:


final_rush=rush_resort.merge(rush_city,on='month')


# In[32]:


final_rush.columns=['month','no_of_guests_in_resort','no_of_guests_city']


# In[33]:


final_rush


# In[34]:


get_ipython().system('pip install sorted-months-weekdays')

get_ipython().system('pip install sort_dataframeby_monthorweek')


# In[ ]:





# In[35]:


import sort_dataframeby_monthorweek as sd


# In[36]:


final_rush=sd.Sort_Dataframeby_Month(final_rush,'month')


# In[37]:


final_rush.columns


# In[38]:


px.line(data_frame=final_rush,x='month',y=['no_of_guests_in_resort', 'no_of_guests_city'])


# In[39]:


data=sd.Sort_Dataframeby_Month(data,'arrival_date_month')


# In[40]:


sns.barplot(x='arrival_date_month',y='adr',data=data ,hue='is_canceled')
plt.xticks(rotation='vertical')
plt.show()


# In[41]:


plt.figure(figsize=(12,8))
sns.boxplot(x='arrival_date_month',y='adr',data=data ,hue='is_canceled')
plt.xticks(rotation='vertical')


plt.ylim(0,800)
plt.show()


# In[42]:


data.columns


# In[43]:


pd.crosstab(index=data['stays_in_weekend_nights'],columns=data['stays_in_week_nights'])


# In[44]:


def week_function(row):
    feature1='stays_in_weekend_nights'
    feature2='stays_in_week_nights'
    
    if row[feature2]==0 and row[feature1] >0 :
        return 'stay_just_weekend'
    
    elif row[feature2]>0 and row[feature1] ==0 :
        return 'stay_just_weekdays'
    
    elif row[feature2]>0 and row[feature1] >0 :
        return 'stay_both_weekdays_weekends'
    
    else:
        return 'undefined_data'


# In[45]:


data2['weekend_or_weekday']=data2.apply(week_function,axis=1)


# In[46]:


data2.head()


# In[47]:


data2['weekend_or_weekday'].value_counts()


# In[48]:


type(sd)


# In[49]:


data2=sd.Sort_Dataframeby_Month(data2,'arrival_date_month')


# In[50]:


data2.groupby(['arrival_date_month','weekend_or_weekday']).size()


# In[51]:


group_data=data2.groupby(['arrival_date_month','weekend_or_weekday']).size().unstack().reset_index()


# In[52]:


sorted_data=sd.Sort_Dataframeby_Month(group_data,'arrival_date_month')


# In[53]:


sorted_data.set_index('arrival_date_month',inplace=True)


# In[54]:


sorted_data


# In[55]:


sorted_data.plot(kind='bar',stacked=True,figsize=(15,10))


# In[56]:


data2.columns


# In[57]:


def family(row):
    if (row['adults']>0) &  (row['children']>0 or row['babies']>0) :
        return 1
    else:
        return 0


# In[58]:


data['is_family']=data.apply(family,axis=1)


# In[64]:


data['total_customer'] = data['adults'] + data['babies'] + data['children']


# In[65]:


data['total_nights']=data['stays_in_week_nights'] + data['stays_in_weekend_nights']


# In[66]:


data.head(3)


# In[67]:


data.columns


# In[68]:


data['deposit_type'].unique()


# In[70]:


dict1={'No Deposit':0, 'Non Refund':1, 'Refundable': 0}


# In[71]:


data['deposit_given']=data['deposit_type'].map(dict1)


# In[72]:


data.columns


# In[73]:


data.drop(columns=['adults', 'children', 'babies', 'deposit_type'],axis=1,inplace=True)


# In[74]:


data.columns


# In[75]:


data.head(6)


# In[76]:


data.dtypes


# In[77]:


cate_features=[col for col in data.columns if data[col].dtype=='object']


# In[78]:


num_features=[col for col in data.columns if data[col].dtype!='object']


# In[79]:


data_cat=data[cate_features]


# In[80]:


data.groupby(['hotel'])['is_canceled'].mean().to_dict()


# In[81]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[82]:


data_cat['cancellation']=data['is_canceled']


# In[83]:


data_cat.head()


# In[84]:


cols=data_cat.columns


# In[85]:


cols=cols[0:-1]


# In[86]:


cols


# In[87]:


for col in cols:
    dict2=data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict2)


# In[88]:


data_cat.head(3)


# In[89]:


data[num_features]


# In[90]:


dataframe=pd.concat([data_cat,data[num_features]],axis=1)


# In[91]:


dataframe.columns


# In[92]:


dataframe.drop(['cancellation'],axis=1,inplace=True)


# In[93]:


dataframe.head(3)


# In[94]:


sns.distplot(dataframe['lead_time'])


# In[95]:


def handle_outlier(col):
    dataframe[col]=np.log1p(dataframe[col])


# In[96]:


handle_outlier('lead_time')


# In[97]:


sns.distplot(dataframe['lead_time'])


# In[98]:


sns.distplot(dataframe['adr'])


# In[99]:


dataframe[dataframe['adr']<0]


# In[100]:


handle_outlier('adr')


# In[101]:


dataframe['adr'].isnull().sum()


# In[102]:


sns.distplot(dataframe['adr'].dropna())


# In[103]:


sns.FacetGrid(data,hue='is_canceled',xlim=(0,500)).map(sns.kdeplot,'lead_time',shade=True).add_legend()


# In[104]:


corr=dataframe.corr()


# In[105]:


corr


# In[106]:


corr['is_canceled'].sort_values(ascending=False)


# In[107]:


corr['is_canceled'].sort_values(ascending=False).index


# In[108]:


features_to_drop=['reservation_status', 'reservation_status_date','arrival_date_year',
       'arrival_date_week_number', 'stays_in_weekend_nights',
       'arrival_date_day_of_month']


# In[109]:


dataframe.drop(features_to_drop,axis=1,inplace=True)


# In[110]:


dataframe.shape


# In[111]:


dataframe.head(2)


# In[112]:


dataframe.isnull().sum()


# In[113]:


dataframe.dropna(inplace=True)


# In[114]:


x=dataframe.drop('is_canceled',axis=1)


# In[115]:


y=dataframe['is_canceled']


# In[116]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[117]:


feature_sel_model=SelectFromModel(Lasso(alpha=0.005))


# In[118]:


feature_sel_model.fit(x,y)


# In[119]:


feature_sel_model.get_support()


# In[120]:


cols=x.columns


# In[121]:


cols


# In[123]:


selected_feature=cols[feature_sel_model.get_support()]
selected_feature


# In[124]:


y


# In[125]:


from sklearn.model_selection import train_test_split


# In[126]:


X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.25)


# In[127]:


X_train.shape


# In[128]:


from sklearn.linear_model import LogisticRegression


# In[131]:


logreg=LogisticRegression()
logreg.fit(X_train,y_train)


# In[132]:


pred=logreg.predict(X_test)


# In[133]:


pred


# In[134]:


from sklearn.metrics import confusion_matrix


# In[135]:


confusion_matrix(y_test,pred)


# In[136]:


from sklearn.metrics import accuracy_score


# In[137]:


accuracy_score(y_test,pred)


# In[138]:


from sklearn.model_selection import cross_val_score


# In[139]:


score=cross_val_score(logreg,x,y,cv=10)


# In[140]:


score


# In[141]:


score.mean()


# In[142]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[143]:


models=[]

models.append(('LogisticRegression',LogisticRegression()))
models.append(('Naive_bayes',GaussianNB()))
models.append(('Random Forest',RandomForestClassifier()))
models.append(('Decision_tree',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))


# In[144]:


for name,model in models:
    print(name)
    model.fit(X_train,y_train)
    
    predictions=model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(predictions,y_test)
    print(cm)
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(predictions,y_test)
    print(acc)
    print('\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




