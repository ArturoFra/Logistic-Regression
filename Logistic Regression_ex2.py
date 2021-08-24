#!/usr/bin/env python
# coding: utf-8

# # Regresión Logística para predicciones bancarias

# In[5]:


import pandas as pd
import numpy as np


# In[38]:


data = pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/bank/bank.csv", sep=";")


# In[ ]:





# In[ ]:





# In[10]:


data.head()


# In[ ]:





# In[12]:


data["y"].value_counts()


# In[13]:


data.shape


# In[ ]:





# In[14]:


data.columns.values


# In[ ]:





# In[15]:


data["y"] = (data["y"]=="yes").astype(int)


# In[16]:


data["y"]


# In[ ]:





# In[ ]:





# In[17]:


data["education"].unique()


# In[ ]:





# In[19]:


data["education"] = np.where(data["education"]=="basic.4y", "Basic", data["education"])
data["education"] = np.where(data["education"]=="basic.6y", "Basic", data["education"])
data["education"] = np.where(data["education"]=="basic.9y", "Basic", data["education"])
data["education"] = np.where(data["education"]=="high.school", "High School", data["education"])
data["education"] = np.where(data["education"]=="professional.course", "Professional Course", data["education"])
data["education"] = np.where(data["education"]=="university.degree", "University Degree", data["education"])
data["education"] = np.where(data["education"]=="illiterate", "Illiterate", data["education"])
data["education"] = np.where(data["education"]=="unknown", "Unknown", data["education"])


# In[ ]:





# In[18]:


data["y"].value_counts()


# In[ ]:





# In[ ]:





# In[20]:


data.groupby("y").mean()


# In[ ]:





# In[21]:


data.groupby("education").mean()


# In[ ]:





# In[22]:


import matplotlib.pyplot as plt


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.education, data.y).plot(kind ="bar")
plt.title("Frecuencia de compra en funcion del nivel de educación")
plt.xlabel("Nivel de educación")
plt.ylabel("Compras")


# In[23]:


table = pd.crosstab(data.marital, data.y)


# In[46]:


table.div(table.sum(1).astype(float), axis = 0).plot(kind ="bar", stacked = True)
plt.title("Diagrama apilado de estado civil con compras")
plt.xlabel("Civil status")
plt.ylabel("Porporción de clientes")


# In[ ]:





# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.day_of_week, data.y).plot(kind ="bar")
plt.title("Frecuencia de compra en funcion del día de la semana")
plt.xlabel("Día de la semana")
plt.ylabel("Compras")


# In[ ]:





# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(data.month, data.y).plot(kind ="bar")
plt.title("Frecuencia de compra en funcion del mes")
plt.xlabel("mes del año")
plt.ylabel("Compras")


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
data.age.hist()
plt.title("Histograma de ventas en función de la edad")
plt.xlabel("Edad")
plt.ylabel("Cliente")


# In[51]:


pd.crosstab(data.age, data.y).plot(kind ="bar")


# In[52]:


pd.crosstab(data.poutcome, data.y).plot(kind ="bar")


# # Conversión de las variables categóricas en dummy

# In[39]:


categories = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
print(data)
for category in categories:
    cat_list = "cat" + "_" + category
    cat_dummies = pd.get_dummies(data[category], prefix = cat_list)
    data_new = data.join(cat_dummies)
    data = data_new


# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


data.columns.values.tolist()


# In[ ]:





# In[30]:


data_vars = data.columns.values.tolist()


# In[31]:


to_keep = [v for v in data_vars if v not in categories]


# In[32]:


bank_data=data[to_keep]
bank_data.head()


# In[ ]:





# In[80]:


bank_data_var = bank_data.columns.values.tolist()
Y=["y"]
X=[v for v in bank_data_var if v not in Y]


# In[82]:


bank_data["y"]


# In[ ]:





# In[ ]:





# # Selección de rasgos en el modelo

# In[34]:


n=12


# In[35]:


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[36]:


lr = LogisticRegression()


# In[44]:


rfe = RFE(lr, n)
rfe = rfe.fit(bank_data[X], bank_data[Y].values.ravel())


# In[47]:


print(rfe.support_)


# In[49]:


print(rfe.ranking_)


# In[51]:


bank_data_var


# In[57]:


z=zip(bank_data_var, rfe.support_, rfe.ranking_)


# In[58]:


list(z)


# In[61]:


cols=["previous","euribor3m","cat_job_management","cat_job_student","cat_job_technician","cat_month_aug",
      "cat_month_dec","cat_month_jul","cat_month_jun","cat_month_mar","cat_day_of_week_wed","cat_poutcome_nonexistent"]


# In[83]:


X = bank_data[cols]
Y=(bank_data["y"] =="yes").astype(int)


# In[84]:


X


# In[ ]:





# In[85]:


Y


# In[ ]:





# # Implementación del modelo con statsmodel.api en Python

# In[64]:


import statsmodels.api as sm


# In[86]:


logit_model = sm.Logit(Y,X)


# In[87]:


result = logit_model.fit()


# In[88]:


result.summary()


# # Implemmentacion con scikit-learn

# In[89]:


from sklearn import linear_model


# In[90]:


logit_model=linear_model.LogisticRegression()
logit_model.fit(X,Y)


# In[91]:


logit_model.score(X,Y)


# In[92]:


Y.mean()


# In[93]:


pd.DataFrame(list(zip(X.columns, np.transpose(logit_model.coef_))))


# # Validación del modelo

# In[96]:


from sklearn.model_selection import train_test_split


# In[106]:


from IPython.display import display, Math, Latex 


# In[98]:


X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.3, random_state=0) 


# In[99]:


lm = linear_model.LogisticRegression()


# In[100]:


lm.fit(X_train,Y_train)


# In[102]:


probs = lm.predict_proba(X_test)


# In[103]:


probs


# In[133]:


predictions = lm.predict(X_test)


# In[134]:


predictions


# In[110]:


display(Math(r"Y_p = \begin{cases}0 & si\ p\leq0.5\\1&si\ p>0.5\end{cases}"))


# In[112]:


display(Math(r"\varepsilon\in (0,1), Y_p = \begin{cases}0 & si\ p\leq\varepsilon\\1&si\ p>\varepsilon\end{cases}"))


# In[115]:


prob = probs[:, 1]
prob_df = pd.DataFrame(prob)
treshold = 0.1
prob_df["Prediction"]=np.where(prob_df[0]>treshold, 1, 0)
prob_df.head()


# In[118]:


pd.crosstab(prob_df.Prediction, columns = "count")


# In[120]:


390/len(prob_df)*100


# In[121]:


treshold=0.15
prob_df["Prediction"]=np.where(prob_df[0]>treshold, 1, 0)
prob_df.head()


# In[122]:


pd.crosstab(prob_df.Prediction, columns = "count")


# In[124]:


365/len(prob_df)*100


# In[125]:


treshold=0.05
prob_df["Prediction"]=np.where(prob_df[0]>treshold, 1, 0)
prob_df.head()


# In[126]:


pd.crosstab(prob_df.Prediction, columns = "count")


# In[127]:


769/len(prob_df)*100


# In[128]:


from sklearn import metrics


# In[136]:


metrics.accuracy_score(Y_test, predictions)


# # Validación cruzada

# In[137]:


from sklearn.model_selection import cross_val_score


# In[147]:


scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring = "accuracy", cv =34)


# In[ ]:





# In[148]:


scores


# In[ ]:





# In[149]:


scores.mean()


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




