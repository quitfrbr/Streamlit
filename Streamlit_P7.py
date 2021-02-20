import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Simulateur de prêt</p>', unsafe_allow_html=True)
# train_exp = pd.read_csv('train_exp.csv')
test_exp = pd.read_csv('test_exp.csv')
test_col_name = pd.read_csv('test_col_name.csv')
train_col_name = pd.read_csv('train_col_name.csv')
target = pd.read_csv('target.csv')
with open('voisin.txt', 'r') as file:
    dic_voisin = json.load(file)


ident = st.selectbox('identifiant', test_exp['SK_ID_CURR'])

ind=test_col_name[test_col_name['SK_ID_CURR']==ident].index

st.write(test_col_name.loc[ind,["CNT_CHILDREN","AMT_INCOME_TOTAL","CODE_GENDER","DAYS_BIRTH"]])


load=pickle.load(open('finalized_model.sav', 'rb'))
prediction = load.predict(test_exp.iloc[:,:-1])


if pd.DataFrame(prediction).loc[ind].values==0:
    st.write("--> Prêt attribué")
else:
    st.write("--> Prêt refusé")
    
ind0=target[target["TARGET"]==0].index
ind1=target[target["TARGET"]==1].index

var = st.selectbox('var', ["tout le dataset","voisin"])

for i in train_col_name.loc[ind,"EXT_SOURCE_2"]:
    ligne=i

if var=="voisin":
    voisin_0=list()
    no_val=list()
    for i in ind0:
        #print(i)
        if len(dic_voisin[str(ind[0])])==0:
            no_val.append("pas0")
            #break
        if len(dic_voisin[str(ind[0])])!=0:
            for j in dic_voisin[str(ind[0])]:
                if i==j:
                    voisin_0.append(i)
        ind0=voisin_0

    voisin_1=list()
    for i in ind1:
        if len(dic_voisin[str(ind[0])])==0:
            no_val.append("pas1")
            #break
        if len(dic_voisin[str(ind[0])])!=0:
            for j in dic_voisin[str(ind[0])]:
                #print(j)
                if i==j:
                    voisin_1.append(i)
        ind1=voisin_1
    if len(no_val)==2:
        st.write("Il n'y a pas de voisin donc voici la comparaison avec tous les autres individus")
        ind0=target[target["TARGET"]==0].index
        ind1=target[target["TARGET"]==1].index
        
a=plt.figure(figsize=(6,3), dpi= 5)

a=sns.kdeplot(train_col_name.loc[ind0,"EXT_SOURCE_2"], shade=True, color="g", label="Prets remboursés", alpha=.5)
a=sns.kdeplot(train_col_name.loc[ind1,"EXT_SOURCE_2"], shade=True, color="deeppink", label="Défauts de payement", alpha=.5)
a.axvline(ligne, color='k', linestyle='--',label=ident)
a.legend(fontsize='medium', title_fontsize='10')

a.title.set_size(15)
a.xaxis.label.set_size(12)
a.yaxis.label.set_size(12)
#a.legend()


plt.show()
st.pyplot(plt)

b=plt.figure(figsize=(6,3), dpi= 5)
for i in train_col_name.loc[ind,"EXT_SOURCE_3"]:
    ligne=i


b=sns.kdeplot(train_col_name.loc[ind0,"EXT_SOURCE_3"], shade=True, color="g", label="Prets remboursés", alpha=.5)
b=sns.kdeplot(train_col_name.loc[ind1,"EXT_SOURCE_3"], shade=True, color="deeppink", label="Défauts de payement", alpha=.5)
b.axvline(ligne, color='k', linestyle='--',label=ident)
b.legend(fontsize='medium', title_fontsize='5')

b.title.set_size(10)
b.xaxis.label.set_size(8)
b.yaxis.label.set_size(10)

plt.show()
st.pyplot(plt)

c=plt.figure(figsize=(6,3), dpi= 5)

for i in train_col_name.loc[ind,"DAYS_BIRTH"]:
    ligne=i

c=sns.kdeplot(train_col_name.loc[ind0,"DAYS_BIRTH"], shade=True, color="g", label="Prets remboursés", alpha=.5)
c=sns.kdeplot(train_col_name.loc[ind1,"DAYS_BIRTH"], shade=True, color="deeppink", label="Défauts de payement", alpha=.5)
c.axvline(ligne, color='k', linestyle='--',label=ident)
c.legend(fontsize='medium', title_fontsize='5')

c.title.set_size(10)
c.xaxis.label.set_size(8)
c.yaxis.label.set_size(10)

plt.show()
st.pyplot(plt)

