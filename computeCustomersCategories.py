# Importation des librairies
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.externals import joblib


'''
Class permettant de prédiction le segment de clients à partir des données d'achat.
Le résultat sera enregistré dans un fichier excel dans le répertoire data
'''
class CustomersSegmentation() :

	categories_label = {0 :'Standard', 1 : 'Fidele', 2 : 'HauteValeur', 3 : 'APotentiel', 4:'FaibleValeur' }

	'''
	Initialisation de la classe et construction de la structure clients à partir du fichier contenant les données de commandes.
	@param data_file : contient les données d'achat
	@param current_date : date du jour qui va servir de base pour les calculs des historiques d'achat.
	'''
	def __init__(self, data_file, current_date) :
		self.classifier = joblib.load('./data/CustomersGBClassifier.pkl')
		data = pd.read_excel(data_file)
		self.customers_data = self._createCustomersStructure(data, current_date)
		self.customers_X_scaled = self._transformAsQuantileMatrix(self.customers_data.drop(['CustomerID'], axis=1))


	'''
	Traitement sur les données pour les mettre au format attendu et supprimer les lignes avec identifiant client vide ou dupliquée.
	'''
	def _ordersDataPreProcessing(self, data) :
		data = data[pd.notnull(data['CustomerID'])]
		data=data.drop_duplicates()
		data['TotalPrice'] = data['Quantity']*data['UnitPrice']
		data['Canceled']=data['Quantity'].apply(lambda q: 0 if q > 0 else 1)
		data['Discount']=data['StockCode'].apply(lambda s: 1 if s == 'D' else 0)
		data['Promotion']=data['UnitPrice'].apply(lambda p: 1 if p == 0 else 0)
		data['UK']=data['Country'].apply(lambda c: 1 if c == "United Kingdom" else 0)
		special_codes=data[data['StockCode'].str.contains('^[a-zA-Z]+', regex=True, na=False)]['StockCode'].unique()
		mask = data['StockCode'].isin(special_codes) & (data['StockCode'] != 'D')
		data = data[~mask]
		return data
	'''
	Création de la structure de données clients. 
	@param now : date du jour et qui va servir de base notamment pour calculer le nombre de jour depuis la dernière commande
	du client.
	'''
	def _createCustomersStructure(self, data, now):
	    data = self._ordersDataPreProcessing(data)
	    
	    temp = data.groupby('CustomerID').agg({'InvoiceNo': lambda x: x.nunique()})
	    c_data=temp.rename(columns={'InvoiceNo': 'NbOrders'})
	    c_data.reset_index(drop = False, inplace = True)
	    
	    temp = data.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
	    temp = temp.groupby(by=['CustomerID'])['TotalPrice'].agg(['sum','mean'])
	    temp=temp.rename(columns={'sum':'TotalSpent', 'mean':'AverageSpent'})
	    temp.reset_index(drop = False, inplace = True)
	    c_data=pd.merge(c_data, temp, on='CustomerID', how='outer')
	    
	    temp = data[data.Quantity > 0].groupby(by=['CustomerID','InvoiceNo'], as_index=False)['TotalPrice'].sum()
	    temp = temp.groupby(by=['CustomerID'])['TotalPrice'].agg(['min','max'])
	    temp=temp.rename(columns={'min':'MinSpent','max':'MaxSpent'})
	    temp.reset_index(drop = False, inplace = True)
	    c_data=pd.merge(c_data, temp, on='CustomerID', how='outer')
	    
	    temp = data.groupby(by=['CustomerID'], as_index=False)['Quantity'].agg(['sum'])
	    temp=temp.rename(columns={'sum': 'TotalQuantity'})
	    temp.reset_index(drop = False, inplace = True)
	    c_data=pd.merge(c_data, temp, on='CustomerID', how='outer')
	    
	    temp = data.groupby(by=['CustomerID'], as_index=False)['Canceled'].agg(['sum'])
	    temp=temp.rename(columns={'sum': 'NbCanceled'})
	    temp.reset_index(drop = False, inplace = True)
	    c_data=pd.merge(c_data, temp, on='CustomerID', how='outer')
	    
	    temp = data.groupby(by=['CustomerID'], as_index=False)['Discount'].agg(['sum'])
	    temp=temp.rename(columns={'sum': 'NbDiscount'})
	    temp.reset_index(drop = False, inplace = True)
	    c_data=pd.merge(c_data, temp, on='CustomerID', how='outer')
	    
	    temp = data.groupby(by=['CustomerID'], as_index=False)['Promotion'].agg(['sum'])
	    temp=temp.rename(columns={'sum': 'NbPromo'})
	    temp.reset_index(drop = False, inplace = True)
	    c_data=pd.merge(c_data, temp, on='CustomerID', how='outer')
	    
	    temp = data.groupby(by=['CustomerID'], as_index=False).agg({'UK': lambda x: 1 if (x.all()>0) else 0})
	    c_data=pd.merge(c_data, temp, on='CustomerID', how='outer')
	    
	    temp = data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (now - x.max()).days})
	    temp=temp.rename(columns={'InvoiceDate': 'LastPurchase'})
	    temp.reset_index(drop = False, inplace = True)
	    c_data=pd.merge(c_data, temp, on='CustomerID', how='outer')
	    return c_data

	'''
	méthode permettant d'attribuer un score (entre 1 et 4) à une valeur en fonction d'une colonne en fonction du quantile 
	auquel il appartient.
	1 premier quantile, ..., 4 dernier quantile.
	''' 
	def _getScore(self, x,p,d):
	    if x <= d[p][0.25]:
	        return 1
	    elif x <= d[p][0.50]:
	        return 2
	    elif x <= d[p][0.75]: 
	        return 3
	    else:
	        return 4
	'''
	Calcul score pour les données dont la valorisation est plus importante si la valeur est faible
	Ex : LastPurchase. Plus la valeur est petite, plus l'achat est récent donc à une valorisation importante pour nous.
	'''
	def _getRScore(self, x,p,d):
	    if x <= d[p][0.25]:
	        return 1
	    elif x <= d[p][0.50]:
	        return 2
	    elif x <= d[p][0.75]: 
	        return 3
	    else:
	        return 4

	'''
	Transforme les données en score de quantile et retourne une matrice
	'''
	def _transformAsQuantileMatrix(self, X) :
	    quantiles = X.quantile(q=[0.25,0.5,0.75])
	    quantiles = quantiles.to_dict()
	    X_std=X.copy()
	    for column in X:
	        if (column == 'LastPurchase') :
	            X_std[column] = X[column].apply(self._getRScore, args=(column,quantiles,))
	        else :
	            X_std[column] = X[column].apply(self._getScore, args=(column,quantiles,))
	    return X_std.as_matrix()
	

	'''
		retourne le label (le sens) du cluster
	'''
	def _get_categoryDesc(self, cat) :
		return self.categories_label[cat]


	'''
		Utilisation du modèle de classification pour prédire le segment clients.
	'''
	def _predictCustomersCategories(self) :
		categories = self.classifier.predict(self.customers_X_scaled)
		self.customers_data['Category'] = categories
		self.customers_data['CategoryDesc'] = self.customers_data['Category'].apply(self._get_categoryDesc)

	
	'''
		Sauvegarde du résultat dans le fichier excel categories_results dans le répertoire data
	'''
	def storeCustomerCategories(self) :
		self._predictCustomersCategories()
		self.customers_data[['CustomerID', 'Category', 'CategoryDesc']].to_excel('data/categories_results.xlsx', index=False)


# Le fichier contenant les données d'achat
input_file = 'data/Online_Retail_Exple.xlsx'
# La date cible (date du jour si on a des données de la veille sinon prendre la date du dernier enregistrement +1 )
now_d = dt.datetime(2011, 12, 10)
segmentation = CustomersSegmentation(input_file,now_d)
segmentation.storeCustomerCategories()

