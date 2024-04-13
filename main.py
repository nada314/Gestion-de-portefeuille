import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO
import time
import datetime

# Définition de la classe de l'agent Q-learning
class QLearningAgent:
    def __init__(self, state_size, action_size, initial_balance=5000, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.balance = initial_balance 
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Taux d'exploration
        self.Q = np.zeros((state_size, action_size))  # Table des valeurs Q
    

    def choose_action(self, state, balance):
        if balance <= 0:  # Si la balance est à 0, éliminer l'option d'achat
            # Choisir aléatoirement entre vendre (1) et conserver (2) si exploration
            if np.random.rand() <= self.epsilon:
                return np.random.choice([1, 2])
            # Sinon, choisir l'action avec la meilleure valeur Q, excluant l'achat
            else:
                return np.argmax(self.Q[state, 1:]) + 1  # Ajoute 1 car on exclut l'option 0 (acheter)
        else:
            # Politique epsilon-greedy normale
            if np.random.rand() <= self.epsilon:
                return np.random.choice(self.action_size)  # Exploration
            return np.argmax(self.Q[state])  # Exploitation

    def update_q(self, state, action, reward, next_state): 
        # Mise à jour de la table Q selon l'équation de Bellman
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action] # Calcul de l'Erreur de Différence Temporelle (TD Error)
        self.Q[state][action] += self.alpha * td_error


def download_data():
    period1 = int(time.mktime(datetime.datetime(2020, 3, 25, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    interval = '1d' # 1d, 1m
    ticker = 'ALNOV.PA'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    query_string = query_string.replace(" ", "%20")
    df = pd.read_csv(query_string)
    print(df)
    df.to_csv('AAPL.csv')
    return df

# Fonction pour préparer les données et définir les états
def prepare_data(df):
    #Cette ligne calcule la variation quotidienne du prix de clôture ('Close') 
    #en pourcentage et stocke les résultats dans une nouvelle colonne appelée 'Price Change'. 
    #La fonction pct_change() de pandas calcule la variation en pourcentage entre chaque élément 
    #et son prédécesseur dans la colonne, ce qui donne une mesure de la performance quotidienne de l'actif.
    df['Price Change'] = df['Close'].pct_change()
    '''
    Cette ligne crée un ensemble de bacs (ou intervalles) qui sont utilisés pour discrétiser (ou catégoriser) 
    les variations de prix continues en un nombre fixe d'états. np.linspace génère un tableau de 10 points 
    linéairement espacés qui couvrent l'ensemble de la gamme de la variation du prix, du minimum au maximum. 
    Ces points définissent les frontières des bacs.
    '''
    bins = np.linspace(df['Price Change'].min(), df['Price Change'].max(), 20) # 19 ségments -> 20 Points
    '''
    Cette ligne de code utilise la fonction np.digitize de la bibliothèque numpy pour attribuer à chaque
    valeur de la colonne 'Price Change' du dataframe (df) un indice correspondant à l'intervalle dans 
    lequel elle se trouve dans le tableau bins.
    '''
    df['State'] = np.digitize(df['Price Change'], bins)
    df['Price'] = df['Close']  # Crée une copie de la colonne 'Close' sous le nom 'Price'
    return df, len(bins) + 1


# Fonction pour simuler l'entraînement de l'agent sur les données historiques
def train_agent(agent, data):
    #pour la comparer à la fin avec la balance finale et voir si on a un gain ou non
    initial_balance = agent.balance
    num_shares = 0
    total_purchase_cost = 0  # Coût total d'achat des actions
    
    #parcours de la dataframe, index contient l'index de la ligne et row contient un tableau des valeurs de la ligne
    for index, row in data.iterrows():
        #data.shape[0] renvoie le nombre de lignes dans le DataFrame 'data'. C'est la première valeur renvoyée par 
        #l'attribut shape, qui est un tuple contenant le nombre de lignes et le nombre de colonnes du DataFrame.
        if index == data.shape[0] - 1:
            break

        #récupération de la valeur de state et price de la ligne courante
        current_state = row['State']
        current_price = row['Price']
        action = agent.choose_action(current_state, agent.balance)

        if action == 0 and agent.balance >= current_price:  # Achat possible
            #le nombre d'action qu'on peut acheter
            num_shares_to_buy = int(agent.balance // current_price)      #retourne un entier
            purchase_cost = num_shares_to_buy * current_price
            #mise à jour de la balance
            agent.balance -= purchase_cost
            total_purchase_cost += purchase_cost  # Mise à jour du coût total d'achat
            num_shares += num_shares_to_buy
            reward = 0  # Pas de profit immédiat lors de l'achat
        elif action == 1 and num_shares > 0:  # Vente possible
            sale_proceeds = num_shares * current_price  # Produit de la vente
            profit = sale_proceeds - total_purchase_cost  # Profit = prix de vente - prix d'achat
            agent.balance += sale_proceeds
            num_shares = 0
            total_purchase_cost = 0  # Réinitialiser après la vente
            reward = profit  # La récompense est le profit réalisé
        elif action == 2:  # Conserver
            reward = -1
        else:  # Action d'achat avec solde insuffisant
            reward = -1
        '''pour éviter un débordement d'index (index out of range), la valeur maximale est définie comme étant 
        le nombre total de lignes moins 2. Cela garantit que même si nous sommes sur l'avant-dernière ligne, 
        la prochaine ligne à laquelle nous accédons est valide.'''
        next_state_index = min(index + 1, data.shape[0] - 2)
        # La méthode iloc est utilisée pour accéder à la ligne par son indice dans le DataFrame.
        next_state = data.iloc[next_state_index]['State']
        print(reward)
        agent.update_q(current_state, action, reward, next_state)

    
    final_balance = agent.balance + num_shares * current_price
    total_profit = final_balance - initial_balance
    print(f"Profit total: {total_profit} Euros")



# Téléchargement et préparation des données

df = download_data()
df_prepared, state_size = prepare_data(df)

# Initialisation de l'agent Q-learning
action_size = 3  # Buy, Hold, Sell
agent = QLearningAgent(state_size, action_size)

# Entraînement de l'agent avec les données historiques
train_agent(agent, df_prepared)

# Affichage de la table Q pour inspection
print("Table Q après entraînement :")
print(agent.Q)