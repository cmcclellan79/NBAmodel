#import statmeents for basic ml and data analysis work
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
#imports for the nba_api data sets that tracks game logs 
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

 
def get_player_id(name):
    # get players ID with the dataset if player is not found by full name in the data it will return a not found error message if it is found it shall return the player list index 
    player_list = [p for p in players.get_players() if p["full_name"] == name]
    if not player_list:
        print(f"[!] Player not found: {name}")
        return None
    return player_list[0]["id"]

def fetch_player_data(name, position, season="2023"): # function that pulls from the 2023 season getting the player id found within the api.
    player_id = get_player_id(name)
    if not player_id:
        return pd.DataFrame()
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        gamelog["PLAYER_NAME"] = name
        gamelog["POSITION"] = position
        return gamelog
    except Exception as e:
        print(f"[!] Error fetching data for {name}: {e}")
        return pd.DataFrame()

def build_dataset(player_dict, season="2023"): # Builds out the data set after fetching the player data 
    all_data = []
    for name, position in player_dict.items():
        df = fetch_player_data(name, position, season)
        if not df.empty:
            all_data.append(df)
        time.sleep(1.5)
    return pd.concat(all_data, ignore_index=True)

def train_model(df, features, target): #Function trains the model based on the data given for linear regression 
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    return model

def plot_feature_importance(model, features): # Plots the important features for the regression
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_df = feature_df.sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_df)
    plt.title("Feature Importance for PTS Prediction")
    plt.tight_layout()
    plt.show()

def train_model_for_position(df, features, target, position_label): #Function trains the model based on the data given for linear regression based on position 
    """Train model for a specific position and show results."""
    df_pos = df[df["POSITION"] == position_label]
    
    if df_pos.empty:
        print(f"[!] No data for position: {position_label}")
        return None
    
    X = df_pos[features]
    y = df_pos[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"\n📊 Results for {position_label}:")
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    
    return model



player_dict = { #demo players in the model 

    "Stephen Curry": "PG",
    "Klay Thompson": "SG",
    "LeBron James": "SF",
    "Draymond Green": "PF",
    "Nikola Jokic": "C"
}

features = ["MIN", "FGA", "FG3A", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV"]
target = "PTS"

nba_df = build_dataset(player_dict)
if not nba_df.empty:
    model = train_model(nba_df, features, target)
    plot_feature_importance(model, features)
