import pandas as pd
import numpy as np
import statsmodels.api as sm
from catboost import CatBoostRegressor, cv, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
import warnings
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns



warnings.filterwarnings("ignore")


def load_excel(excel_path, sheet, skip_rows, header_row):
        try:
            data = pd.read_excel(excel_path, sheet_name=sheet, skiprows=skip_rows, header=header_row)
            print("Dataset")
            return data
        except Exception as e:
            print(f"Nao deu {e}")
            return None


excel_path = r"C:\\Users\\Naja Informatica\\Desktop\\TCC\\database\\tst\\WorldEnergyBalancesHighlights2024.xlsx"
skip_rows = 1
header_row = 0

raw_data = load_excel(excel_path, 3, skip_rows, header_row)


def filter_brazil(raw_data):
    filtered_df = raw_data[raw_data['Country'] == 'Brazil']
    return filtered_df

brasil_df = filter_brazil(raw_data)


def filter_renewable(brasil_df):
    filtered_type = brasil_df[brasil_df['Product'].isin(['Renewables and waste','Renewable sources'])]
    return filtered_type

renew_br_df = filter_renewable(brasil_df)


melted_df = pd.melt(renew_br_df, id_vars=['Country', 'Product', 'Flow', 'NoCountry', 'NoProduct', 'NoFlow'],

                    var_name='Year', value_name='Value')




pd.set_option('mode.chained_assignment', None)

def pivot_flow(melted_df):
    pivoted_df = melted_df.pivot(index='Year', columns='Flow', values='Value')
    return pivoted_df

pivoted_df = pivot_flow(melted_df)


# CABECALHO
years_to_drop = range(1971, 1990)
pivoted_df = pivoted_df.drop(years_to_drop)
pivoted_df.replace("..", 0, inplace=True)
final_df = pivoted_df.astype(float)

print(final_df.shape)

final_df.reset_index(drop=False, inplace=True)
final_df['Year'] = final_df['Year'].astype(str).str.replace('Provisional', '').astype(int)
final_df.set_index('Year', inplace=True)
final_df.index = pd.to_datetime(final_df.index, format='%Y')
final_df = final_df.asfreq('YS')

if not pd.api.types.is_datetime64_any_dtype(final_df.index):
    final_df.index = pd.to_datetime(final_df.index, errors='coerce')
if final_df.index.isna().any():
    print("Cabecalho com na")

####################ETL  ACABOU############################

print("\n Dataset limpo:")
print(final_df.shape)
print(final_df.head)


############SARIMA###########################SARIMA ############SARIMA ##########SARIMA #########

all_test_predictions = []
all_future_forecasts = []



y = final_df['Production (PJ)']

n_splits = 5
p, d, q = 4, 1, 1
P, D, Q, m = 0, 1, 0, 25


def fit_sarima_model(train):
    model = sm.tsa.SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m), suppress_warnings=True)
    return model.fit()


def calculate_rmse_mape(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    epsilon = 1e-10  # Pequeno valor para evitar divisão por zero
    mape = np.mean(np.abs((true_values - predictions) / (true_values + epsilon))) * 100
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    return mse, rmse, mape, mae, r2

def calculate_sarima_metrics(train, test, fitted_model):
    train_predictions = fitted_model.predict(start=train.index[0], end=train.index[-1])
    test_forecast_result = fitted_model.get_forecast(steps=len(test))
    test_predictions = test_forecast_result.predicted_mean

    train_mse, train_rmse, train_mape, train_mae, train_r2 = calculate_rmse_mape(train, train_predictions)
    test_mse, test_rmse, test_mape, test_mae, test_r2 = calculate_rmse_mape(test, test_predictions)

    train_mbd = np.mean(train_predictions - train)
    test_mbd = np.mean(test_predictions - test)

    residuals = fitted_model.resid
    adf_result = adfuller(residuals)
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]

    max_lag = min(10, len(residuals) - 1)
    ljung_box_result = acorr_ljungbox(residuals, lags=range(1, max_lag + 1), return_df=True)
    ljung_box_stat = ljung_box_result['lb_stat'].values[0]
    ljung_box_pvalue = ljung_box_result['lb_pvalue'].values[0]

    metrics = {
        'AIC': fitted_model.aic,
        'BIC': fitted_model.bic,
        'Log-Likelihood': fitted_model.llf,
        'Train MSE': train_mse,
        'Train RMSE': train_rmse,
        'Train MAPE': train_mape,
        'Train MAE': train_mae,
        'Train MBD': train_mbd,
        'Train R2': train_r2,
        'Test MSE': test_mse,
        'Test RMSE': test_rmse,
        'Test MAPE': test_mape,
        'Test MAE': test_mae,
        'Test MBD': test_mbd,
        'Test R2': test_r2,
        'ADF Stat': adf_stat,
        'ADF p-value': adf_pvalue,
        'Ljung-Box Stat': ljung_box_stat,
        'Ljung-Box p-value': ljung_box_pvalue
    }

    return metrics

metrics_list = [
    'AIC', 'BIC', 'Log-Likelihood', 'Train MSE', 'Train RMSE', 'Train MAPE', 'Train MAE', 'Train MBD', 'Train R2',
    'Test MSE', 'Test RMSE', 'Test MAPE', 'Test MAE', 'Test MBD', 'Test R2', 'ADF Stat', 'ADF p-value',
    'Ljung-Box Stat', 'Ljung-Box p-value'
]

metrics_accumulator = {
    'AIC': [], 'BIC': [], 'Log-Likelihood': [], 'Train MSE': [], 'Train RMSE': [], 'Train MAPE': [], 'Train MAE': [],
    'Train MBD': [], 'Train R2': [], 'Test MSE': [], 'Test RMSE': [], 'Test MAPE': [], 'Test MAE': [], 'Test MBD': [],
    'Test R2': [], 'ADF Stat': [], 'ADF p-value': [], 'Ljung-Box Stat': [], 'Ljung-Box p-value': []
}

tscv = TimeSeriesSplit(n_splits=n_splits)

for train_index, test_index in tscv.split(y):
    train, test = y.iloc[train_index], y.iloc[test_index]
    fitted_model = fit_sarima_model(train)

    fold_metrics = calculate_sarima_metrics(train, test, fitted_model)

    for key in metrics_accumulator:
        metrics_accumulator[key].append(fold_metrics[key])

avg_metrics = {key: np.mean(metrics_accumulator[key]) for key in metrics_accumulator}

#######################################################################

train_metrics = [
    avg_metrics.get('AIC'), avg_metrics.get('BIC'), avg_metrics.get('Log-Likelihood'), avg_metrics.get('Train MSE'),
    avg_metrics.get('Train RMSE'), avg_metrics.get('Train MAPE'), avg_metrics.get('Train MAE'), avg_metrics.get('Train MBD'),
    avg_metrics.get('Train R2'), avg_metrics.get('ADF Stat'), avg_metrics.get('ADF p-value'), avg_metrics.get('Ljung-Box Stat'),
    avg_metrics.get('Ljung-Box p-value')
]

test_metrics = [
    None, None, None, avg_metrics.get('Test MSE'), avg_metrics.get('Test RMSE'), avg_metrics.get('Test MAPE'),
    avg_metrics.get('Test MAE'), avg_metrics.get('Test MBD'), avg_metrics.get('Test R2'),
    None, None, None, None
]


max_length = max(len(metrics_list), len(train_metrics), len(test_metrics))


train_metrics += [None] * (max_length - len(train_metrics))
test_metrics += [None] * (max_length - len(test_metrics))
metrics_list += ['Unknown'] * (max_length - len(metrics_list))


combined_ARmetrics_df = pd.DataFrame({
    'Metric': metrics_list,
    'Train': train_metrics,
    'Test': test_metrics
})


print("\nSARIMA Metrics:")
print(combined_ARmetrics_df)

################################ Previsão Futura ############################################


final_model = fit_sarima_model(y)
future_forecast_result = final_model.get_forecast(steps=12)
future_predictions = future_forecast_result.predicted_mean


forecast_years = list(range(2024, 2024 + len(future_predictions)))

future_results_df = pd.DataFrame({
    'Ano': forecast_years,
    'Future SARIMA Forecast': future_predictions
})

print("\nSARIMA Forecasts:")
print(future_results_df)

#######CATBOOST#############CATBOOST#############CATBOOST#############CATBOOST#############CATBOOST

final_df.reset_index(inplace=True)
print(final_df.columns)
final_df['Year'] = final_df['Year'].dt.year
final_df = final_df.sort_values('Year')

final_df.dropna(inplace=True)

def create_lagged_features(df, target_column, lags=3):
    for lag in range(1, lags + 1):
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df.dropna()

lags = 3

final_df = create_lagged_features(final_df, 'Production (PJ)', lags)

features = final_df[['Commercial and public services (PJ)', 'Electricity output (GWh)',
                     'Electricity, CHP and heat plants (PJ)',
                     'Industry (PJ)',
                     'Total energy supply (PJ)', 'Total final consumption (PJ)'] +
                    [f'Production (PJ)_lag_{i}' for i in range(1, lags + 1)]]
target = final_df['Production (PJ)']

data_pool = Pool(features, target)

train_size = int(0.7 * len(final_df))
train_df = final_df[:train_size]
test_df = final_df[train_size:]

train_features = train_df.drop(columns=['Production (PJ)'])
train_target = train_df['Production (PJ)']

test_features = test_df.drop(columns=['Production (PJ)'])
test_target = test_df['Production (PJ)']

train_pool = Pool(train_features, train_target)


params = {
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 3,
    'l2_leaf_reg': 1,
    'one_hot_max_size': 5,
    'colsample_bylevel': 0.8,
    'bagging_temperature': 0.2,
    'random_strength': 1,
    'subsample': 1,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'task_type': 'CPU',
    'verbose': 300,
    'random_state': 42
}

cv_results = cv(
    params=params,
    dtrain=data_pool,
    fold_count=3,
    shuffle=True,
    partition_random_seed=42,
    verbose=False
)

model = CatBoostRegressor(**params)
model.fit(data_pool)

def iterative_forecast(model, initial_features, steps=12):
    forecasted_values = []
    current_features = initial_features.copy()

    for _ in range(steps):
        prediction = model.predict([current_features])[0]
        forecasted_values.append(prediction)
        current_features = current_features[1:] + [prediction]

    return forecasted_values

last_known_values = features.iloc[-1].tolist()
forecast = iterative_forecast(model, last_known_values, steps=12)
forecastCAT_years = list(range(2024, 2024 + len(forecast)))
forecast_df = pd.DataFrame(forecast, index=forecastCAT_years, columns=['Catboost Forecasted Production (PJ)'])
future_results_df = future_results_df.merge(forecast_df, left_on='Ano', right_index=True, how='left')

print("\nPrevisão de 12 passos:", forecast_df)

print("\ndf geral", future_results_df)

# AVALIAÇÃO

train_predictions = model.predict(train_features)
train_mse, train_rmse, train_mape, train_mae, train_r2 = calculate_rmse_mape(train_target, train_predictions)
train_mbd = np.mean(train_predictions - train_target)

print(f'RMSE Treino: {train_rmse}')
print(f'MAE Treino: {train_mae}')
print(f'MAPE Treino: {train_mape}%')
print(f'R² Treino: {train_r2}')
print(f'MBD Treino: {train_mbd}')

test_predictions = model.predict(test_features)
test_mse, test_rmse, test_mape, test_mae, test_r2 = calculate_rmse_mape(test_target, test_predictions)
test_mbd = np.mean(test_predictions - test_target)

print(f'RMSE Teste: {test_rmse}')
print(f'MAE Teste: {test_mae}')
print(f'MAPE Teste: {test_mape}%')
print(f'R² Teste: {test_r2}')
print(f'MBD Teste: {test_mbd}')


combinedCAT_metrics = {
    'CATMetric': ['CATMSE', 'CATRMSE', 'CATMAPE', 'CATMAE', 'CATR2', 'CATMBD'],
    'Train': [train_mse, train_rmse, train_mape, train_mae, train_r2, train_mbd],
    'Test': [test_mse, test_rmse, test_mape, test_mae, test_r2, test_mbd]
}


combinedCAT_metrics_df = pd.DataFrame(combinedCAT_metrics)

print("\n Catboost metrics", combinedCAT_metrics_df)

##################################stacking#######################################


media_modelos = (future_results_df['Catboost Forecasted Production (PJ)'] + future_results_df['Future SARIMA Forecast'])/ 2
future_results_df['Media Modelos'] = media_modelos

print("\n Resultados finais:")
print(future_results_df)
forecastCAT_years = list(range(2024, 2024 + len(forecast)))

#EXPORTS p dataviz

final_df.to_csv(r'C:\Users\Naja Informatica\Desktop\TCC\exports\final_df.csv')
future_results_df.to_csv(r'C:\Users\Naja Informatica\Desktop\TCC\exports\future_results_df.csv')
combined_ARmetrics_df.to_csv(r'C:\Users\Naja Informatica\Desktop\TCC\exports\combinedARmetrics_df.csv')
combinedCAT_metrics_df.to_csv(r'C:\Users\Naja Informatica\Desktop\TCC\exports\combinedCATmetrics_df.csv')


caminhos = [
    r"C:\Users\Naja Informatica\Desktop\TCC\exports\combinedARmetrics_df.csv",
    r"C:\Users\Naja Informatica\Desktop\TCC\exports\combinedCATmetrics_df.csv",
    r"C:\Users\Naja Informatica\Desktop\TCC\exports\final_df.csv",
    r"C:\Users\Naja Informatica\Desktop\TCC\exports\future_results_df.csv"
]


combinedARmetrics_df = pd.read_csv(caminhos[0])
combinedCATmetrics_df = pd.read_csv(caminhos[1])
final_df = pd.read_csv(caminhos[2])
future_results_df = pd.read_csv(caminhos[3])


######etl visualização

print(future_results_df.columns)
print(final_df.columns)

future_results_df = future_results_df.rename(columns={'Ano': 'Year'})

data_1980_2023 = {
    'Year': final_df['Year'],  # Passa a série diretamente
    'production (pj)': final_df['Production (PJ)']  # Passa a série diretamente
}
df_1980_2023 = pd.DataFrame(data_1980_2023)


data_2024_2034 = {
    'Year': future_results_df['Year'],  # Passa a série diretamente
    'Future SARIMA Forecast': future_results_df['Future SARIMA Forecast'],
    'Catboost Forecasted Production (PJ)': future_results_df['Catboost Forecasted Production (PJ)'],
    'Media Modelos': future_results_df['Media Modelos']
}
future_results_df_corrected = pd.DataFrame(data_2024_2034)

full_df = pd.concat([df_1980_2023, future_results_df_corrected], ignore_index=True)


print(full_df)
print(full_df.columns)
print(full_df.shape)
print(full_df)

##########GRAFICOS

sns.set_style("whitegrid")
sns.set_palette("Set2")

plt.figure(figsize=(17, 8))
plt.plot(full_df['Year'], full_df['production (pj)'], label='Ano', color='b', linewidth=7)
plt.title('Produção Histórica dados IEA', fontsize=16)
plt.xlabel('Ano', fontsize=14)
plt.ylabel('Produção (PJ)', fontsize=14)
plt.xlim([min(full_df['Year']), max(full_df['Year'])])
plt.ylim([min(full_df['production (pj)']), max(full_df['production (pj)'])])
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color='#dcdcdc', alpha=0.4)
plt.legend(loc='upper left', fontsize=12)
plt.savefig(r"C:\Users\Naja Informatica\Desktop\TCC\exports\producao_hist.png", dpi=700, bbox_inches='tight')
plt.show()



full_df['Combined Production'] = full_df['production (pj)'].fillna(full_df['Future SARIMA Forecast'])
plt.figure(figsize=(17, 10))
plt.plot(full_df['Year'], full_df['Combined Production'], color='b', linewidth=7)
plt.plot(full_df['Year'], full_df['production (pj)'], label='Produção histórica', color='b', linewidth=7)
plt.plot(full_df['Year'], full_df['Future SARIMA Forecast'], label='Previsão SARIMA', color='orange', linewidth=7)
plt.title('Figura 2: Previsão de produção de biomassa por meio do modelo ARIMA (2024-2034)', fontsize=16)
plt.xlabel('Ano', fontsize=14)
plt.ylabel('Produção (PJ)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color='#dcdcdc', alpha=0.4)
plt.legend(loc='upper left', fontsize=12)
plt.savefig(r"C:\Users\Naja Informatica\Desktop\TCC\exports\producao_previsoesar.png", dpi=700, bbox_inches='tight')
plt.show()


full_df['Combined ProductionCAT'] = full_df['production (pj)'].fillna(full_df['Catboost Forecasted Production (PJ)'])
plt.figure(figsize=(17, 10))
plt.plot(full_df['Year'], full_df['production (pj)'], label='Produção histórica', color='b', linewidth=7)
plt.plot(full_df['Year'], full_df['Combined ProductionCAT'], color='b', linewidth=7)
plt.plot(full_df['Year'], full_df['Catboost Forecasted Production (PJ)'], label='Previsão Catboost', color='green', linewidth=7)
plt.title('Figura 3: Previsão de produção de biomassa por meio de Categorical Boosting (2024-2034)', fontsize=14)
plt.xlabel('Ano', fontsize=14)
plt.ylabel('Produção (PJ)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color='#dcdcdc', alpha=0.4)
plt.legend(loc='upper left', fontsize=12)
plt.savefig(r"C:\Users\Naja Informatica\Desktop\TCC\exports\producao_previsoescat.png", dpi=700, bbox_inches='tight')
plt.show()


full_df['Combined ProductionALL'] = full_df['production (pj)'].fillna(full_df['Media Modelos'])
plt.figure(figsize=(17, 10))
plt.plot(full_df['Year'], full_df['production (pj)'], label='Produção histórica', color='b', linewidth=7)
plt.plot(full_df['Year'], full_df['Combined ProductionALL'], color='b', linewidth=7)
plt.plot(full_df['Year'],full_df['Media Modelos'], label='Média dos Modelos', color='purple', linewidth=7)
plt.title('Figura 4: Previsão de biomassa brasileira por modelo híbrido de Machine Learning', fontsize=16)
plt.xlabel('Ano', fontsize=14)
plt.ylabel('Produção (PJ)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color='#dcdcdc', alpha=0.4)
plt.legend(loc='upper left', fontsize=12)
plt.savefig(r"C:\Users\Naja Informatica\Desktop\TCC\exports\producao_previsoesfin.png", dpi=700, bbox_inches='tight')
plt.show()