import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
#
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