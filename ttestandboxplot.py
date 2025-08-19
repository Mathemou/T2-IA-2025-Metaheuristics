import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu

# Dados
gwo_neural = np.array([181.38, 110.28, 27.96, 29.48, 20.21, 19.32, 109.29, 79.83, 
                       106.41, 66.64, 41.99, 42.35, 142.68, 21.99, 105.24, 106.50, 
                       117.57, 44.35, 8.82, 115.68, 118.29, 44.68, 109.20, 69.60, 
                       119.28, 108.12, 44.50, 105.15, 110.55, 111.36])

baseado_regras = np.array([12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 
                           9.95, 19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 
                           15.13, 22.50, 25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 
                           13.80, 15.13, 12.35, 16.19])

genetico_neural = np.array([38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 
                            39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 
                            67.17, 53.54, 33.59, 49.24, 52.65, 16.35, 44.05, 56.59, 
                            63.23, 43.96, 43.82, 19.19, 28.36, 18.65])

humano = np.array([27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 
                   20.05, 31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45, 
                   12.59, 33.01, 21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 
                   22.96, 9.41, 35.22])

# Organizar em dicionário
resultados = {
    "GWO Neural": gwo_neural,
    "Baseado em Regras": baseado_regras,
    "Genético Neural": genetico_neural,
    "Humano": humano
}

# Comparações par a par
pares = [(a, b) for i, a in enumerate(resultados) for j, b in enumerate(resultados) if i < j]

tabela = []
for a, b in pares:
    t_stat, p_t = ttest_ind(resultados[a], resultados[b])
    u_stat, p_w = mannwhitneyu(resultados[a], resultados[b])
    tabela.append([a, b, p_t, "Significativo" if p_t < 0.05 else "NS",
                        p_w, "Significativo" if p_w < 0.05 else "NS"])

df = pd.DataFrame(tabela, columns=["Método A", "Método B", "p-valor T", "Conclusão T", "p-valor Wilcoxon", "Conclusão W"])
print(df)

# Boxplot
df_plot = pd.DataFrame({m: resultados[m] for m in resultados})
df_plot_melted = df_plot.melt(var_name="Método", value_name="Score")

plt.figure(figsize=(8,6))
sns.boxplot(data=df_plot_melted, x="Método", y="Score", palette="Set2")
sns.swarmplot(data=df_plot_melted, x="Método", y="Score", color="black", alpha=0.6)
plt.title("Comparação de Desempenho dos Agentes")
plt.xticks(rotation=15)
plt.show()
