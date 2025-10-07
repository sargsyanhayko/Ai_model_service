import pandas as pd

# Загружаем данные
clustering_data = pd.read_pickle("clustering_data.pkl")

# Посмотрим, какие TIN вообще есть
print("Все уникальные TIN:")
print(clustering_data['TIN'].unique())

# Посмотрим, какие года есть для этих TIN
print("\nВсе годы в данных:")
print(clustering_data['TAX_YEAR'].unique())

# Если хотим проверить конкретные TIN
tins_to_check = ['01282006','00029448','00024444']
print("\nДоступные года для выбранных TIN:")
print(clustering_data[clustering_data['TIN'].isin(tins_to_check)][['TIN','TAX_YEAR']])
