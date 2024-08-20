# Обробка відсутніх значень. Заповнення відсутнього віку середнім віком або прогнозуйте вік за допомогою
# іншої моделі.
from helper import (pd, np, SimpleImputer, OneHotEncoder, plt, norm, sns, kendalltau, re, StandardScaler,
                    train_test_split, train_test_split, LinearRegression, joblib, mean_squared_error, r2_score,
                    cross_val_score, Lasso, ElasticNet, Ridge, RandomForestRegressor, Dense, Sequential, gym, spaces,
                    GridSearchCV, accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                    StratifiedKFold, confusion_matrix)

titanic = pd.read_csv('titanic.csv')

# # Скидаємо дані які не впливають на наші процеси

data_age = titanic[['Age']]

# Функція для виділення титулу зі строки Name
def extract_title(name):
    # Шукаємо слово, що закінчується на крапку, за яким слідує пробіл
    title_search = re.search(' ([A-Za-z]+)\\.', name)
    # Якщо титул знайдено, повертаємо його
    if title_search:
        return title_search.group(1)
    return ""

# Створення нового стовпця 'Title', використовуючи функцію extract_title
titanic['Title'] = titanic['Name'].apply(extract_title)

titanic = titanic.drop(['Name', 'Ticket','Age', 'Cabin'], axis=1)
titanic = titanic.dropna(subset=['Embarked'])

# Створюємо екземпляр SimpleImputer з стратегією заміни відсутніх значень середніми
imputer_mean = SimpleImputer(strategy='mean')

# # Застосування заповнення пропущених значень з різними стратегіями
imputed_mean = imputer_mean.fit_transform(data_age)

# # Конвертування результатів в DataFrame
imputed_mean_df = pd.DataFrame(imputed_mean, columns=data_age.columns)
# обєднюємо стовпчик 'Age' у загальний датафрейм
titanic = pd.concat([titanic, imputed_mean_df], axis=1)

# Кодування категоріальних змінних ('sex', 'Embarked') за допомогою One-Hot Encoding
categorical_column = ['Sex', 'Embarked', 'Title']
# Витягування категоріальної змінної з даних
data = titanic[['Sex', 'Embarked', 'Title']]
# Створення екземпляру OneHotEncoder
encoder = OneHotEncoder()
# Виконання one-hot encoding
encoded_data = encoder.fit_transform(data).toarray()
# Створення DataFrame з закодованими даними
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Sex', 'Embarked', 'Title']))
# Об'єднання закодованих даних з вихідним DataFrame
titanic_data_encoded = pd.concat([titanic, encoded_df], axis=1)

# скидуємо не потрібний матеріал

titanic = titanic_data_encoded.drop(['Sex_nan','Embarked_nan', 'Title_Capt',
                        'Title_Col', 'Title_Col', 'Title_Countess', 'Title_Don',
                        'Title_Dr', 'Title_Jonkheer', 'Title_Lady',
                        'Title_Major', 'Title_Master', 'Title_Mlle',
                        'Title_Mme', 'Title_Rev', 'Title_nan'], axis=1)


titanic = titanic_data_encoded.drop(['Sex', 'Embarked', 'Title'], axis=1)
# print(titanic.to_markdown())
# 5. Інженерія ознак.
# Створення нової ознаки FamilySize
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1

# Створення нової ознаки IsAlone
titanic['IsAlone'] = (titanic['FamilySize'] == 1).astype(int)
# titanic['Survived'] = titanic['Survived'].replace({0: 'ні', 1: 'так'})
print(titanic.to_markdown())

# - Масштабування ознак, якщо використовуються моделі, чутливі до масштабування ознак, такі як SVM або KNN.
# Створення екземпляру StandardScaler
scaler = StandardScaler()
# Застосування стандартизації до даних
scaled_data = scaler.fit_transform(titanic)
# Конвертування результату в DataFrame
scaled_titanic = pd.DataFrame(scaled_data, columns=titanic.columns)
# scaled_titanic = scaled_titanic['Survived'].replace({0: 'ні', 1: 'так'})


# 6. - Розділіть дані на навчальний і тестовий набори для оцінки продуктивності ваших моделей.
x = scaled_titanic.drop(['Survived'], axis=1).drop([889, 890])
y = scaled_titanic['Survived'].drop([889, 890])
# print(x.to_markdown())

# Розділіть дані на тренувальний та тестовий набори (наприклад, в співвідношенні 80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Створення екземпляру LinearRegression
# model = LinearRegression()
# # Підгонка моделі до тренувальних даних
# model.fit(x_train, y_train)
# # Прогнозування на тестових даних
# y_pred = model.predict(x_test)
#
# # joblib.dump(model, 'model.dump')
#
# # Оцінка моделі, обчислення метрик, таких як середньоквадратична
# # помилка та коефіцієнт детермінації:
# # Обчислення середньоквадратичної помилки
# mse = mean_squared_error(y_test, y_pred)
# # Обчислення коефіцієнта детермінації
# r2 = r2_score(y_test, y_pred)
# # Виведення метрик оцінкиValidating and finalisation of model 22
# print("Середньоквадратична помилка:", mse)
# print("Коефіцієнт детермінації:", r2)
#
#
# # Оцінка моделі за допомогою крос-валідації:
# scores = cross_val_score(model, x_train, y_train, cv=5)
# # Виведення результатів крос-валідації
# print("Оцінки крос-валідації згідно тренувальних даних:", scores)
# print("Середнє значення кросвалідації згідно тренувальних даних:", scores.mean())
#
#
# # Створення екземпляру Lasso регресії
# lasso = Lasso(alpha=0.5)
# # Підгонка моделі до тренувальних даних
# lasso.fit(x_train, y_train)
# # Прогнозування на тестових даних
# y_pred = lasso.predict(x_test)
# # Обчислення середньоквадратичної помилки
# mse = mean_squared_error(y_test, y_pred)
# print("Середньоквадратична помилка (Lasso):", mse)
# # Визначте діапазон параметрів alpha, які ви хочете перевірити
# param_grid = {
# #     'alpha': [0.1, 1, 10, 100]
# }

# # Визначте модель Lasso
# lasso = Lasso()
#
# # Використайте GridSearchCV для пошуку найкращих параметрів
# grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(x_train, y_train)
#
# # Оцінка моделі за допомогою крос-валідації:
# scores = cross_val_score(lasso, x_train, y_train, cv=5)
# # Виведення результатів крос-валідації
# print("Оцінки крос-валідації згідно тренувальних даних:", scores)
# print("Середнє значення кросвалідації згідно тренувальних даних:", scores.mean())
# # Оцінка якості моделі на тестових даних
# best_model = grid_search.best_estimator_
# best_score = best_model.score(x_test, y_test)
# print("Найкращий результат на тестових даних:", best_score)
# # Обчислення середньоквадратичної помилки
# mse = mean_squared_error(y_test, y_pred)
# # Обчислення коефіцієнта детермінації
# r2 = r2_score(y_test, y_pred)
# # Виведення метрик оцінкиValidating and finalisation of model 22
# print("Середньоквадратична помилка:", mse)
# print("Коефіцієнт детермінації:", r2)

#
#
# # Створення екземпляру ElasticNet регресії
# elasticnet = ElasticNet(alpha=0.5, l1_ratio=0.5)
# # Підгонка моделі до тренувальних даних
# elasticnet.fit(x_train, y_train)
# # Прогнозування на тестових даних
# y_pred = elasticnet.predict(x_test)
# # Обчислення середньоквадратичної помилки
# mse = mean_squared_error(y_test, y_pred)
# print("Середньоквадратична помилка (ElasticNet):", mse)
#
# # Визначення сітки гіперпараметрів
# param_grid = {'alpha': [0.1, 1, 10],
#               'l1_ratio': [0.1, 0.5, 0.7, 0.9]}
#
# # Створення моделі ElasticNet
# elastic_net = ElasticNet()
#
# # Хресна перевірка з гіперпараметрами
# grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid,
#                            scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
#
# # Підгонка моделі
# grid_search.fit(x_train, y_train)
#
# # Найкращі параметри
# best_params = grid_search.best_params_
# print("Найкращі параметри:", best_params)
#
# # Оцінка моделі на тестовому наборі
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Середньо-квадратична помилка на тестовому наборі:", mse)
#
# # Оцінка моделі за допомогою крос-валідації:
# scores = cross_val_score(elastic_net, x_train, y_train, cv=5)
# # Виведення результатів крос-валідації
# print("Оцінки крос-валідації згідно тренувальних даних:", scores)
# print("Середнє значення кросвалідації згідно тренувальних даних:", scores.mean())
# # Оцінка якості моделі на тестових даних
# best_model = grid_search.best_estimator_
# best_score = best_model.score(x_test, y_test)
# print("Найкращий результат на тестових даних:", best_score)
# # Обчислення середньоквадратичної помилки
# mse = mean_squared_error(y_test, y_pred)
# # Обчислення коефіцієнта детермінації
# r2 = r2_score(y_test, y_pred)
# # Виведення метрик оцінкиValidating and finalisation of model 22
# print("Середньоквадратична помилка:", mse)
# print("Коефіцієнт детермінації:", r2)


#
# Створення екземпляру Ridge регресії
ridge = Ridge(alpha=0.5)
# Підгонка моделі до тренувальних даних
ridge.fit(x_train, y_train)
# Прогнозування на тестових даних
y_pred = ridge.predict(x_test)
# Обчислення середньоквадратичної помилки
mse = mean_squared_error(y_test, y_pred)
print("Середньоквадратична помилка (Ridge):", mse)

# Визначення сітки гіперпараметрів
param_grid = {'alpha': [0.1, 1, 10]}

# Створення моделі Ridge
ridge = Ridge()

# Хресна перевірка з гіперпараметрами
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)

# Підгонка моделі
grid_search.fit(x_train, y_train)

# Найкращі параметри
best_params = grid_search.best_params_
print("Найкращі параметри:", best_params)

# Оцінка моделі на тестовому наборі
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Середньо-квадратична помилка на тестовому наборі:", mse)

# Оцінка моделі за допомогою крос-валідації:
scores = cross_val_score(ridge, x_train, y_train, cv=5)
# Виведення результатів крос-валідації
print("Оцінки крос-валідації згідно тренувальних даних:", scores)
print("Середнє значення кросвалідації згідно тренувальних даних:", scores.mean())
# Оцінка якості моделі на тестових даних
best_model = grid_search.best_estimator_
best_score = best_model.score(x_test, y_test)
print("Найкращий результат на тестових даних:", best_score)
# Обчислення середньоквадратичної помилки
mse = mean_squared_error(y_test, y_pred)
# Обчислення коефіцієнта детермінації
r2 = r2_score(y_test, y_pred)
# Виведення метрик оцінкиValidating and finalisation of model 22
print("Середньоквадратична помилка:", mse)
print("Коефіцієнт детермінації:", r2)



# # Спробуємо використати підсилювальне градієнтне навчання
# class TitanicEnv(gym.Env):
#     def __init__(self, df):
#         super(TitanicEnv, self).__init__()
#         self.df = df
#         self.observation_space = spaces.Discrete(len(df))  # Кількість пасажирів
#         self.action_space = spaces.Discrete(2)  # Дія: 0 - не вижив, 1 - вижив
#         self.current_step = 0
#
#     def reset(self):
#         self.current_step = 0
#         return self.df.iloc[self.current_step]
#
#     def step(self, action):
#         assert self.action_space.contains(action)
#         reward = self._get_reward(action)
#         self.current_step += 1
#         done = self.current_step >= len(self.df)
#         if done:
#             next_state = None
#         else:
#             next_state = self.df.iloc[self.current_step]
#         return next_state, reward, done, {}
#
#     def _get_reward(self, action):
#         # Отримати винагороду на основі вибору дії та даних пасажира
#         passenger = self.df.iloc[self.current_step]
#         if (action == 1 and passenger['Survived'] == 1) or (action == 0 and passenger['Survived'] == 0):
#             return 1  # Позитивна винагорода за правильне рішення
#         else:
#             return 0  # Відсутність винагороди за неправильне рішення
#
#     def render(self, mode='human'):
#         # Додаткові можливості візуалізації
#         pass
#
#     def close(self):
#         pass
#
#
# # Завантажуємо дані
# df = scaled_titanic
#
# # Створюємо середовище
# env = TitanicEnv(df)
#
# # Початковий стан
# state = env.reset()
# print("Початковий стан:")
# print(state)
#
# # Здійснити дію
# action = 1  # Наприклад, припустимо, що агент вирішив, що пасажир виживе
# next_state, reward, done, _ = env.step(action)
# print("\nНаступний стан після дії {}: reward={}, done={}".format(action, reward, done))
# print(next_state)
# x = scaled_titanic.drop(['Survived'], axis=1).drop([889, 890])
# y = scaled_titanic['Survived'].drop([889, 890])
# n_splits = 5

# # Створення моделі нейронної мережі
# model = Sequential()
# model.add(Dense(12, activation='relu', input_dim=x_train.shape[1]))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# # Компіляція моделі
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Підгонка моделі
# model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)
#
# # Оцінка моделі на тестовому наборі
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f'Test Accuracy: {accuracy}')
# y_pred = model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error on Test Data:", mse)



 # Розвідувальний аналіз даних (EDA)

# - Аналіз розподілу ключових характеристик та їх зв'язку з цільовою змінною, Вижив.
# - Використання візуалізацій для кращого розуміння даних: гістограми, коробкові діаграми та точкові діаграми.
# - Дослідження кореляцій між характеристиками.
# Вибір відповідних ознак і цільової змінної


# sns.barplot(x='Pclass', y='Fare', hue='Survived', data=titanic_data_encoded, errorbar=None, palette='muted')
# plt.show()
# sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic, errorbar=None, palette='muted')
# plt.show()
# # sns.barplot(x='Pclass', y='Fare', hue='Survived', data=titanic_data_encoded, errorbar=None, palette='muted')
# plt.show()
# # sns.barplot(x='Pclass', y='Age', hue='Survived', data=titanic, errorbar=None, palette='muted')
# plt.show()
# # sns.scatterplot(x='Fare', y='Age', data=titanic, palette='bright', hue='Survived')
# plt.show()
#
# # будуємо графік розподілу віку
# plot_df = titanic['Age'].value_counts()
# sns.barplot(plot_df)
# plt.show()
#
# # Вибір числових стовпців
# numeric_columns = titanic.select_dtypes(include=['float64', 'int64'])
#
# # Створення теплової карти кореляції
# sns.heatmap(data=numeric_columns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()

















