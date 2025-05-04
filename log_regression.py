def load_and_prepare_skempi(file_path):
    # Универсальная газовая постоянная (ккал / (моль*K))
    R = 1.987e-3
    T = 298

    df = pd.read_csv(file_path, sep=';')

    # Удалим строки с отсутствующими аффинностями
    df = df.dropna(subset=['Affinity_mut_parsed', 'Affinity_wt_parsed'])

    # Переводим в float
    df['Affinity_mut_parsed'] = df['Affinity_mut_parsed'].astype(float)
    df['Affinity_wt_parsed'] = df['Affinity_wt_parsed'].astype(float)

    # Вычисляем ∆∆G
    df['ddG'] = R * T * np.log(df['Affinity_mut_parsed'] / df['Affinity_wt_parsed'])

    # Метка: 1 — если ddG > 0 (энергия увеличилась), 0 — иначе
    df['ddG_sign'] = (df['ddG'] > 0).astype(int)

    print(df[['Mutation(s)_cleaned', 'Affinity_mut_parsed', 'Affinity_wt_parsed', 'ddG', 'ddG_sign']].head())

    return df

# Используем только мутацию как входной признак (упрощённо)
X_raw = df[['Mutation(s)_cleaned']].copy()
y = df['ddG_sign']

# One-hot кодируем мутации
encoder = OneHotEncoder()
X = encoder.fit_transform(X_raw)

# Разделим на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучим логистическую регрессию
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Предсказание и оценка
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
