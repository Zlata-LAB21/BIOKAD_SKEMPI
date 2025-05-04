# Базовые свойства для 20 стандартных аминокислот
amino_acid_props = {
    'A': {'hydrophobicity': 1.8,  'volume':  88.6},
    'R': {'hydrophobicity': -4.5, 'volume': 173.4},
    'N': {'hydrophobicity': -3.5, 'volume': 114.1},
    'D': {'hydrophobicity': -3.5, 'volume': 111.1},
    'C': {'hydrophobicity': 2.5,  'volume': 108.5},
    'E': {'hydrophobicity': -3.5, 'volume': 138.4},
    'Q': {'hydrophobicity': -3.5, 'volume': 143.8},
    'G': {'hydrophobicity': -0.4, 'volume': 60.1},
    'H': {'hydrophobicity': -3.2, 'volume': 153.2},
    'I': {'hydrophobicity': 4.5,  'volume': 166.7},
    'L': {'hydrophobicity': 3.8,  'volume': 166.7},
    'K': {'hydrophobicity': -3.9, 'volume': 168.6},
    'M': {'hydrophobicity': 1.9,  'volume': 162.9},
    'F': {'hydrophobicity': 2.8,  'volume': 189.9},
    'P': {'hydrophobicity': -1.6, 'volume': 112.7},
    'S': {'hydrophobicity': -0.8, 'volume': 89.0},
    'T': {'hydrophobicity': -0.7, 'volume': 116.1},
    'W': {'hydrophobicity': -0.9, 'volume': 227.8},
    'Y': {'hydrophobicity': -1.3, 'volume': 193.6},
    'V': {'hydrophobicity': 4.2,  'volume': 140.0},
}


def parse_mutation(mutation):
    import re
    # Пример: 'LI38G' → (L, 38, G)
    match = re.match(r'([A-Z]+)(\d+)([A-Z]+)', mutation)
    if match:
        from_aa, pos, to_aa = match.groups()
        return from_aa, int(pos), to_aa
    return None, None, None

# Преобразуем только одиночные мутации
df_single = df[~df['Mutation(s)_cleaned'].str.contains(',')].copy()
parsed = df_single['Mutation(s)_cleaned'].apply(parse_mutation)
df_single[['from_aa', 'pos', 'to_aa']] = pd.DataFrame(parsed.tolist(), index=df_single.index)

# Удалим строки с ошибками парсинга
df_single = df_single.dropna(subset=['from_aa', 'to_aa'])

# Добавим свойства
df_single['from_hydro'] = df_single['from_aa'].map(lambda aa: amino_acid_props.get(aa, {}).get('hydrophobicity', 0))
df_single['to_hydro'] = df_single['to_aa'].map(lambda aa: amino_acid_props.get(aa, {}).get('hydrophobicity', 0))
df_single['from_vol'] = df_single['from_aa'].map(lambda aa: amino_acid_props.get(aa, {}).get('volume', 0))
df_single['to_vol'] = df_single['to_aa'].map(lambda aa: amino_acid_props.get(aa, {}).get('volume', 0))

# Разница свойств
df_single['hydro_change'] = df_single['to_hydro'] - df_single['from_hydro']
df_single['vol_change'] = df_single['to_vol'] - df_single['from_vol']


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Признаки и целевая переменная
features = ['hydro_change', 'vol_change']
X = df_single[features]
y = df_single['ddG_sign']

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy (Random Forest, биофиз. признаки): {acc:.2f}")
