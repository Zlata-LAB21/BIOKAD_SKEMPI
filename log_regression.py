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
