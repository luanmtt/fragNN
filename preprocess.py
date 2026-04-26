'''

Feature Extraction:

    url; Perfume; Brand; Country; Gender; Rating Value; Rating Count; Year; Top; Middle; Base; Perfumer1; Perfumer2; mainaccord1; mainaccord2; mainaccord3; mainaccord4; mainaccord5
        
        ↓
        
    Perfume; Brand; Gender; Top; Middle; Bottom; mainaccord*

Pseudocode:

    LOAD fra_cleaned.csv with latin1 encoding

    CLEAN
      fill missing accords (2-5) with empty string
        → tem que ser feito.

    ENCODE ACCORDS
      collect all unique accord values across the 5 accord columns
      for each row, build a binary vector of size 84
        for each accord in vocabulary:
          if accord appears in any of the 5 columns → 1 
          else → 0

    ENCODE TARGET
      men    → 0
      women  → 1
      unisex → 2

    BALANCE CLASSES
      find the class with fewest rows (men ~ 5k)
      undersample women and unisex to match

    SPLIT
      shuffle the dataset
      80% → train.csv
      20% → test.csv

    SAVE
      columns: 84 accord flags + rating + count + year + label
      no index, no header strings — pure floats + int label

'''


import pandas as pd
import numpy as np


df = pd.read_csv('datasets/fra_cleaned.csv', sep=';', encoding='latin1')
rows, cols = df.shape


# --------------------------------------------------------------------------------------------------
# tratamento original dos dados requisitados

'''

'''

df = df[['Perfume', 
         'Brand', 
         'Gender', 
         'Top', 
         'Middle', 
         'Base',
         'mainaccord1',
         'mainaccord2',
         'mainaccord3',
         'mainaccord4',
         'mainaccord5']] # pega apenas as colunas de interesse.


accords = ['mainaccord1', 'mainaccord2','mainaccord3', 'mainaccord4', 'mainaccord5']
for acc in accords:
    df[acc] = pd.Series(df[acc].values).fillna('')
    

df['Top']       = pd.Series(df['Top'].values).fillna('')
df['Middle']    = pd.Series(df['Middle'].values).fillna('')
df['Base']      = pd.Series(df['Base'].values).fillna('')

df['top_notes']     = df['Top'].str.split(', ')
df['middle_notes']  = df['Middle'].str.split(', ')
df['base_notes']    = df['Base'].str.split(', ')


# --------------------------------------------------------------------------------------------------
# Vocabulário de notas dentre: 'Top', 'Middle' e 'Bottom'

'''

    A partir de um set(), que é uma coleção de dados únicos, há
    a iteração dentre todas as células nas colunas top, middle e base escaneando
    por notas. No dataset, elas estão na forma: 'watermelon, mango, 
    papaya, pineapple, peach, lemon', logo, temos que splitar pela vírgula.
    O resultado será: ['watermelon', 'mango', 'papaya', 'pineapple', 'peach', 'lemon']
    
    Quando se acha uma nota única, a adiciona em all_notes. 
    Processo final: se ordena all_notes em ordem alfabética.
    É construído uma tradução de 'str' -> 'int' da seguinte forma:
    
    [ 'aldehydes', 'amber' ... ]
        
        ↓

    {'aldehydes': 0, 'amber': 1, ...}

'''

all_notes = set()

for col in ['top_notes', 'middle_notes', 'base_notes']:
    for cell in df[col]:

        if isinstance(cell, list):
            notes = cell
        else:
            notes = str(cell).split(', ')
        
        for note in notes:
            if note.strip() != '':
                all_notes.add(note.strip())


all_notes = sorted(all_notes)
note_indexer = {note: i for i, note in enumerate(all_notes)}
print(f'All notes: {len(all_notes)}')


# --------------------------------------------------------------------------------------------------
# Vocabulário de acordes dentro de 'mainaccords':

'''

O procedimento é igual ao procedimento acima.

'''

all_accords = set()

for col in accords:
    for val in df[col]:

        if val.strip() != '':
            all_accords.add(val.strip())
 
all_accords = sorted(all_accords)
accord_indexer = {acc: i for i, acc in enumerate(all_accords)}
print(f'All accords: {len(all_accords)}')


# --------------------------------------------------------------------------------------------------
# Pegando os números totais de notas e acordes, computando num_features.

'''

    Além de pegar esses números, aqui está sendo feita a verificação de 
    existência de todas as notas. Itera-se com itertuples pelas colunas
    de top, middle e base, em todas as suas células, checando o vocábulário
    numérico de note_indexer por equivalentes. Se acha, temos aquele espaço = 1.
    
    O mesmo processo é feito com os accords e accords_indexer.

'''

rows = len(df)

num_notes = len(all_notes)
num_accords = len(all_accords)
num_features = num_notes + num_accords

X = np.zeros((rows, num_features), dtype=np.float32)

for i, row in enumerate(df.itertuples()):

    for col in ['top_notes','middle_notes', 'base_notes']:
        
        cell = getattr(row, col)
        if isinstance(cell, list):
            notes = cell
        
        else:
            notes = str(cell).split(', ')

        for note in notes:
            note = note.strip()

            if note in note_indexer:
                X[i, note_indexer[note]] = 1.0

    for col in accords:
        val = getattr(row,col).strip()

        if val in accord_indexer:
            X[i,num_notes + accord_indexer[val]] = 1.0
    
        
print(f'X shape: {X.shape}')


# --------------------------------------------------------------------------------------------------
# Encoding das características target:

'''
    
    Nessa etapa temos a construção de dois arrays 'labeleds'.

    No fim, temos:   

        • labeled_genders contendo os valores possíveis de gênero.
        
        • labeled_accords contendo os valores possíveis de acordes comparativamente à accord_indexer.           
            → remoção de linhas nas quais um acorde foi incongruente à accord_indexer.
'''

genders = { 'men': 0, 'women': 1, 'unisex': 2 }

labeled_genders = np.array([genders[g] for g in df['Gender']], dtype=np.int32)

labeled_accords = np.array([

    accord_indexer.get(str(a).strip(), -1)
    for a in df['mainaccord1']
    
], dtype=np.int32)

mask = labeled_accords != -1

X = X[mask]
labeled_genders = labeled_genders[mask]
labeled_accords = labeled_accords[mask]

print(f"Rows after filtering unknown accords: {len(X)}")


# --------------------------------------------------------------------------------------------------
# Class balance:

'''
    
    Para garantir o balanço entre os gêneros, é necessário cortar por baixo.
    ! - ver se isso não vai capar os femininos e masculinos. 

'''

min_gender_count = min(np.sum(labeled_genders == c) for c in [0,1,2])
print(f"Balancing to {min_gender_count} rows per gender class.")


balanced_idx = []

for c in [0,1,2]:
    
    '''
        
        np.where() retorna um tuple se a condição for satisfeita. [0]
        é o índice do tuple no qual se encontra um array de índices 
        relativos aos hits.

        depois se seleciona aleatoriamente um quantidade de 'min_gender_count'
        de idx's de cada tipo.

        essas quantidades são appendadas em balanced_idx.a

    '''

    idx = np.where(labeled_genders == c)[0]
    chosen = np.random.choice(idx, size=min_gender_count, replace=False)
    balanced_idx.extend(chosen.tolist())


balanced_idx = np.array(balanced_idx)
np.random.shuffle(balanced_idx) # dados sofrem shuffle: dados sequenciais são mais fáceis de obter biases.

# é feita a extração apenas das linhas que são equivalentes à máscara.
X = X[balanced_idx]
labeled_genders = labeled_genders[balanced_idx]
labeled_accords = labeled_accords[balanced_idx]

print(f"Balanced dataset size: {len(X)}")


# --------------------------------------------------------------------------------------------------
# Divisão 80/20: 80 para treinamento e 20 para predicts.


split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]

lb_gen_train,lb_gen_test    = labeled_genders[:split], labeled_genders[split:]
lb_acc_train, lb_acc_test  = labeled_accords[:split], labeled_accords[split:]


# --------------------------------------------------------------------------------------------------
# Salvar dataset final:

'''
    
    Salvando o dataset: 
        → concatena os labels novos advindos da tradução com indexers.
        → concatena o df de interesse às novas labels e transforma para .csv.

'''

def save_split(X, labeled_genders, labeled_accords, filename):

    labels = np.stack([labeled_genders, labeled_accords], axis=1)
    data = np.concatenate([X, labels.astype(np.float32)], axis=1)

    pd.DataFrame(data).to_csv(filename, index=False, header=False)

    print(f"Saved {filename} - shape {data.shape}")



save_split(X_train, lb_acc_train, lb_gen_train, 'data/train.csv')
save_split(X_test, lb_acc_test, lb_gen_test, 'data/test.csv')


# --------------------------------------------------------------------------------------------------
# Salvando .txts de vocabulários

'''
    
    Salvando os vocabulários:

        → Uma nota por linha em 'note_vocab'
        → Um acorde por linha em 'accord_vocab'

'''

with open('data/note_vocab.txt', 'w') as f:
    for note in all_notes:
        f.write(note + '\n')


with open('data/accord_vocab.txt', 'w') as f:
    for acc in all_accords:
        f.write(acc + '\n')


print("Vocabs saved.")
print(f"Input size: {num_features}")
print(f"Train rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")


# --------------------------------------------------------------------------------------------------
