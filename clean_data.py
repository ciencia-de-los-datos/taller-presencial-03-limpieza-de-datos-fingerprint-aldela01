"""Taller evaluable presencial"""

import nltk
import pandas as pd


def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame"""

    df=pd.read_csv(input_file)
    
    return df

def create_fingerprint(df):
    """Cree una nueva columna en el DataFrame que contenga el fingerprint de la columna 'text'"""

    # 1. Copie la columna 'text' a la columna 'fingerprint'

    df = df.copy()
    df["key"] = df["text"]
    # 2. Remueva los espacios en blanco al principio y al final de la cadena
    df["key"] = df["key"].str.strip()
    # 3. Convierta el texto a minúsculas
    df["key"] = df["key"].str.lower()
    # # 4. Transforme palabras que pueden (o no) contener guiones por su version sin guion.
    df["key"] = df["key"].str.replace("-", "")
    # # 5. Remueva puntuación y caracteres de control
    df["key"] = df["key"].str.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    # # 6. Convierta el texto a una lista de tokens
    df["key"] = df["key"].str.split()
    # 7. Transforme cada palabra con un stemmer de Porter
    stemmer = nltk.PorterStemmer()
    # 8. Ordene la lista de tokens y remueve duplicados
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))
    # # # 9. Convierta la lista de tokens a una cadena de texto separada por espacios
    df["key"] = df["key"].str.join(" ")

    return df



def generate_cleaned_column(df):
    """Crea la columna 'cleaned' en el DataFrame"""

    df = df.copy()

    # 1. Ordene el dataframe por 'fingerprint' y 'text'
    df = df.sort_values(by=["key", "text"],ascending=[True, True])
    # 2. Seleccione la primera fila de cada grupo de 'fingerprint' creando una variable keys
    keys = df.drop_duplicates(subset="key", keep="first")
    # 3.  Cree un diccionario con 'fingerprint' como clave y 'text' como valor
    key_dict = dict(zip(keys["key"], keys["text"]))
    # 4. Cree la columna 'cleaned' usando el diccionario
    df["cleaned"] = df["key"].map(key_dict)
    return df

def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""
    # Solo contiene una columna llamada 'texto' al igual
    # que en el archivo original pero con los datos limpios
    df=df.copy()
    df=df[["cleaned"]]
    df=df.rename(columns={"cleaned":"text"})
    df.to_csv(output_file, index=False)


def main(input_file, output_file):
    """Ejecuta la limpieza de datos"""

    df = load_data(input_file)
    df = create_fingerprint(df)
    df = generate_cleaned_column(df)
    #Rename key column to fingerprint
    #df = df.rename(columns={"key":"fingerprint"})
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )
