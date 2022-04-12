from numpy import NaN
import pandas as pd
import re
import os
from tqdm import tqdm


def load_and_prepare(in_folder: str):
    """
    The in_folder will contain two files:
     - person.ttl
     - name.ttl

    Combining the data to generate the y values (0 or 1),
    """

    # Reading and processing the name.ttl file
    names = {}

    print('reading name file . . .')
    with open(f'{in_folder}/name.ttl', 'r') as namefile:
        for i, line in enumerate(tqdm(namefile)):
            idx = re.search(r"(/)(Q[^>]+)(>)", line)
            if idx:
                idx = idx.group(2)
            else:
                idx = None

            name = re.search(
                r'(<http://xmlns.com/foaf/0.1/name>)\s"(.*)"@[^.]+.', line)
            if name:
                name = name.group(2)
            else:
                name = None

            names.update({i: {'idx': idx, 'name': name}})

    names_df = pd.DataFrame(names).T

    # Reading and processing the person.ttl file
    person_names = {}

    print('reading person file . . .')
    with open(f'{in_folder}/person.ttl', 'r') as namefile:
        for i, line in enumerate(tqdm(namefile)):
            idx = re.search(r"(/)(Q[^>]+)(>)", line)
            if idx:
                idx = idx.group(2)
            else:
                idx = None

            person_names.update({i: {'idx': idx}})

    person_df = pd.DataFrame(person_names).T

    print('Saving names file as csv . . .')
    names_df.to_csv(os.path.join(
        in_folder, 'unlabelled_names.csv'), index=False, encoding='utf-8-sig')

    print('Saving person file as csv . . .')
    person_df.to_csv(os.path.join(
        in_folder, 'person.csv'), index=False, encoding='utf-8-sig')

    # Merging both files to label the dataset properly
    person_df_list = list(person_df.idx)

    matches_bool = names_df.idx.isin(person_df_list)

    names_df['label'] = matches_bool * 1

    names_df.dropna()

    print('Saving labelled_dataset file as csv . . .')
    names_df.to_csv(os.path.join(
        in_folder, 'labelled_dataset.csv'), index=False, encoding='utf-8-sig')
