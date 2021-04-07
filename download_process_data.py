import pandas as pd
import unidecode

raw_tolkien_chars = pd.read_html('https://www.behindthename.com/namesakes/list/tolkien/name')
tolkien_names = raw_tolkien_chars[2]['Name']
tolkien_names.to_csv('data/raw/tolkien_names.csv', index=False, header=False)
processed_tolkien_names = tolkien_names.apply(
    unidecode.unidecode).str.lower().str.replace('-', ' ')
processed_tolkien_names = list(
    set([name[0] for name in processed_tolkien_names.str.split()]))
processed_tolkien_names = pd.DataFrame(data=processed_tolkien_names,
                          columns=['name']).sort_values('name')
processed_tolkien_names['tolkien'] = 1

# https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=medguide.page
raw_medication_guide = pd.read_csv('data/raw/medication_guides.csv')
drug_names = raw_medication_guide['Drug Name']
drug_names.to_csv('data/raw/drug_names.csv', index=False, header=False)
processed_drug_names = drug_names.str.lower().str.replace('.', '').str.replace(
    '-', ' ').str.replace('/', ' ').str.replace("'", ' ').str.replace(",", ' ')
processed_drug_names = list(
    set([name[0] for name in processed_drug_names.str.split()]))
processed_drug_names = pd.DataFrame(data=processed_drug_names,
                        columns=['name']).sort_values('name')
processed_drug_names['tolkien'] = 0

dataset = pd.concat([processed_tolkien_names, processed_drug_names], ignore_index=True)
dataset.to_csv('data/processed/dataset.csv', index=False, header=True)
