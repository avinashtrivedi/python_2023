import sankey_hw as sk
import json
import pandas as pd

artists = pd.read_json('Artists.json')

artists

artists = artists.drop(columns=['ConstituentID', 'DisplayName', 'ArtistBio', 'EndDate', 'Wiki QID', 'ULAN'])

artists['Decade'] = artists['BeginDate'] - (artists['BeginDate'] % 10)

artists = artists.drop('BeginDate', axis=1)

#Nationality and Decade Sankey Diagram
nat_dec = artists.groupby(['Nationality', 'Decade']).size()
for i, v in nat_dec.items():
    if i[1] == 0:
        nat_dec = nat_dec.drop(i)

for i, v in nat_dec.items():
    if v < 30:
        nat_dec = nat_dec.drop(i)

df_nat_dec = pd.DataFrame(columns = ['Nationality', 'Decade', 'Count'])
for i, v in nat_dec.items():
    df_nat_dec.loc[len(df_nat_dec)] = [i[0], i[1], v]

# sk.make_sankey(df_nat_dec, 'Nationality', 'Decade', vals='Count')


#Nationality and Gender Sankey Diagram
nat_gen = artists.groupby(['Nationality', 'Gender']).size()

for i, v in nat_gen.items():
    if v < 30:
        nat_gen = nat_gen.drop(i)

df_nat_gen = pd.DataFrame(columns = ['Nationality', 'Gender', 'Count'])
for i, v in nat_gen.items():
    df_nat_gen.loc[len(df_nat_gen)] = [i[0], i[1], v]

# sk.make_sankey(df_nat_gen, 'Nationality', 'Gender', vals='Count')


#Gender and Decade Sankey Diagram
gen_dec = artists.groupby(['Gender', 'Decade']).size()

for i, v in gen_dec.items():
    if i[1] == 0:
        gen_dec = gen_dec.drop(i)

for i, v in gen_dec.items():
    if v < 20:
        gen_dec = gen_dec.drop(i)

df_gen_dec = pd.DataFrame(columns = ['Gender', 'Decade', 'Count'])
for i, v in gen_dec.items():
    df_gen_dec.loc[len(df_gen_dec)] = [i[0], i[1], v]

# sk.make_sankey(df_gen_dec, 'Gender', 'Decade', vals='Count')


#Nationality, Gender, and Decade Sankey Diagram
nat_gen_dec = artists.groupby(['Nationality', 'Gender', 'Decade']).size()

for i, v in nat_gen_dec.items():
    if i[2] == 0:
        nat_gen_dec = nat_gen_dec.drop(i)

for i, v in nat_gen_dec.items():
    if v < 20:
        nat_gen_dec = nat_gen_dec.drop(i)

df_nat_gen_dec = pd.DataFrame(columns = ['Nationality', 'Gender', 'Decade', 'Count'])
for i, v in nat_gen_dec.items():
    df_nat_gen_dec.loc[len(df_nat_gen_dec)] = [i[0], i[1], i[2], v]

sk.make_sankey(df_nat_gen_dec, 'Nationality', 'Gender', 'Decade',vals='Count')
