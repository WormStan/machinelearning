from pandas import read_csv, set_option

filename = './section_6/pima_data.csv'

names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

data = read_csv(filename, names = names)
print(data.shape)
print(data.dtypes)

set_option('display.width', 100)
set_option('precision',2)

print(data.describe())

print(data.groupby('class').size())

print(data.corr())

print(data.skew())
