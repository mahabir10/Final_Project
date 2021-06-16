from numpy import load

data = load('cora.npz')
lst = data.files
print(data['feature_shape'])

for item in lst:
    print(item)
    ## print(data[item])
