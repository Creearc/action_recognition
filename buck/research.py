import json


with open('result.json', 'r') as f:
    data = json.load(f)

with open('../data/kinetics_classnames.json', 'r') as f:
    tmp_classes = json.load(f)
print(tmp_classes.keys())
with open('../data/h_classnames.json', 'r') as f:
    h_classes = json.load(f)
    
classes = dict()

for key, value in tmp_classes.items():
    classes[value] = key
print(classes)

for key, value in data.items():
    elements = list(set(value))

    counts = dict()

    for e in elements:
        count = value.count(e)
        if count > 4:
            counts[classes[e]] = count
    
    data[key] = counts.copy()

    print(key, data[key])    

print('_'*100)

result = dict()
for key, value in data.items():
    for key2, value2 in value.items():

        if not(key2 in result):
            result[key2] = dict()
        result[key2][key] = value2

print('_'*100)
out = dict()
for key, value in result.items():
    print(key, value)

    mx, ind = 0, 0
    for key2, value2 in value.items():
        if value2 > mx:
            mx = value2
            ind = key2

    print(key, ind, mx)

    out[tmp_classes[key]] = h_classes[ind]

print(out)
with open('class_converter.json', 'w') as f:
    json.dump(out, f)
