import numpy as np
import sys
from copy import deepcopy


# make noNULL file with: grep -v NULL rawdata_mkmk01.csv | cut -f1,3,4,6- -d, > rawdata_mkmk01_noNULL.csv
EPS = 1e-8
np.random.seed(1234)
#### LOOK AT THIS FUNCTION!!!! GETTING STD = 0
def bucket(x, buckets):
    x = float(x)
    n = len(buckets)
    label = n
    for i in range(len(buckets)):
        if x <= buckets[i]:
            label = i
            break
    template = [0. for j in range(n + 1)]
    template[label] = 1.
    return template

def onehot(x, choices):
    if not x in choices:
        print('could not find "{}" in choices'.format(x))
        print(choices)
        raise Exception()
    label = choices.index(x)
    template = [0. for j in range(len(choices))]
    template[label] = 1.
    return template

def continuous(x):
    return [float(x)]

def toBinaryAttr(sent_attr, val):
    # convert sensitive attribute to binary
    if len(val) == 2:
        # print('sensitive variable is already binary')
        return val

    if len(val) == 1: 
        raise ValueError("No implmentation for converting continuous variable to binary variable")

    # convert sensitive attribute to binary
    if sensitive == 'race':
        label = np.argmax(val)
        cat2bin = {4:0, 0:1, 1:1, 2:1, 3:1} # white, Nonwhite
        if label not in cat2bin: 
            print(label)
            raise ValueError('fail to convert sent variable to binary')
        label = cat2bin[label]
        template = [0., 0.]
        template[label] = 1.
        return template
    elif sensitive == 'age':
        label = np.argmax(val)
        cat2bin = {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1} # <= 35,  > 35
        if label not in cat2bin: 
            print(label)
            raise ValueError('fail to convert sent variable to binary')
        label = cat2bin[label]
        template = [0., 0.]
        template[label] = 1.
        return template
    elif sensitive == 'education':
        label = np.argmax(val)
        # low education: 0 all other ; high education: 1 (Some-college, Bachelors, Masters, Doctorate)
        cat2bin = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:1, 10:0, 11:1, 12:0, 13:0, 14:1, 15:1}
        if label not in cat2bin: 
            print(label)
            raise ValueError('fail to convert sent variable to binary')
        label = cat2bin[label]
        template = [0., 0.]
        template[label] = 1.
        return template
    else:
        raise NotImplementedError

def parse_row(row, headers, headers_use):
    new_row_dict = {}
    for i in range(len(row)):
        x = row[i]
        hdr = headers[i]
        new_row_dict[hdr] = fns[hdr](x)
    sens_att = new_row_dict[sensitive] # sens_att = toBinaryAttr(sensitive, new_row_dict[sensitive])
    label = new_row_dict[target]
    new_row = []
    for h in headers_use:
        new_row = new_row + new_row_dict[h]
    return new_row, label, sens_att

def whiten(X, mn, std):
    mntile = np.tile(mn, (X.shape[0], 1))
    stdtile = np.maximum(np.tile(std, (X.shape[0], 1)), EPS)
    X = X - mntile
    X = np.divide(X, stdtile)
    return X


if __name__ == '__main__':
    f_in_tr = 'data/adult.data'
    f_in_te = 'data/adult.test'

    headers = 'age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income'.split(',')
    headers_use = deepcopy(headers)
    target = 'income' # 'income' 'education-num'
    sensitive = 'sex' # 'age' 'sex' 'race'
    # remove sensitive attribute from X
    headers_use.remove(sensitive)
    # remove target attribute from X
    headers_use.remove(target)
    if target == 'education-num':
        headers_use.remove('education')

    print("# of attributes in dataset:", len(headers))
    print("# of attributes in X:", len(headers_use))


    f_out_np = 'data/adult_%s_%s.npz' % (target, sensitive)
    hd_file = 'data/adult_%s_%s.headers' % (target, sensitive)
    f_out_csv = 'data/adult.csv'

    header_list = open(hd_file, 'w')

    REMOVE_MISSING = True
    MISSING_TOKEN = '?'

    options = {
        'age': 'buckets',
        'workclass': 'Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked',
        'fnlwgt': 'continuous',
        'education': 'Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool',
        'education-num': 'continuous',
        'marital-status': 'Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse',
        'occupation': 'Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces',
        'relationship': 'Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried',
        'race': 'White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black',
        'sex': 'Female, Male',
        'capital-gain': 'continuous',
        'capital-loss': 'continuous',
        'hours-per-week': 'continuous',
        'native-country': 'United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands',
        'income': ' <=50K,>50K'
    }

    buckets = {'age': [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]} # original bucket {'age': [40,]} 

    options = {k: [s.strip() for s in sorted(options[k].split(','))] for k in options}
    print(options)

    fns = {
        'age': lambda x: bucket(x, buckets['age']),
        'workclass': lambda x: onehot(x, options['workclass']),
        'fnlwgt': lambda x: continuous(x),
        'education': lambda x: onehot(x, options['education']),
        'education-num': lambda x: continuous(x),
        'marital-status': lambda x: onehot(x, options['marital-status']),
        'occupation': lambda x: onehot(x, options['occupation']),
        'relationship': lambda x: onehot(x, options['relationship']),
        'race': lambda x: onehot(x, options['race']),
        'sex': lambda x: onehot(x, options['sex']),
        'capital-gain': lambda x: continuous(x),
        'capital-loss': lambda x: continuous(x),
        'hours-per-week': lambda x: continuous(x),
        'native-country': lambda x: onehot(x, options['native-country']),
        'income': lambda x: onehot(x.strip('.'), options['income']),
    }

    D = {}
    for f, phase in [(f_in_tr, 'training'), (f_in_te, 'test')]:
        dat = [s.strip().split(',') for s in open(f, 'r').readlines()]

        X = []
        Y = []
        A = []
        print(phase)

        for r in dat:
            row = [s.strip() for s in r]
            if MISSING_TOKEN in row and REMOVE_MISSING:
                continue
            if row in ([''], ['|1x3 Cross validator']):
                continue
            newrow, label, sens_att = parse_row(row, headers, headers_use)
            X.append(newrow)
            Y.append(label)
            A.append(sens_att)

        npX = np.array(X)
        npY = np.array(Y)
        npA = np.array(A)
        # npA = np.expand_dims(npA[:,1], 1)

        D[phase] = {}
        D[phase]['X'] = npX
        D[phase]['Y'] = npY
        D[phase]['A'] = npA

        print("npX.shape:",npX.shape)
        print("npY.shape:",npY.shape)
        print("npA.shape:",npA.shape)

    #should do normalization and centering
    ## for X
    print("do normalization and centring for X")
    mn = np.mean(D['training']['X'], axis=0)
    std = np.std(D['training']['X'], axis=0)
    D['training']['X'] = whiten(D['training']['X'], mn, std)
    D['test']['X'] = whiten(D['test']['X'], mn, std)
    ## for Y
    if target != "income":
        mn = np.mean(D['training']['Y'], axis=0)
        std = np.std(D['training']['Y'], axis=0)
        D['training']['Y'] = whiten(D['training']['Y'], mn, std)
        D['test']['Y'] = whiten(D['test']['Y'], mn, std)

    #should write headers file
    f = open(hd_file, 'w')
    i = 0
    for h in headers_use:
        if options[h] == 'continuous':
            f.write('{:d},{}\n'.format(i, h))
            i += 1
        elif options[h][0] == 'buckets':
            for b in buckets[h]:
                colname = '{}_{:d}'.format(h, b)
                f.write('{:d},{}\n'.format(i, colname))
                i += 1
        else:
            for opt in options[h]:
                colname = '{}_{}'.format(h, opt)
                f.write('{:d},{}\n'.format(i, colname))
                i += 1

    n = D['training']['X'].shape[0]
    shuf = np.random.permutation(n)
    valid_pct = 0.2
    valid_ct = int(n * valid_pct)
    valid_inds = shuf[:valid_ct]
    train_inds = shuf[valid_ct:]

    np.savez(f_out_np, x_train=D['training']['X'], x_test=D['test']['X'],
                y_train=D['training']['Y'], y_test=D['test']['Y'],
                attr_train=D['training']['A'], attr_test=D['test']['A'],
             train_inds=train_inds, valid_inds=valid_inds)