from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from deslib.static.oracle import Oracle
from deslib.util import diversity, datasets
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from sklearn.metrics import cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from rpy2.rinterface_lib import openrlib

import numpy as np
import subprocess
import csv, random, os

#os.environ['R_HOME'] = '/home/marcos/miniconda3/envs/l/lib/R'
os.environ['R_HOME'] = r"c:/python310/lib/R"
#os.environ['R_HOME'] = '/home/marcos/miniconda3/envs/l/lib/R'

import pandas as pd
from rpy2.robjects import pandas2ri

pandas2ri.activate()
import rpy2.robjects.packages as rpackages

ecol = rpackages.importr('ECoL')
import rpy2.robjects as robjects


def open_data(base_name, local_data):
    dataset_file = Marff.abre_arff(local_data + base_name + ".arff")
    X_data, y_data, dataset = Marff.retorna_instacias(dataset_file)
    # print(dataset_file['description'])
    dic = Marff.retorna_dic_data(dataset_file)
    del dic['data']
    return X_data, y_data, dataset, dic

def split_data(X_data, y_data):
    #
    indices = np.arange(len(X_data))
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(X_data, y_data, indices, test_size=0.5,
                                                                           stratify=y_data)
    X_test, X_vali, y_test, y_vali, id_test, id_vali = train_test_split(X_temp, y_temp, id_temp, test_size=0.5,
                                                                        stratify=y_temp)
    del X_data, y_data, X_temp, y_temp, id_temp

    return X_train, y_train, X_test, y_test, X_vali, y_vali, id_train, id_test, id_vali

def biuld_bags_stratify(y_train, X_train=None, X_data=None, y_data=None, ind=None, types="ind"):
    print('atencao pq nao e a funcao de bag oficial')
    X = []
    y = []
    if types == "sample":

        max_samples = int(round((len(y_train) * 0.5), 1))
        random_state = check_random_state(random.seed())
        indices = random_state.randint(0, len(y_train), max_samples)

        for i in indices:
            X.append(X_train[i])
            y.append(y_train[i])

        return X, y

    if types == "ind":
        indices = ind
        X_bag, xx, y_bag, yy, idx, idxx = train_test_split(X_train, y_train, indices, test_size=0.5, stratify=y_train)
        # print(indices)
        # print(idx)
        idx = idx.tolist()
        for i in idx:
            X.append(X_data[i])
            y.append(y_data[i])
        # exit(0)
        return X, y, idx

def biuld_bags(y_train, X_train=None, X_data=None, y_data=None, ind=None, tam=0.5, types="ind"):
    # constroi os bags de forma randomica, duas formas por indices do treino, ou por instancias
    X = []
    y = []
    if types == "sample":

        max_samples = int(round((len(y_train) * tam), 1))
        random_state = check_random_state(random.seed())
        indices = random_state.randint(0, len(y_train), max_samples)

        for i in indices:
            X.append(X_train[i])
            y.append(y_train[i])
 
        return X, y

    if types == "ind":

        idx = random.choices(ind, k=int(round((len(y_train) * tam), 1)))
        for i in idx:
            X.append(X_data[i])
            y.append(y_data[i])

        return X, y, idx

def generate_csv(dic):
    # grava um arquivo csv com o nome dos atributos na primeira linha, e das instancias, arquivo para script do R
    with open("/media/marcos/Data/Tese/teste.csv", 'w') as f:
        w = csv.writer(f)
        w.writerow(dic['class'])
        w.writerows(dic['data'])

def complexity_data():
    proc = subprocess.Popen(['Rscript /home/marcos/Documentos/new_8.r'],
                            stdout=subprocess.PIPE, shell=True)
    (cont_arq, err) = proc.communicate()
    cont_arq = (cont_arq.decode("utf-8"))
    # print(cont_arq)
    cont_arq = cont_arq.split()
    complex = []
    # x=[]
    for i in cont_arq:

        if (i[:3] == "ove" or i[:3] == "nei" or i[:3] == "dim" or i[:3] == "lin" or i[:3] == "bal" or i[:3] == "net"):
            # x.append(i)
            continue
        else:
            if i == "Inf":
                # print(type(i))
                i = 0.0
                print("ERROR")
            complex.append(float(i))
            # print(x)
    # exit(0)
    return complex

def complexity_data2(X_data, y_data):
    # comp = []
    dfx = pd.DataFrame(X_data, copy=False)
    dfy = robjects.IntVector(y_data)
    complex = ecol.complexity(dfx, dfy, type="class")
    complex = np.asarray(complex)
    return complex

def complexity_data3(X_data, y_data, group, types=None):
    dfx = pd.DataFrame(X_data, copy=False)
    dfy = robjects.IntVector(y_data)
    complex = np.array([])
    if group[0] == 'overlapping':
        over = ecol.overlapping(dfx, dfy, measures=types[0],summary='mean')
        over = np.asarray(over)
        complex = np.append(complex, over[0])
    if group[1] == "neighborhood":
        nei = ecol.neighborhood(dfx, dfy, measures=types[1],summary='mean')
        nei = np.asarray(nei)
        complex = np.append(complex, nei[0])
    
    complex = complex.tolist()
    del dfx, dfy
    return complex

def paralell_process(process):
    y = []
    x = os.popen('Rscript {}'.format(process)).read()
    x = x.split()
    # print(x)
    y.append(x)

    return y

def biuld_dic(X, y, dic):
    # constroi o dicionario para o construir o csv do R (instancias e classes)
    d = dict()
    g = []
    g.append(y)
    g = np.array(g)
    g.astype(int)
    d['class'] = dic['class']
    d['data'] = np.concatenate((X, g.T), axis=1)
    d['data'] = d['data'].tolist()

    return d

def p2_problem():
    p2 = datasets.make_P2([500, 500])
    data = dict()
    data['data'] = list()
    X = p2[0].tolist()
    y = p2[1].tolist()

    y = [int(i) for i in y]

    # print(y)

    for i in range(len(X)):
        X[i].append(y[i])

    data['data'] = X

    info = dict()
    info['description'] = "v"
    info['relation'] = 'p2'
    info['attributes'] = [('A1', 'REAL'), ('A2', 'REAL'), ('Class', (['0.0', '1.0']))]
    Marff.cria_arff(info, data, ["0.0", "1.0"], "/media/marcos/Data/Tese/Bases4/Dataset/", "P2")
    return data

def biuld_classifier_tree(X_train, y_train, X_val, y_val, X_test=None, y_test=None, score_train=False):
    '''
    se score_train false, retorna o score sobre o proprio treino, e a predicao sobre a validacao, caso contrario,
    retorna duas acuracias, treino e validacao mais a predicao sobre a validacao
    retorna um perceptron com sua acuracia e com a lista de predicao
    :param X_train: X do treino
    :param y_train: y do treino
    :param X_val: X valida ou teste
    :param y_val: y valida, ou teste
    :param X_test: retorna o predict e o segundo score
    :return: classificador, accuracia, lista de predicao
    '''
    # constroi os classificadores, e retorna classificador, score e predict
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    score = tree.score(X_val, y_val)
    if X_test!=None and y_test!=None and score_train==False:

        predict = tree.predict(X_test)

        return tree, score, predict

    elif (score_train):
        score2 = tree.score(X_test, y_test)
        predict = tree.predict(X_test)
        return tree, score, score2, predict

    else:

        return tree, score

def biuld_classifier(X_train, y_train, X_val, y_val, X_test=None, y_test=None, score_train=False):
    '''
    se score_train false, retorna o score sobre o proprio treino, e a predicao sobre a validacao, caso contrario,
    retorna duas acuracias, treino e validacao mais a predicao sobre a validacao
    retorna um perceptron com sua acuracia e com a lista de predicao
    :param X_train: X do treino
    :param y_train: y do treino
    :param X_val: X valida ou teste
    :param y_val: y valida, ou teste
    :param X_test: retorna o predict e o segundo score
    :return: classificador, accuracia, lista de predicao
    '''
    # constroi os classificadores, e retorna classificador, score e predict
    perc = Perceptron(n_jobs=4, max_iter=100, tol=1.0)
    perc.fit(X_train, y_train)
    score = perc.score(X_val, y_val)
    if X_test!=None and y_test!=None and score_train==False:

        predict = perc.predict(X_test)

        return perc, score, predict

    elif (score_train):
        score2 = perc.score(X_test, y_test)
        predict = perc.predict(X_test)
        return perc, score, score2, predict

    else:

        return perc, score

def biuld_classifier_tree(X_train, y_train, X_val, y_val, X_test=None, y_test=None, score_train=False):
    '''
    se score_train false, retorna o score sobre o proprio treino, e a predicao sobre a validacao, caso contrario,
    retorna duas acuracias, treino e validacao mais a predicao sobre a validacao
    retorna um perceptron com sua acuracia e com a lista de predicao
    :param X_train: X do treino
    :param y_train: y do treino
    :param X_val: X valida ou teste
    :param y_val: y valida, ou teste
    :param X_test: retorna o predict e o segundo score
    :return: classificador, accuracia, lista de predicao
    '''
    # constroi os classificadores, e retorna classificador, score e predict
    perc = DecisionTreeClassifier()
    perc.fit(X_train, y_train)
    score = perc.score(X_val, y_val)
    if X_test.all()!=None and y_test.all()!=None and score_train==False:

        predict = perc.predict(X_test)

        return perc, score, predict

    elif (score_train):
        score2 = perc.score(X_test, y_test)
        predict = perc.predict(X_test)
        return perc, score, score2, predict

    else:

        return perc, score

def biuld_classifier_naive(X_train, y_train, X_val, y_val, X_test=None, y_test=None, score_train=False):
    '''
        retorna um perceptron com sua acuracia e com a lista de predicao
        :param X_train: X do treino
        :param y_train: y do treino
        :param X_val: X valida ou teste
        :param y_val: y valida, ou teste
        :param X_test: retorna o predict e o segundo score
        :return: classificador, accuracia, lista de predicao
    '''
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    score = nb.score(X_val, y_val)
    if X_test != None and y_test != None and score_train == False:

        predict = nb.predict(X_test)
        return nb, score, predict

    elif (score_train):
        score2 = nb.score(X_test, y_test)
        predict = nb.predict(X_test)
        return nb, score, score2, predict

    else:

        return nb, score

def biuld_classifier_over(X, y, X_val, y_val, tam):
    '''
    retorna um perceptron com sua acuracia e com a lista de predicao
    :param X_train: X do treino
    :param y_train: y do treino
    :param X_val: X valida ou teste
    :param y_val: y valida, ou teste
    :param tam: tamanho da divisao do dataset de treino
    :return: classificador*, accuracia*, lista de predicao*
    a acuracia e o classificador sao referentes a parte do dataset de treino
    '''
    X_train = []
    y_train = []

    # print(np.around(len(y)*tam))
    lista_aleatoria = random.sample(range(0, len(y)), int(np.around(len(y) * tam)))
    for i in lista_aleatoria:
        X_train.append(X[i])
        y_train.append(y[i])
    # constroi os classificadores, e retorna classificador, score e predict
    perc = perceptron.Perceptron(n_jobs=7, max_iter=100, tol=10.0)
    perc.fit(X_train, y_train)
    score = perc.score(X_val, y_val)
    predict = perc.predict(X_val)

    # rms = sqrt(mean_squared_error(y_val, predict))
    return perc, score, predict

def voting_classifier(pool, X_val, y_val):
    #voting = EnsembleVoteClassifier(clfs=pool, voting='hard', refit=False)
    voting = EnsembleVoteClassifier(clfs=pool, voting='hard')

    voting.fit(X_val, y_val)
    result = voting.score(X_val, y_val)
    return result

def min_max_norm(dataset):
    """
    :param dataset: dataset [samples,features]
    :return: dataset normalizado por minmax
    """

    norm_list = list()
    min_value = np.min(dataset)
    max_value = np.max(dataset)

    if min_value == max_value:
        # print("entrei")
        for i in dataset:
            norm_list.append(0)
        return norm_list

    for value in dataset:
        tmp = (value - min_value) / (max_value - min_value)
        norm_list.append(tmp)

    return norm_list

def dispersion_norm(complexity):
    """
    :param complexity: listtas de complexidades
    :return: distancia media par a par das complexidades, normalizadas
    """
    result = []
    scaler = MinMaxScaler()
    scaler.fit(complexity)
    complexity = scaler.transform(complexity)
    dista = pairwise_distances(complexity, n_jobs=6)
    dista = dista.tolist()
    for i in dista:
        result.append(np.mean(i))
    return result

def dispersion(complexity):
    """
    :param complexity: listtas de complexidades
    :return: distancia media par a par das complexidades(biblioteca)
    """
    result = []
    dista = pairwise_distances(complexity, n_jobs=6)
    dista = dista.tolist()

    for i in dista:
        result.append(np.mean(i))
    return result

def dispersion2(complexity):
    """
    Atencao, foi refeita e nao testada dia 3-10-2019
    :param complexity: listas de complexidades
    :return: media das distancias feitas de forma manual
    """
    result = []

    complexity = list(complexity)
    # print(complexity)
    n = len(complexity) - 1
    for j in range(len(complexity)):
        dista = 0
        for l in range(len(complexity)):
            if (j == l):
                continue
            else:
                a = complexity[j]
                b = complexity[l]
                dista += np.sqrt(sum(((a - b)) ** 2 for a, b in zip(a, b)))
            dista = dista / n
        result.append((dista))

    return result

def dispersion_linear(complexity):
    # print(complexity)

    """
    :param complexity: listas de complexidades
    :return: media das distancias feitas de forma manual a-b normalizadas
    """
    result = []
    result1 = []

    complexity = list(complexity)
    complexity = np.array(complexity)
    # print((complexity))
    n = (len(complexity)) - 1
    complexity = complexity.T

    # exit(0)
    for i in complexity:
        dist = []
        for j in range(len(i)):
            dista = 0
            for l in range(len(i)):
                if (j == l):
                    continue
                else:
                    dista += abs(i[j] - i[l])
            dist.append((dista) / n)
        result.append(dist)
    result = np.array(result)
    # print(result)
    for i in result:
        r = min_max_norm(i)

        result1.append(r)
    result1 = np.array(result1)
    del result, r, dist, complexity
    result1 = result1.T
    result1 = result1.tolist()
    #print(result1)

    return result1

def diversitys(y_test, predicts):
    q_test = []
    double_faults = []
    for i in range(len(predicts)):
        db = []
        for j in range(len(predicts)):
            if i == j:
                continue
            else:
                #  q.append(diversity.Q_statistic(y_test,predicts[i],predicts[j]))
                db.append(diversity.double_fault(y_test, predicts[i], predicts[j]))
                #coloquei um paramentro novo na funcao de retorno _process_predictions
        double_faults.append(np.mean(db))

    return double_faults

def diversity_kapa(y_test, predicts):
    size=len(y_test)
    kappa_mean=[]
    kappa_std=[]
    kappa_all=[]
    for i in range(len(predicts)):
        k=[]
        for j in range(len(predicts)):
            if i == j:
                continue
            else:
                a, b, c, d = diversity._process_predictions(y_test,predicts[i],predicts[j],True)
                if (a==size):
                    k_p=1
                else:
                    m=a+b+c+d
                    q1=np.around((a+d)/m,2)
                    q2=np.around((((a+b)*(a+c))+((c+d)*(b+d)))/(m*m),2)
                    k_p=np.around((q1-q2)/(1-q2),2)
                k.append(k_p)
        kappa_all.append(k)
        kappa_mean.append(np.mean(k))
        kappa_std.append(np.std(k))

    return kappa_mean, kappa_std, kappa_all

def biuld_csv_result(complexity_result, score, Q_test, Df, disp):
    global base_name
    header = ['overlapping.F1', 'overlapping.F1v', 'overlapping.F2', 'overlapping.F3', 'overlapping.F4',
              'neighborhood.N1', 'neighborhood.N2', 'neighborhood.N3', 'neighborhood.N4', 'neighborhood.T1',
              'neighborhood.LSCAvg', 'linearity.L1', 'linearity.L2', 'linearity.L3', '000000.T2', 'dimensionality.T3',
              'dimensionality.T4', 'balance.C1', 'balance.C2', 'network.Density', 'network.ClsCoef', 'network.Hubs',
              'Score', 'Q_test', 'DoubleFault', 'Disper']
    for i in range(len(complexity_result)):
        complexity_result[i].append(score[i])
        complexity_result[i].append(Q_test[i])
        complexity_result[i].append(Df[i])
        complexity_result[i].append(disp[i])
    complexity_result = np.round(complexity_result, 3)
    with open("/media/marcos/Data/Tese/Resultados_" + base_name + ".csv", 'a') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(complexity_result)

def save_bag(inds, types, local, base_name, iteration):
    if types == 'validation':
        # print('entreivali')
        if (os.path.exists(local + "Validacao/" + str(iteration)) == False):
            os.system("mkdir -p " + local + "/" + str(iteration))
        with open(local + "/" + str(iteration) + "/" + base_name + ".csv", 'w') as f:
            # print('entreivali')
            w = csv.writer(f)
            w.writerow(inds)

    if types == "test":
        if (os.path.exists(local + "Teste/" + str(iteration)) == False):
            os.system("mkdir -p " + local + "/" + str(iteration))
        with open(local + "/" + str(iteration) + "/" + base_name + ".csv", 'w') as f:
            w = csv.writer(f)
            w.writerow(inds)

    if types == "train":
        if (os.path.exists(local + "Treino/" + str(iteration)) == False):
            os.system("mkdir -p " + local + "/" + str(iteration))
        with open(local + "/" + str(iteration) + "/" + base_name + ".csv", 'w') as f:
            w = csv.writer(f)
            w.writerow(inds)

    if types == "bags":
        if (os.path.exists(local + "Bags/" + str(iteration)) == False):
            os.system("mkdir -p " + local + "/" + str(iteration))
        with open(local + "/" + str(iteration) + "/" + base_name + ".csv", 'a') as f:
            w = csv.writer(f)
            w.writerow(inds)

def oracle(poll, X, y, X_test, y_test):
    orc = Oracle(poll)
    orc.fit(X, y)
    # orc.predict(X_test,y_test)
    return orc.score(X_test, y_test)

def routine_save_bags(local_dataset, local, base_name, iteration):
    # rotina para criar treino teste e validcao alem dos 100 bags, local e onde esta o dataset orig
    X_data, y_data, dataset, dic = open_data(base_name, local_dataset)
    X_train, y_train, X_test, y_test, X_vali, y_vali, id_train, id_test, id_vali = split_data(X_data, y_data)
    save_bag(id_train, 'train', local + "Treino/", base_name, (iteration))
    save_bag(id_vali, 'validation', local + "Validacao/", base_name, str(iteration))
    save_bag(id_test, 'test', local + "Teste/", base_name, str(iteration))
    for i in range(0, 100):
        #       X_bag, y_bag, id = biuld_bags_stratify(y_train,X_train=X_train, X_data=X_data, y_data=y_data, ind=id_train, types="ind")
        X_bag, y_bag, id = biuld_bags(y_train, X_data=X_data, y_data=y_data, ind=id_train, types="ind")
        # print(len(id))
        id.insert(0, i)
        save_bag(id, 'bags', local + "/Bags/", base_name, str(iteration))
    return X_train, y_train, X_test, y_test, X_vali, y_vali

def open_bag(local_bag, base_name):
    bags = dict()
    bags['nome'] = list()
    bags['inst'] = list()

    if not os.path.exists(local_bag + base_name + '.csv'):
        open(local_bag + base_name + '.csv', 'w').close()

    with open(local_bag + base_name + '.csv', 'w') as f:
        reader = csv.reader(f)
        indx = list(reader)
        # print(indx)
    for i in indx:
        bags['nome'].append(i[0])
        bags['inst'].append(i[1:])

    return bags

def biuld_x_y(indx_bag, X, y):
    '''
    Recebe o indice de instancias de um bag
    :param indx_bag:
    :param vet_classes: false, retorna o vetor de classes
    :return: X_data, y_data
    '''
    # global nome_base, classes, caminho_base
    X_data = []
    y_data = []
    # print(len(X))
    #print(len(X))
    for i in indx_bag:
       # print(i)
        X_data.append(X[int(i)])
        y_data.append(y[int(i)])
    return X_data, y_data

def open_test_vali(local, base_name, iteration):
    # print("open bag/teste_vali mudar as saidas")
    if not os.path.exists(local + "Teste/" + str(iteration) + "/" + base_name + '.csv'):
        open(local + "Teste/" + str(iteration) + "/" + base_name + '.csv', 'w').close()

    with open(local + "Teste/" + str(iteration) + "/" + base_name + '.csv', 'r') as f:
        reader = csv.reader(f)
        teste = list(reader)
    
    if not os.path.exists(local + "Validacao/" + str(iteration) + "/" + base_name + '.csv'):
        open(local + "Validacao/" + str(iteration) + "/" + base_name + '.csv', 'w').close()

    with open(local + "Validacao/" + str(iteration) + "/" + base_name + '.csv', 'r') as f:
        reader = csv.reader(f)
        vali = list(reader)
        
    return teste[0], vali[0]

def open_training(local, base_name, iteration):
    with open(local + "Treino/" + str(iteration) + "/" + base_name + '.csv', 'r') as f:
        reader = csv.reader(f)
        treino = list(reader)
    return treino[0]

def open_data_jonathan(local,name):

    base = open(local+name+".arff", "r")
    #print(base.readline())
    sample = []
    y = []
    for i in base:
        x = i.split(";")
        y.append(x[0])
        x = x[1:-1]
        x = [float(elem) for elem in x]
        sample.append(x)
    y = [int(elem) for elem in y]
    X = np.array(sample)
    y = np.array(y)
    return X,y

def generate_bags(X_train, y_train, tam_bags=0.5, nr_bags=100):
    indices = np.arange(len(X_train))
    bags = dict()
    bags['name'] = list()
    bags['inst'] = list()
    for i in range(0, nr_bags):
        X_bag, X_temp, y_bag, y_temp, id_bag, id_temp = train_test_split(X_train, y_train, indices, test_size=tam_bags, stratify=y_train)
        bags['name'].append(i)
        bags['inst'].append(id_bag.tolist())    
        del  X_bag, X_temp, y_bag, y_temp, id_bag, id_temp
    return bags                                                                     



def main():
    #p2_problem()
    comp = []
    # p2_problem()
   # import Marff as mf
    # for i in range(10):
    #    wine=mf.abre_arff("/home/marcos/Documentos/wine.arff")
    #   X,y,_=mf.retorna_instacias(wine,True)
    #  grupo = ["overlapping", 'neighborhood', '', '', '', '']
    # comp.append(complex_data4(X,y,grupo))
    # x=[[.0, .2, 1.],[.3,2,1],[.5,1,3],[1.,5,6]]
    # x=np.array(x)
    # x=x.T
    # #x=x.tolist()
    # print(x)
    # y=min_max_norm(x)
    # print(y)
    # for i in x:
    # print(min_max_norm(i))

    # #import time
    # base_name='Wine'
    # x=[]
    # repeticao="4/"
    # local_dataset = "/media/marcos/Data/Tese/Bases2/Dataset/"
    # #local = "/media/marcos/Data/Tese/Bases3/"
    # #caminho_base = "/media/marcos/Data/Tese/Bases2/"
    # #cpx_caminho = "/media/marcos/Data/Tese/Bases3/Bags/"
    # X_data, y_data, dataset, dic =open_data(base_name,local_dataset)
    # result=complexity_data3(X_data,y_data,["overlapping",'neighborhood','','','',''],["F4",'N1','','','',''])
    #
    # #print(result)
    # x=[[1,2],[7,5],[3,5]]
    # print(dispersion2(x),dispersion(x))


  #  p2_problem()


if __name__ == "__main__":
    main()

