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

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from joblib import  Parallel, delayed

import collections, Graficos_ga as graf

from traitlets import ClassBasedTraitType


import Cpx


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


class poolGeneration():
    def __init__(self):
        self.group = ["overlapping", 'neighborhood', '', '', '', '']
        self.types = ["F1", 'T1', '', '', '', '']

        self.method_disperse = True

        self.fit_value1 = 1.0
        self.fit_value2 = 1.0
        self.fit_value3 = -1.0

        self.nr_generation = 19
        self.nr_individual = 100
        self.nr_pop=100

        self.proba_crossover = 0.99
        self.proba_mutation = 0.01

        self.nr_child=100
        self.cont_crossover = 1
        self.iteration=21 #numero de variações de bags 
        self.dist_temp=0

        self.jobs = 8
        self.stop_criteria="maxdistance"#maxacc
        self.classifier="tree"#tree,perc
        self.save_info=False
        self.seq = -1
        self.base_name = "Base1"
        # self.file_out = "maxdistanceree"

        self.tem2 = []

        self.acc_temp = 0
        self.tam_bags = 0.5
        self.nr_bags = 100
        self.file_out = "isto_e_um_teste"

        self.local = "saida"

        self.c = []
        self.bags_saved = []
        

    def generate_bags(self, X_train, y_train, tam_bags=0.5, nr_bags=100):
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

    def sequencia(self):
        self.seq += 1 
        return self.seq

    def split_data(self, X_data, y_data):
        self.X = X_data
        self.y = y_data

        indices = np.arange(len(X_data))
        self.X_train, X_temp, self.y_train, y_temp, self.id_train, id_temp = train_test_split(X_data, y_data, indices, test_size=0.5,
                                                                            stratify=y_data)
        self.X_test, self.X_vali, self.y_test, self.y_vali, self.id_test, self.id_vali = train_test_split(X_temp, y_temp,id_temp, test_size=0.5,
                                                                            stratify=y_temp)

        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_vali, self.y_vali, self.id_train, self.id_test, self.id_vali

    def get_complexity(self, first_evaluate=False, population=None):

        dist = dict()
        dist['name'] = list()
        dist['dist'] = list()
        dist['diver'] = list()
        dist['score'] = list()
        dist['score_g'] = list()
    
        if (first_evaluate == True and self.generation == 0):
            dist['name'] = self.pop
            r = []
            #r = Parallel(n_jobs=self.jobs)(delayed(self.parallel_distance2)(i, self.bags, self.group, self.types) for i in range(len(dist['name'])))
            for i in range(len(dist['name'])):
                r.append(self.parallel_distance2(i, self.bags, self.group, self.types))
            
            c, score,  pred, pool = zip(*r)

            self.c = c
            #print(c)
            
        elif (first_evaluate == False and population == None):
            begin = self.name_individual - self.nr_individual
            for i in range(begin, self.name_individual):
                x = []
                x.append(i)
                dist['name'].append(x)
            # r = Parallel(n_jobs=self.jobs)(
            #     delayed(self.parallel_distance2)(j, self.bags, self.group, self.types) for j in range(100, self.nr_individual + 100))
            
            r = []
            for j in range(100, self.nr_individual + 100):
                r.append(self.parallel_distance2(j, self.bags, self.group, self.types))

            c, score, pred, pool = zip(*r)
            self.c = c

        elif (population != None):
            dist['name'] = population
            indices = []
            for i in population:
                indices.append(self.bags['name'].index(i[0]))
            # r = Parallel(n_jobs=self.jobs)(delayed(self.parallel_distance2)(i,self.bags, self.group, self.types) for i in indices)
            
            r = []
            for i in indices: 
                r.append(self.parallel_distance2(i,self.bags, self.group, self.types))
            c, score, pred, pool = zip(*r)
            self.c = c

        dist['dist'] = Cpx.dispersion_linear(c)
        dist['score'] = score
        d = self.diversity_ga(pred, self.y_val)
        dist['diver'] = Cpx.min_max_norm(d)
        dist['score_g']=Cpx.voting_classifier(pool, self.X_val, self.y_val)
        
        self.dist = dist
        return

    def diversity_ga(self, pred, y):
        pred=np.array(pred)
        d =Cpx.diversitys(y, pred)
        return d


    def parallel_distance2(self, i, bags, group, types):
        """

        :param i: lista de indices do bag a ser testado
        :param bags: lista com todos os bags
        :param grupo: lista com o nome dos grupos de complexidades ex: [overllaping,,,,,]
        :param tipos: lista com o nome da complexidade
        :return: listas de complexidade, score do prorpio bag, score sobre a validacao, e a predicao sobre a validacao
        """
        indx_bag1 = bags['inst'][i]
        X_bag, y_bag = self.biuld_bags(indx_bag1)
        cpx = (Cpx.complexity_data3(X_bag, y_bag, group, types))
        
        
        if self.classifier=="perc":
            estimator, score, pred = Cpx.biuld_classifier(X_bag, y_bag, X_bag, y_bag, self.X_val, self.y_val)
        elif self.classifier=="tree":
            estimator, score, pred = Cpx.biuld_classifier_tree(X_bag, y_bag, X_bag, y_bag, self.X_val, self.y_val)
        return cpx,  score,  pred, estimator

    def biuld_bags(self, indx_bag):
        
        '''
        Recebe o indice de instancias de um bag
        :param indx_bag:
        :param vet_classes: false, retorna o vetor de classes
        :return: X_data, y_data
        '''
        X_data = []
        y_data = []
        for i in indx_bag:
            X_data.append(self.X_train[int(i)])
            y_data.append(self.y_train[int(i)])
        return X_data, y_data


    def evaluate_linear_dispersion(self, ind1):
        dist = self.dist
        for i in range(len(dist['name'])):
            if (dist['name'][i][0] == ind1[0]):
                dst1 = dist['dist'][i][0]
                dist2 = dist['dist'][i][1]
                diver=dist['diver'][i]
                break
        return dst1, dist2, diver,

    def crossover(self, ind1, ind2):
        '''
        Para funcionar os bags devem ter o mesmo tamanho
        :param ind1:
        :param ind2:
        :return:
        '''
            
        individual = False
        indx = self.bags['name'].index(ind1[0])
        indx2 = self.bags['name'].index(ind2[0])
        indx_bag1 = self.bags['inst'][indx]
        indx_bag2 = self.bags['inst'][indx2]
        _, y_data = self.biuld_bags(indx_bag1)
        cont = 0

        while (individual != True):
            ind_out1 = self.short_cross(y_data, indx_bag1, indx_bag2)
            individual = self.verify_bag(ind_out1)
            cont = cont + 1
            if cont == 30:
                print("Stratification error")
                exit(0)

        ind1[0] = self.name_individual
        ind2[0] = self.name_individual

        self.bags['name'].append(self.name_individual)
        self.bags['inst'].append(ind_out1)
        self.name_individual += 1

        if (self.method_disperse == True):
            self.cont_crossover = self.cont_crossover + 1
            if (self.cont_crossover == self.nr_individual + 1):
                self.cont_crossover = 1
                self.get_complexity(first_evaluate=False, population=None)

        return creator.Individual(ind1), creator.Individual(ind2)
    

    def short_cross(self, y_data, indx_bag1, indx_bag2):
        beginning = finish = 0
        ind_out1 = []
        while (y_data[beginning] == y_data[finish]):
            beginning = random.randint(0, len(y_data) - 1)
            finish = random.randint(beginning, len(y_data) - 1)
        for i in range(len(y_data)):
            if (i <= beginning or i >= finish):
                ind_out1.append(indx_bag1[i])
            else:
                ind_out1.append(indx_bag2[i])
        return ind_out1

    def verify_bag(self, ind_out):

        classes = collections.Counter(self.y_train)
        _, y = self.biuld_bags(ind_out)
        counter = collections.Counter(y)
        if len(counter.values()) == len(classes) and min(counter.values())  >= 2:
            return True
        else:
            return False


    def mutation(self, ind):
        ind_out = []
        indx = self.bags['name'].index(ind[0])
        indx_bag1 = self.bags['inst'][indx]
        X, y_data = self.biuld_bags(indx_bag1)
        inst = 0
        inst2 = len(y_data)

        if (self.generation == 0 and self.off == []):
            ind2 = random.randint(0, 99)
        else:
            ind2 = random.sample(self.off, 1)
            ind2 = ind2[0]
        indx2 = self.bags['name'].index(ind2)
        indx_bag2 = self.bags['inst'][indx2]
        X2, y2_data = self.biuld_bags(indx_bag2)

        while y_data[inst] != y2_data[inst2 - 1]:
            inst = random.randint(0, len(y_data) - 1)
        for i in range(len(indx_bag1)):
            if (i == inst):
                ind_out.append(indx_bag2[i])
            else:
                ind_out.append(indx_bag1[i])

        self.bags['name'].append(self.name_individual)
        self.bags['inst'].append(ind_out)
        ind[0] = self.name_individual
        self.name_individual += 1
        if (self.method_disperse == True):
            self.cont_crossover = self.cont_crossover + 1
            if (self.cont_crossover == self.nr_individual + 1):
                self.cont_crossover = 1
                self.get_complexity(first_evaluate=False, population=None)

        return ind,

    def the_function(self, population, gen, fitness):

        generation = gen
        if self.save_info:
            if self.repetition == 1:
                self.save_generation_info(generation,fitness, self.c)
        self.off = []
        base_name = self.base_name + str(generation)
        bags_ant = self.bags
        bags = dict()
        bags['name'] = list()
        bags['inst'] = list()
        for j in population:
            # print(bags_ant)
            indx = bags_ant['name'].index(j[0])
            bags['name'].append(bags_ant['name'][indx])
            bags['inst'].append(bags_ant['inst'][indx])
        del bags_ant
        for i in range(len(population)):
            self.off.append(population[i][0])
        if self.stop_criteria=="maxdistance":
            self.max_distance(fitness, generation=self.generation, population=self.off, bags=bags)
        elif self.stop_criteria =="maxacc":
            self.max_acc(self.dist['score_g'], generation=self.generation, population=self.off, bags=bags)
        if generation == self.nr_generation:
            if self.stop_criteria == "maxdistance" or self.stop_criteria=="maxacc":
                self.save_bags(self.pop_temp, self.bags_temp, self.gen_temp, self.base_name)
            else:
                self.save_bags(self.off, bags, base_name=self.base_name)
        if (self.method_disperse == True and generation != self.nr_generation):
            self.get_complexity(population=population)
        return population

    def save_bags(self, pop_temp, bags_temp, gen_temp=None, base_name=None, type=0,generations_escolhida="x"):
    
        if type==0:

            for j in pop_temp:
                name = []
                indx = self.bags['name'].index(j)
                nm = self.bags['inst'][indx]
                name.append(self.bags['name'][indx])
                name.extend(nm)
                self.bags_saved.append(name)
                Cpx.save_bag(name, 'bags', self.local + "/Bags", base_name + self.file_out, self.iteration)

        elif(type==1):
            x = open(generations_escolhida, "a")
            x.write(base_name + ";" + str(gen_temp) + "\n")
            x.close()
            for j in pop_temp:
                name = []
                indx = bags_temp['name'].index(str(j))
                nm = bags_temp['inst'][indx]
                name.append(bags_temp['name'][indx])
                name.extend(nm)
                if self.classifier=="perc":
                    Cpx.save_bag(name, 'bags', self.local + "/Bags", base_name + self.file_out, self.iteration)
                elif self.classifier=="tree":
                    Cpx.save_bag(name, 'bags', self.local + "/tree/Bags", base_name + self.file_out, self.iteration)

        elif type==2:
            x = open(generations_escolhida, "a")
            x.write(base_name + ";" + str(gen_temp) + "\n")
            x.close()
            for j in pop_temp:
                name = []
                indx = bags_temp['name'].index(str(j))
                nm = bags_temp['inst'][indx]
                name.append(bags_temp['name'][indx])
                name.extend(nm)
                Cpx.save_bag(name, 'bags', self.local + "tree/Bags", base_name + self.file_out, self.iteration)


    def save_generation_info(self, generation, fitness, complexity):
        '''
        :param generation:
        :param fitness:
        :param complexidade:
        :return:
        :salva em arquivo todos os dados da geracao o ultimo comando salva um grafico dos 2 primeiros fitness (1d) com o nome do arquivo de saida
        '''

        temp = np.array(complexity)
        temp=temp.T
        k = []
        with open(self.base_name +str(generation)+ self.file_out+'.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';')
            for i in range(len(complexity)):
                k=complexity[i]
                k.append(fitness[0][i])
                k.append(fitness[1][i])
                if fitness[2]:
                    k.append(fitness[2][i])
                spamwriter.writerow(k)
            m1 = np.std(temp[0])
            m2 = np.std(temp[1])
            m3 = np.std(fitness[0])
            m4 = np.std(fitness[1])
            m6 = np.mean(temp[0])
            m7 = np.mean(temp[1])
            m8 = np.mean(fitness[0])
            m9 = np.mean(fitness[1])
            if fitness[2]:
                m10 = np.mean(fitness[2])
                m5 = np.std(fitness[2])
                tem=[m1,m2, m3, m4, m5, m6, m7,m8, m9, m10]
                self.tem2.append(tem)
                spamwriter.writerow(tem)
                spamwriter.writerow('\n')
                del m5, m10
            else:
                tem = [m1, m2, m3, m4, m6, m7, m8, m9]
                self.tem2.append(tem)
                spamwriter.writerow(tem)
                spamwriter.writerow('\n')
            if generation == self.nr_generation:
                for i in self.tem2:
                    spamwriter.writerow(i)
        graf.grafico_disper(self.base_name, ["Disp complexity overlapping", "Disp complexity neighborhood", "Disp diversity"], fitness[0], fitness[1], valor3=fitness[2], legend="Lithuanian", i= self.repeticao, gr=generation,pasta= self.arquivo_de_saida)
        del tem, k, m1, m2, temp,m3, m4, m6, m7,m8, m9

    def max_distance(self, fitness, generation=None, population=None, bags=None):
        '''
        :param fit1: fittnes 1
        :param fit2:
        :param fit3:
        ideal para distancia linear
        '''
        if fitness[2]:
            dist_dist_media = np.mean(Cpx.dispersion(np.column_stack([fitness[0], fitness[1], fitness[2]])))
        else:
            dist_dist_media = np.mean(Cpx.dispersion(np.column_stack([fitness[0], fitness[1]])))
        if dist_dist_media > self.dist_temp:
            self.dist_temp = dist_dist_media
            self.pop_temp = population
            self.gen_temp = generation
            self.bags_temp = bags

    def max_acc(self, acc,generation=None, population=None, bags=None):
        '''
        :param fit1: fittnes 1
        :param fit2:
        :param fit3:
        :param population: popoluacao atual geralmente o off
        :param bags: bags atuais
        :return:
        ideal para distancia linear
        '''

        if acc > self.acc_temp:
            self.acc_temp = acc
            self.pop_temp = population
            self.gen_temp = generation
            self.bags_temp = bags

    def generate(self, X_train, y_train, X_val, y_val, iteration = 20):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.iteration = iteration

        for t in range(1, self.iteration):

            self.name_individual = 100
            self.off = []
            self.seq = -1
            self.repetition = t



            self.generation = 0
            self.bags = self.generate_bags(self.X_train, self.y_train, self.tam_bags, self.nr_bags)
            creator.create("FitnessMult", base.Fitness, weights=(self.fit_value1, self.fit_value2, self.fit_value3))
            creator.create("Individual", list, fitness=creator.FitnessMult)
            toolbox = base.Toolbox()
            toolbox.register("attr_item", self.sequencia)
            toolbox.register("individual", tools.initRepeat, creator.Individual,
                            toolbox.attr_item, 1)
            population = toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            self.pop = toolbox.population(n=self.nr_pop)
            if self.method_disperse == True:
                self.get_complexity(first_evaluate=True)

            toolbox.register("evaluate", self.evaluate_linear_dispersion)
            toolbox.register("mate", self.crossover)
            toolbox.register("mutate", self.mutation)
            toolbox.register("select", tools.selNSGA2)
            self.pop = algorithms.eaMuPlusLambda(self.pop, toolbox, self.nr_child, self.nr_individual, self.proba_crossover, self.proba_mutation,
                                                self.nr_generation,
                                                    generation_function=self.the_function)




