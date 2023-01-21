# pool_generate_marcos
Algoritmo para exploração da diversidade usando complexidade de dados, para pool generation


poolGeneretin Class

Gera os bags dos dados com base na diversidade e na complexidade dos dados. 
```python
 poolGenetate()
```
### 🛠 Tecnologias

As seguintes ferramentas foram usadas na construção do projeto:

- [Python](https://www.python.org/)
- [R](https://www.r-project.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [deap](https://deap.readthedocs.io/en/)
### Pré-requisitos

Antes de começar, você vai precisar ter instalado em sua máquina as seguintes ferramentas:
[Git](https://git-scm.com), [Python](https://www.python.org/), [R](https://www.r-project.org/). 
Além disto é bom ter um editor para trabalhar com o código como [VSCode](https://code.visualstudio.com/)

### 🎲 Rodando o Pool Generare

```bash
# Clone este repositório
$ git clone https://github.com/matheusMrs07/pool_generate_marcos.git

# Acesse a pasta do projeto no terminal/cmd
$ cd pool_generate_marcos

# Instalar os pacotes necessarios para usar o projeto
$ pip install -r requirements.txt

```

Abrir o arquivo `sample.ipynb` para ver um exemplo da utilização do projeto.


### Configurações do Projeto 

Para rodar esse projeto, você vai precisar configurar as seguinte variáves: 

- `group`: Grupo de Complexidades usadas para geração dos bags.
    ``` python
    pool_generate.group = ["overlapping", 'neighborhood', '', '', '', '']
    ```
- `types`: indices de complexidades que serão usados.
     ``` python
    pool_generate.types = ["F1", 'T1', '', '', '', '']
    ```
- method_disperse = True

- fit_value1 = 1.0
- fit_value2 = 1.0
- fit_value3 = -1.0

- nr_generation = 19
- nr_individual = 100
- nr_pop=100

- proba_crossover = 0.99
- proba_mutation = 0.01

- nr_child=100
- cont_crossover = 1
- iteration=21 #numero de variações de bags 
- dist_temp=0

- jobs = 8
- stop_criteria="maxdistance"#maxacc
- classifier="tree"#tree,perc
- save_info=False
- seq = -1
- base_name = "Base1"
- file_out = "maxdistanceree"

- tem2 = []

- acc_temp = 0
- tam_bags = 0.5
- nr_bags = 100
- file_out = "isto_e_um_teste"

- local = "saida"

- c = []
- bags_saved = []

### Bibliografia

Para entender o funcinamento e analizar os resultados dessa abordagem acesse o artigo: [Exploring diversity in data complexity and classifier decision spaces for pool generation](https://doi.org/10.1016/j.inffus.2022.09.001)