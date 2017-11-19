
# coding: utf-8

# In[1]:


#!/usr/bin/python
import sys
import pickle
import os
sys.path.append("../tools/")


# In[2]:


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[3]:


from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
get_ipython().magic(u'matplotlib inline')


# # Funções definidas
# Funções foram criadas ao longo do código para facilitar a manipulação dos dados. Todas as funções no início do código.

# In[4]:


def F_RetirarNaNs(data_dict):
    '''
    Retira os NaNs 
    data_dict: dicionário com diversas features para cada pessoa
    return: dicionário de entrada, exceto as features que possuem valor NaN
    '''
    
    for k1, v1 in data_dict.iteritems():
        data_dict[k1] = {key:value for key, value in v1.iteritems() if value != 'NaN'}
    
    return data_dict


# In[5]:


def F_FeatureScalling(data_dict, not_scale, features):
    '''
    Cria nova escala para features de cada pessoa no dicionário de entrada
    data_dict: dicionário com diversas features para cada pessoa
    not_scale: features que não devem ser alteradas
    return: tuple com features do dicionário em nova escala e um dicionário com o conversor de escalas
    '''
    
    from sklearn.preprocessing import MinMaxScaler
    
    #Criar escala para cada feature
    scale_feature = dict()
    for feature in features: 
        if not feature in not_scale:
            scale_feature[feature] = MinMaxScaler()
            scale_feature[feature].fit_transform(np.array([data_dict[key][feature]
                                                for key in data_dict.keys()
                                                if feature in data_dict[key].keys()]).reshape(-1,1))
    
    #Aplicar escala sobre os campos de interesse
    for k1, v1 in data_dict.iteritems():
        for key, value in v1.iteritems():
            if key in features and not key in not_scale:
                data_dict[k1][key] = scale_feature[key].transform(np.array(value).reshape(-1,1))[0][0]
            else:
                data_dict[k1][key] = value
    
    return data_dict, scale_feature


# In[6]:


def F_OrdemFeatures(data_dict, features):
    '''
    Retorna uma lista com as features ordenadas por valor
    data_dict: dicionários com as features
    features: features a serem ordenadas
    return: lista com ordem das features
    '''
    
    ordem = {}
    
    for feature in features:
        
        aux = {keys: values[feature] for keys, values in data_dict.items() if feature in values.keys()}
        
        ordem[feature] = sorted(aux.keys(), key=lambda x:data_dict[x][feature])
        
    return ordem


# In[7]:


def F_PlotOutliers(test_list, ordem):
    '''
    Criar gráfico com valores ordenados
    test_list: lista com os dados
    ordem: ordem de plotagem 
    return: maiores valores em cada gráfico (análise de outliers)
    '''
    
    outliers_dict = {}
    n_dados = len(test_list)
    
    f, gp = plt.subplots(n_dados,1, sharex= True)

    f.set_figheight((2.5*n_dados)//2+1)
    f.set_figwidth(15)

    for jj, feature in enumerate(test_list):
        max_value = 0
        person = ''
        
        try:

            for ii, point in enumerate(ordem[feature]):
                if feature in data_dict[point]:

                    salary = data_dict[point][feature]

                    if max_value < salary:
                        max_value = salary
                        person = point

                    #Registrando os valores extremos encontrados
                    outliers_dict[feature] = [max_value, person]

                    gp[jj].scatter(ii, salary)

                gp[jj].set_title(feature)
        except:
            pass

    plt.show()
    
    return outliers_dict


# In[8]:


def F_PessoaEmail(dir_, address):
    
    '''
    Procura no repositório dir_ o endereço de destino contido no email
    dir_: diretório dos emails
    address: email procurado
    '''
    lista = []
    if email in address.keys():
        with open('{}/{}'.format(dir_, address[email], 'r')) as fl:
            lista = [line.split('/')[2] for line in fl]
    
    return lista


# In[9]:


def F_ExtraiPoisAbreviados(data_dict):
    '''
    Extrair os POis e abrevia os nomes pela regra de sobrenome-Nome(primeira letra), exemplo, Kenneth Lee Lay para lay-k
    data_dict: dicionário com dados das pessoas
    return: dictionary com nomes convertidos por pessoa
    '''

    aux_list = {}
    for ii in [key for key, value in data_dict.items() if value['poi']==True]:
        string = ii.split(' ')
        name = string[0]

        second_name = string[1][0] if len(string) in [3,2] else string[2][0]
        
        try:
            aux_list[data_dict[ii]['email_address']] = '-'.join([name, second_name]).lower()
        except:
            print 'Erro na chave {key}'.format(key=ii)
        
    return aux_list


# In[10]:


def F_StatMsg(email, aux_list):
    '''
    Retorna a quantidade de e-mails trocados com POis
    email: dictionary com os destinos dos emails enviados por uma pessoa
    return: dictionary com contagem do total de emails enviados para POis para cada indivíduo
    '''
    
    count, ratio = {}, {}
    for key in email.keys():
        
        aux = aux_list.copy()
        if key in aux.keys():
        #Não contar a própria pessoa -- acontece quando a pessoa é um poi
            aux.pop(key)
        
        #Somente os que são poi
        lista = [ii for ii in email[key] if ii in aux.values()]
        
        #total de poi com troca de mensagens
        count[key] = len(set(lista))
        
        #razao entre mensagens com poi e total de mensagens
        if len(email[key])>0:
            ratio[key] = len(lista)/float(len(email[key]))
        else:
            ratio[key]= 0
    
    return count, ratio


# In[11]:


def F_VerificarExisteFeature():
    '''Verificar se existe chave sem alguma característica 
        return: key e feature faltante, além da quantidade de keys que precisam ser corrigidas
    '''
    
    person_feature = defaultdict(list)
    
    all_features = {key for key, value in my_dataset.items() if all(ii == features_list for ii in value.keys())}
    all_persons = {key for key in my_dataset.keys()}
    
    #procura as pessoas que não tem todas as features e encontra a features faltante
    for jj in all_persons - all_features:
        for ii in features_list:
            if not ii in my_dataset[jj].keys():
                person_feature[jj].append(ii)
    
    print 'Quantidade de pessoas com falta de informações, ', len(person_feature)
    return person_feature


# As features selecionadas tem relação com ganhos financeiros e gastos. Novas features serão adicionadas para criar relações de network entre as pessoas no dataset, através dos e-mails.

# In[12]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments', 'bonus', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'restricted_stock','shared_receipt_with_poi','ratio_to','ratio_from'] 


# In[13]:


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# # Cleaning Data
# Temos que alguns campos estão como 'NaN', logo estes campos não serão considerados neste momento para facilitar a análise de outliers (análise gráfica) e outras análises que a falta de dados pode enviesar. Se os campos com 'NaN' estiverem na feature_list eles serão futuramente adicionados com valor 0.

# Antes de retirar os valores NaN

# In[14]:


data_dict['METTS MARK']


# In[15]:


data_dict = F_RetirarNaNs(data_dict)


# Após ser retirado os NaNs

# In[16]:


data_dict['METTS MARK']


# A pasta email_by_address será utilizada nesse trabalho. Como algumas keys em data_dict não possuem e-mail, essas pessoas serão desconsideradas. Isso reduz a base de dados, mas trará maior consistência do que supor que essas não trocam e-mails.

# In[17]:


for key in data_dict.keys():
    if not 'email_address' in data_dict[key].keys():
        data_dict.pop(key)


# # Task 2: Remove outliers
# A análise gráfica será utilizada para descobrir se existem outliers na base de dados.

# In[18]:


plot_list = ['salary','total_payments', 'bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 
             'restricted_stock', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']
ordem = F_OrdemFeatures(data_dict, plot_list)
F_PlotOutliers(plot_list, ordem)


# Temos que os valores extremos nos gráficos estão associadas a pessoas com altos cargos na Enron, como:
# 
# **LAVORATO JOHN J**: Principais executivos
# 
# **LAY KENNETH L**: fundador e CEO da companhia na época do escândalo
# 
# **SKILLING JEFFREY K**: fundador e ex-CEO da companhia na época do escândalo
# 
# Essas pessoas são chefes da companhia, assim possuem salários e bonificações diferenciadas dos demais funcionários e seus dados não podem ser considerados outliers.
# 
# *É necessário analisar os outros pontos que podem ser outleirs para determinar se também se tratam de pessoas com altos cargos.*

# ### SALARY

# In[19]:


ordem['salary'][-3:]


# Os pontos que poderiam ser outliers são de chefes da companhia. Nesse caso temos que além de **LAY KENNETH L** e **SKILLING JEFFREY K** temos **FREVERT MARK A** que era na época CEO de uma subsidiára do grupo a _Enron Whosale Services_

# ### EXERCISED STOCK OPTIONS

# In[20]:


ordem['exercised_stock_options'][-2:]


# Para exercised_stock_options temos dois altos executivos o que justifica os outliers.

# ### RESTRICTED STOCK

# In[21]:


ordem['restricted_stock'][-3:]


# No caso de restricted stock temos pessoas com altos cargos que recebiam remuneração através de ações associadas a desempenho. 
# 
# Temos que as 3 pessoas são altos executivos o que justifica os altos ganhos.

# ### EMAILS

# A quantidade de emails pode variar muito independente da pessoa e cargo devido as funções exercidas, por isso não será considerada um outlier os valores extremos no gráfico.

# Após o cleaning data não existem outliers. O que temos é uma discrepância que pode ser considerada normal dentro de renda de uma companhia.

# # Task 3 --  Create New Features

# As maioria das features estão relacionadas a finanças, mas como vimos na análise de outliers, os rendimentos podem variar muito dentro de uma companhia. Uma forma de deixar a análise de POis mais robusta e trazer para o dataset as relações entre pessoas e POis.
# 
# Para trazer a relação entre pessoas no dataset as features relacionadas a envio de e-mails nos auxiliam nesse sentido, mas podemos complementar com a base de dados email_by_address de onde podemos tirar com quantos POis uma pessoa se relacionou enviando ou recebendo e-mails.
# 
# Essa cria uma nova feature, mas como não temos o e-mail de todas as pessoas no dataset optei por excluir as pessoas sem e-mail, ou seja, aquelas que não podem ser rastreadas (visto em cleaning data). Com isso o dataset fica mais consistente, porém a quantidade de dados que é pequena ficará ainda menor.

# Para cada pessoa em data_dict pego os destinatários dos emails from e to

# In[22]:


email_poi = ((data_dict[key]['email_address'], data_dict[key]['poi']) for key in data_dict.keys() 
                if 'email_address' in data_dict[key] and 'poi' in data_dict[key])

address_from = {file_.split('_')[1][:-4]:file_ for file_ in os.listdir('emails_by_address') if file_.split('_')[0]=='from'}
address_to = {file_.split('_')[1][:-4]:file_ for file_ in os.listdir('emails_by_address') if file_.split('_')[0]=='to'}

from_, to_ = {}, {}
for email, poi in email_poi:
    from_[email] = F_PessoaEmail('emails_by_address', address_from)
    to_[email] = F_PessoaEmail('emails_by_address', address_to)      


# In[23]:


poi_aux_list = F_ExtraiPoisAbreviados(data_dict)


# In[24]:


poi_aux_list


# Determinar a quantidade POis com que a pessoa interage por e-mails

# In[25]:


from_count, from_ratio = F_StatMsg(from_, poi_aux_list)
to_count, to_ratio = F_StatMsg(to_, poi_aux_list)


# Aqui tenho as novas características construídas, preciso adicionar ao data_dict
# 
# from_ratio e to_ratio não serão incluídos, pois existem os campos from_messages, from_poi_to_this_person, from_this_person_to_poi e to_messages que serão combinados para gerar uma feature.
# 
# As featrues to_count e from_count me dão a quantidade de POis que receberam ou enviaram mensagem para a key no dataset, ou seja, eu tenho a quantidade de relacionamentos com POis de uma determinada pessoa. Podemos supor que pessoas que não trocaram e-mais com POis não tem relação com a fraude da Enron e portanto não são POis.

# In[26]:


for key in data_dict.keys():
    
    email = data_dict[key]['email_address']
    
    data_dict[key]['from_count'] = from_count[email]
    data_dict[key]['to_count'] = to_count[email]


# Fazendo uam simples verificação se a atualização ocorreu corretamente

# In[27]:


data_dict[data_dict.keys()[27]]


# Adicionando a razão de envio/recebimento de e-mails de uma pessoa com um POis e apagando campos desnecessários para continuidade das análises

# In[28]:


for key, value in data_dict.items():
    try:
        
        if value['to_messages'] == 0:
            value['ratio_to'] = 0
        
        if value['from_messages'] == 0:
            value['ratio_from'] = 0 
        
        if not 0 in [value['to_messages'],value['from_messages']]:
            value['ratio_to'] = value['from_this_person_to_poi'] / float(value['to_messages'])
            value['ratio_from'] = value['from_poi_to_this_person'] / float(value['from_messages'])
            
        value.pop('from_this_person_to_poi')
        value.pop('to_messages')
        value.pop('from_poi_to_this_person')
        value.pop('from_messages')
            
        
    except:
        #se o campo não existe, não considero na base final
        data_dict.pop(key)


# Com todas as alterações ocorridas o dataset ficou reduzido a 86 pessoas (ver abaixo), porém os dados estão mais consistentes que supor valores para dados que estejam faltando.
# 
# É um dataset pequeno o que pode dificultar a qualidade do classificador

# In[29]:


len(data_dict)


# In[30]:


data_dict[data_dict.keys()[9]]


# # Feature Scalling
# Para evitar erros computacionais os valores devem ser bem condicionados, logo vou deixar os dados entre 0 e 1.
# 
# Vou adicionar as novas features criadas ao data set
# 
# Teremos 13 features para treinamento sendo que 4 foram criadas a partir do dataset inicial, onde 2 foram criadas com auxílio dos dados externos contidos em email_by_address.
# 
# Com isso temos 4 características que destacam relação entre pessoas e as demais estão associadas a finanças. O escândalo da Enron envolveu seus principais executivos que são pessoas que possuem padrões diferenciados de bônus, salários, gastos e ganhos especiais com ações (stocks) o que justifica usar as características que estão em features_list

# In[31]:


features_list = ['poi','salary','total_payments', 'bonus', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi', 
                 'to_count', 'from_count','ratio_to','ratio_from'] 


# In[32]:


data_dict['METTS MARK']


# In[33]:


data_dict, scale  = F_FeatureScalling(data_dict, ['email_address','poi'], features_list)


# In[34]:


data_dict['METTS MARK']


# Adicionar ao dataset final as correções e features executadas no dataset inicial

# In[35]:


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


# # Corrigindo a base

# Inicialmente havia apagado todos os valores com NaN para não gerar algum erro nas análises iniciais. 
# 
# As features em features_list serão utilizadas para criar o trainning e label sets, logo preciso que todos os elementos em my_dataset contenham essas features e por isso vou retornar ou criar as features apagadas com valor 0.
# 
# Ao adicionar valor 0 para features que eram Nan ou não existiam, estou perdendo consistência no dataset ao supor valores, mas esse passo será necessário para não reduzir ainda mais o dataset o que prejudicaria a performance do classificador e traria risco de overfitting

# In[36]:


not_person_feature = F_VerificarExisteFeature()
not_person_feature


# In[37]:


for key, values in not_person_feature.items():
    for value in values:
        my_dataset[key][value] = 0


# Verificando novamente se existe algum campo com problema

# In[38]:


F_VerificarExisteFeature()


# # Extrair features e labels

# In[39]:


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# # Task 4: Try a varity of classifiers

# Foram testados vários classificadores com vários parâmetros difirentes

# In[40]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[41]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


# ### Importando função para teste

# In[43]:


from tester import test_classifier
from sklearn.model_selection import GridSearchCV


# Vou utilizar vários classificadores e para melhorar o desempenho preciso ajustar o parâmetros de cada classificador. Para que isso ocorra vou usar GridSeachCV para fazer um teste exaustivo sobre várias configurações possíveis de parâmetros.
# 
# Essa busca vai fazer o algoritmo se ajustar melhor aos dados e consequentemente melhorar o desempenho do classificador

# ### Naive Bayes

# Usando Naive Bayes para teste simples

# In[44]:


clf = GaussianNB()
for ii in features_list:
    if ii != 'poi':
        print '\n',ii
        test_classifier(clf, my_dataset, ['poi',ii])


# Para classificador Naiva Bayes temos que o melhor encontrado foi usar somente uma feature (exercised_stock_options)

# In[45]:


features_completa = features_list[:]
features_completa.remove('ratio_from')
features_completa.remove('total_payments')

clf = GaussianNB()
test_classifier(clf, my_dataset, ['poi','exercised_stock_options'])


# ## Decision Tree

# In[46]:


clf = DecisionTreeClassifier()
for ii in features_list:
    if ii != 'poi':
        print '\n', ii
        test_classifier(clf, my_dataset, ['poi',ii])


# A partir da performance das features individualmente (resultados acima), foram testadas várias combinações manualmente.
# 
# O melhor resultado em acurácia, precision e recall para o classificador com configurações padrão está abaixo.

# In[63]:


clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, ['poi','expenses','total_stock_value','ratio_to','expenses'])


# Posso melhorar o desempenho do classificador mudando os seus parâmetros com uma busca exaustiva.
# 
# Os parâmetros para avaliar o classificador são acurácia, precisão e abrangência. Em nossa base a grande maioria das pessoas não são POis e é possível perceber que o classificador fica enviesado para retornar falso e assim ter uma alta acurácia. Precisamos priorizar a precisão e recall do classificador para saber com confiança se ele pode identificar um POis entra a maioria que não esteve envolvida no escandâlo da Enron.
# 
# Ter alta precision nos garante que o classificador tem baixo erro quando afirma que uma pessoa é um POI, enquanto um alto recall nos garante que o classificador pode encontrar uma grande quantidade de POis.

# In[79]:


clf = GridSearchCV(DecisionTreeClassifier(),{'min_samples_split':[2,3,4], 'max_depth':[2,3,4,5,None]},scoring='recall' )
test_classifier(clf, my_dataset, ['poi','expenses','total_stock_value','ratio_to','expenses'])


# Melhor parâmetro encontrado para Decision Trees foi

# In[80]:


clf.best_params_


# Usando o parâmetro vemos que temos uma precisão de aproximadamente 45% e recall de 56%.

# In[87]:


clf = DecisionTreeClassifier(max_depth=None, min_samples_split=3)
test_classifier(clf, my_dataset, ['poi','expenses','total_stock_value','ratio_to'])


# ## Random Forest

# Agora tentando melhorar a previsão vou usar Random Forest Classifier, pois possui uma tendência menor que Decision Trees em sofrer overfitting, embora a quantidade de características seja pequena.

# In[233]:


clf = RandomForestClassifier()
for ii in features_list:
    if ii != 'poi':

        print '\n',ii
        test_classifier(clf, my_dataset, ['poi',ii])


# In[234]:


clf = RandomForestClassifier()
test_classifier(clf, my_dataset, ['poi','expenses','bonus'])


# In[235]:


test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from'])


# In[236]:


test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','total_stock_value'])


# In[237]:


test_classifier(clf, my_dataset, ['poi','expenses','bonus','total_stock_value','ratio_from'])


# In[238]:


test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to',
                                  'total_stock_value','ratio_from'])


# In[240]:


test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to',
                                  'total_stock_value','from_count'])


# In[244]:


test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to',
                                  'total_stock_value','from_count','exercised_stock_options'])


# In[245]:


test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to',
                                  'total_stock_value','from_count','to_count','exercised_stock_options'])


# Após testar a uso de features temos que temos o conjunto abaixo apresentou o melhor resultado ao equilibrar precision e recall e será utilizada para testar para configurar o classificador

# In[250]:


clf = GridSearchCV(RandomForestClassifier(),{'n_estimators':[10,20,30], 'max_depth':[3,4,5,6,None], 
                                            'min_samples_split':[2,3,4,5]})
test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to','total_stock_value','from_count',
                                  'exercised_stock_options'])


# In[251]:


clf.best_params_


# Procurando encontrar os melhores parâmetros para recall

# In[252]:


clf = GridSearchCV(RandomForestClassifier(),{'n_estimators':[10,20,30], 'max_depth':[3,4,5,6,None], 
                                            'min_samples_split':[2,3,4,5]}, scoring = 'recall')
test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to','total_stock_value','from_count',
                                  'exercised_stock_options'])


# In[253]:


clf.best_params_


# Testando os melhores parâmetros encontrados para recall

# In[44]:


clf = RandomForestClassifier(max_depth=6, min_samples_split=5, n_estimators=30)
test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to','total_stock_value','from_count',
                                  'exercised_stock_options'])


# Procurando os melhores parâmetros para precision

# In[254]:


clf = GridSearchCV(RandomForestClassifier(),{'n_estimators':[10,20,30], 'max_depth':[3,4,5,6,None], 
                                            'min_samples_split':[2,3,4,5]}, scoring = 'precision')
test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to','total_stock_value','from_count',
                                  'exercised_stock_options'])


# In[255]:


clf.best_params_


# Testando os parâmetros encontrados para precision

# In[43]:


clf = RandomForestClassifier(max_depth=3, min_samples_split=4, n_estimators=30)
test_classifier(clf, my_dataset, ['poi','expenses','bonus','ratio_from','ratio_to','total_stock_value','from_count',
                                  'exercised_stock_options'])


# Usanto todas as features no classificador

# In[45]:


clf = GridSearchCV(RandomForestClassifier(),{'n_estimators':[30,50,70], 'max_depth':[3,4,5,6,None], 
                                            'min_samples_split':[2,3,4,5]}, scoring = 'recall')
test_classifier(clf, my_dataset, features_list)

clf.best_params_


# In[47]:


clf = RandomForestClassifier(max_depth=4, min_samples_split=2, n_estimators=30)
test_classifier(clf, my_dataset, features_list)


# # Melhor resultado

# Após vários testes temos que Random Forest não conseguiu resultados melhores que Decision Tree quando olhamos precision e recall. Como foi estabelecido que ambos devem estar acima de .3 somente o classificador Decision Tree foi aceito.
# 
# Os melhores parâmetros encontrados para Decision Tree classifier foram os default, exceto min_samples_split que foi determinado igual 3 para melhor ajuste.
# 
# Os valores encontrados para o melhor classificador foram:
# > Precision = 0.43804
# 
# > Recall = 0.55500
# 
# > Acurracy = 0.87144
# 
# > predictions = 9000	
# 
# > True positives = 555
# 
# > False positives = 712	
# 
# > False negatives = 445
# 
# > True negatives= 7288

# In[45]:


clf = DecisionTreeClassifier(max_depth=None, min_samples_split=3)
test_classifier(clf, my_dataset, ['poi','expenses','total_stock_value','ratio_to'])


# In[46]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

