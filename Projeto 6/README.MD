Obs: O material abaixo tem uma mistura de português e inglês. O motivo é que meus feedbacks foram de americanos, mas estava fazendo o projeto em português.

Data visualization with D3.js

To visualize:

	1. CLone/Download the repository

	2. Unzip

	3. Create a HTTP Server, there is a example in https://www.youtube.com/watch?v=4boogIZa1ko

	4. Open the browser in localhost like the video of the example

Resumo:
The visualization encourage the user to analyse many variables of the PISA grouped by gender and country. There is 400 variables available to analyse with use of bar plots.
It's possible compare the performance of the countries wich partipate(and non-participate) of the PISA in each variable and group by gender.

Insights que podem ser retirados das visualização:
1. Mapa
	1.1. Pelo mapa podemos notar que a Tunísia é o único país do continente africano que participa do PISA
	1.2. Ásia possue poucos representantes
	1.3. A China é representada por regiões, ou seja, a avaliação não oferece um panorama geral da educação no país
	1.4. A Grã-Bretanha não participa do PISA. Sendo um dos poucos países da Europa ocidental a não participar
2. Dados
	2.1. Educational level of the mother (ISCED)
		2.1.1. 25% dos estudante avaliados na Tunísia responderam que suas mãe não possuem instrução básica, enquanto no Brasil esse valor fica em torno de 15% , Argentina vai  a 10%,  Uruguai menos de 5% e Rússia 0%
		2.1.2. Na Rússia a maioria das mães está no nível educacional ISCED 5B E ISCED 5A,6 , enquanto no Brasil, Tunísia e Uruguai se encontram nos níveis mais baixos. Podemos supor que a Rússsia possue uma política
				educacional mais ampla
		2.1.3. No Japão a maioria das mães estão no faixa de educacional ISCED 3A, ISCED 4, portanto mais baixa que da Rússia. É um resultado interessante dado que o Japão possue um histórico de amplos investimentos
				em educação
	
	2.2. Educational level of the father
		2.2.1. No Japão a maioria dos pais está no nível educacional mais alto (ISCED 5A,6). Com um histórico do PISA poderíamos tentar entender que a diferença de educação entre pais e mães no Japão vem diminuindo 
				nos últimos anos. Sabendo que a sociedade japonesa foi machista durante vários anos isso pode ter afetado o acesso a educação das mulheres no passado.
		2.2.2. Comparar o nível de educação dos pais nas regiões da China mostra grandes discrepâncias,  enquanto Taipei mostra excelente nível de educação é possível ver que Macao tem níveis baixos
				Podemos comparar Taipei a países como bom sistema educacional como Finlândia e Macao a países em desenvolvimento como Brasil
		2.2.3. Quando olhamos os Estados Unidos vemos um padrão semelhante no nível educacional dos pais em diferentes regiões. Existe sempre uma concentração de resultados nos níveis (ISCED 3A, ISCED 4) e (ISCED 5A, 6)
	2.3 Attitudes
		2.3.1. Primeira coisa a se notar que vários não possuem dados sobre várias "attitudes" medidas, o que mostra que avaliação é não totalmente padronizada
		2.3.2. Para o item "Attitudes - Homework more fun" é possíve ver que no Japão a quantidade de estudantes que concordam e discordam (aproximadamente 30% para cada lado) dessa afirmação é verdadeira, enquanto na Austrália 
				em mais de 40% concordam e menos de 20% discordam. Não temos dados para entender os motivo, mas podemos supor que o coisas como nível de dificuldade, pressão poderiam desencadear essa variância
			    Stratificando por gênero os dado acima, vemos que existem pouca diferença se foram meninos ou meninas que responderam
		2.3.3. Para o item "Attitudes - Homework more fun" quando analisamos os países nórdicos, vemos que existe uma grande diferença nas respostas. A Finlândia que possue um dos melhores sistemas educacionais do mundo
				tem somente 20% dos estudantes respondendo "Strongly agree" para este item, enquanto Suécia e Noruega tem mais de 40% respondendo "Strongly agree". Podemos supor que o nível de cobrança nas tarefas dos finlandeses
				pode ser superior ou devemos ter outras variáveis latentes
		2.3.4. A pergunta "Math Interest - Enjoy Maths", é possível ver que existe em a maoria das repostas se concentram em "Aggree" e "Disagree", mas normalmente em "Disagree". No caso do Peru vemos que mais de 40% dos estudantes
				responderam "Agree" o que mostra uma grande aceitação da disciplina
		2.3.5. A pergunta "Math Self-Concept - Learn Quickly" temos que em alguns países a percepção dos estudantes sobre a dificuldade de aprender matemática pode variar bastatente, por exemplo, 
				no Japão mais de 50% dos estudantes responderam "Disagree" e no Brasil mais de 40%, enquanto na Finlândia mais de 40% responderam "Aggree"
		
	3. A análise exploratório pode avançar pelo interesse do usuário. Existem quase 400 variáveis para se analisar para mais de 65 países com vários itens por resposta dos estudantes
	
Design escolhido:
1. Cores escuras foram utilizadas por serem mais confortável para o olho humano
2. Cores escuras ajudam a perceber mudanças nos valores
3. As variáveis são categóricas o que justifica o uso de bar chart para toda as variáveis
4. A posição do mapa, gráficos e barra de seleção foram modificadas várias vezes durante o projeto para o layout final
5. A quantidade de gráfico foi limitada a 4 gráficos para facilicar a visualização a estimular a interação como mapa e gráficos
6. A página foi evoluindo para fornecer as informações iniciais para explicar do que se trata a página e as interações que o usuário pode fazer
	
The page is a interactive map and graphs about data gathering in Programme for International Student Assessment (PISA).
You can check informatio about PISA at http://www.oecd.org/pisa/.

Feedbacks-- Abaixo os feedbacks recebidos e em seguida as modificações que foram implementadas:
1. There is a problem, though. Your README does not say what I have to do to see your visualization. What steps should I follow to generate the viz?
	Modificação: Detalhar como utilizar os arquivos do GitHub para gerar a visualização

2. This chat is not very good at formatting messages, so I'll do my best.
	Modificação: Dimensionar a visualização para qualquer resolução de tela

3. When I just start, the default country is Brazil, but the country is not highlighted. I would recommend you: don't show anything by default OR highlight Brazil from the beginning
	Modificação-Para facilitar o entendimento a visualização começa com Brasil pintado com a cor de seleção

4. The legend does not say anything about the 'black' countries. It would be nice if you add something like 'countries without data' or whatever
	Modificação-Mudei a legenda e fiz uma introdução para esclarecer que não somente alguns países participam do PISA

5. Some countries such as Russia does not have any info, but it's not black. There is something wrong or maybe there is some data but not for the specific option I tried to explore? It would be good if a message would clarify this situation
	Modificação: Corrigindo erros de importação

6. maybe you could sort the labels in the xaxis. It would look much better
	Modificação: Criada ordenação das labels	

7. Cores diferentes e legendas para facilitar o entendimento do usuário sobre os dados que estão sendo exibidos
	Modificação: legendas e cores pastel para facilita a visualização