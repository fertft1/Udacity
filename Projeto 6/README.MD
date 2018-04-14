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