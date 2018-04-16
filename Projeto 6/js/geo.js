/*O código cuida da criação, atualização e interação do usuário com o mapa*/
function draw(geo){
	//alguns países são regiões segregadas
	countries_pisa = ["Albania","United Arab Emirates","Argentina","Australia",
						"Austria","Belgium","Bulgaria","Brazil","Canada",
						"Switzerland","Chile","Colombia","Costa Rica",
						"Czech Republic","Germany","Denmark","Spain","Estonia",
						"Finland","France","United Kingdom","Greece","China",
						"Croatia","Hungary","Indonesia","Ireland","Iceland",
						"Israel","Italy","Jordan","Japan","Kazakhstan","Korea",
						"Liechtenstein","Lithuania","Luxembourg","Latvia",
						"China","Mexico","Montenegro","Malaysia","Netherlands",
						"Norway","New Zealand","Peru","Poland","Portugal",
						"Qatar","China","Russia","USA","Romania",
						"Russian Federation","Singapore","Serbia",
						"Slovak Republic","Slovenia","Sweden","Chinese Taipei",
						"Thailand","Tunisia","Turkey","Uruguay","Vietnam"];

	//Area do SVG
	var margin = 10,
		width = 900,
		height = 850,
		selecao = "",
		gender = "None";
		
	//projeção -- passando latitude e longitude temos de converter para pixels
	//scale is similar to google maps zoom
	//translate is change the center of the map
	var projection = d3.geoMercator()
					.scale(120)
					.translate([width/2, height/2.3]);
	
	/*Para criar o SVG precisamos das coordenadas dos polígonos -- passamos os 
	polígonos por path e a projeção que queremos utilizar para desenhar os
	pixels no SVG recebe os dados e aplica a projeção para renderizar a página
	path é uma função*/
	var path = d3.geoPath().projection(projection);
	
	//criando o objeto svg e passando seus atributos			
	var svg = d3.select(".mapa")
				.append("svg")
				.attr("viewBox","0 0 " + width + " " + height);
				
	var g = svg.append("g").attr("class","map");
	
	var map = svg.selectAll("path")
				.data(geo.features)
				.enter()
				.append("path")
				.filter(function(d){
							if(d.properties.name !='Antarctica'){	
								return true;
							};
						}
				).attr("d",path)
				.attr("class", function(d){return d.properties.name})
				.style("fill",countries_startcolor)
				.style("stroke","steelblue")
				.style("stroke-width",1)
				.on("click", country_select);
				
				//adicionando legendas do SVG
				var legend = svg;
				legend.append("rect")
						.attr("x",5)
						.attr("y",1)
						.attr("width",200)
						.attr("height",60)
						.attr("fill","none")
						.style("stroke","white");
				
				legend.append("rect")
						.attr("x",10)
						.attr("y",15)
						.attr("width",10)
						.attr("height",10)
						.attr("fill","rgb(200,200,200)");
						
				legend.append("text")
						.html("Participates of PISA")
						.attr("fill","darkblue")
						.attr("dx","2em")
						.attr("y",25);
				
				legend.append("rect")
						.attr("x",10)
						.attr("y",40)
						.attr("width",10)
						.attr("height",10)
						.attr("fill","rgb(0,0,0)");
						
				legend.append("text")
						.html("Non-Participates(no data)")
						.attr("fill","darkblue")
						.attr("dx","2em")
						.attr("dy",50);
				
				var legend_2 = svg;
				legend_2.append("text")
						.attr("x",width-400)
						.attr("class","selected-country")
						.attr("fill","darkblue")
						.attr("y",35);
				
	function country_select(d,i){
		/*Função chamada quando um país é selelcionado
		  a função faz as atualizações necessárias 
		  (cores e informações nos gráficos) e traz a nova visualização*/
		
		if(countries_pisa.includes(d.properties.name)){
				
			debugger;
			
			if(reversed_cases[selected_country] != undefined){
				selected_country = reversed_cases[selected_country];
			};
			
			//selecao antiga
			d3.select('.'+selected_country).style('fill','rgb(200,200,200)');
			
			var subregions = special_cases[d.properties.name];
			d3.selectAll('.button-aux').remove();
			
			if( subregions != null){
				debugger;
				subregions.forEach(
					function(d){
						d3.selectAll('.control-group')
						.append('button')
							.attr('class','button-aux')
							.attr("onclick","subregion("+"'" +d+"'" + ")")
							.html(d);
					}
				);
			};
			
			selecao = d; //guardo a selecao
			d3.selectAll(".selected-country")
				.html(selecao.properties.name).style('font-size',30);
			
			d3.select(this).style("fill",'darkblue');
			
			d3.select(".country")
				.text(selecao.properties.name)
				.style("color","steelblue");
			
			selected_country = d.properties.name;
			
			//atualização dos gráficos 
			if(current_graph==null){
				src.forEach(
					function(d){
						draw_graphs([d.split(".")[0],1,null],pisa);
					}
				);
			}else{
				current_graph.forEach(
					function(d){
						draw_graphs([d.split(".")[0],1,null],pisa);
					}
				);
			};
		
		}else{
			alert("This country not participate of PISA," + 
			"there is not a dataset available");
		};
	};
	
	function countries_startcolor(d){
		//atualiza as cores
		
		//cores inicias para diferenciar quem participa do pisa e não participa
		if(countries_pisa.includes(d.properties.name)){
			return 'rgb(200,200,200)'; 
		}
		else{ 
			return 'rgb(0,0,0)'; 
		}
	};
	
	src.forEach(function(d){d3.csv("dados/" + d,draw_graphs);});
	
};