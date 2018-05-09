function draw(geo){
	//alguns países são regiões segregadas
	countries_pisa = ["Albania","United Arab Emirates","Argentina","Australia","Austria","Belgium","Bulgaria","Brazil","Canada","Switzerland","Chile","Colombia","Costa Rica","Czech Republic","Germany","Denmark",
						"Spain","Estonia","Finland","France","United Kingdom","Greece","China","Croatia","Hungary","Indonesia","Ireland","Iceland","Israel","Italy","Jordan","Japan","Kazakhstan",
						"Korea","Liechtenstein","Lithuania","Luxembourg","Latvia","China","Mexico","Montenegro","Malaysia","Netherlands","Norway","New Zealand","Peru","Poland","Portugal","Qatar",
						"China","Russia","USA","Romania","Russian Federation","Singapore","Serbia","Slovak Republic","Slovenia",
						"Sweden","Chinese Taipei","Thailand","Tunisia","Turkey","Uruguay","Vietnam"];

	//Area do SVG
	var margin = -600,
		width = 1000 - margin,
		height = 600 - margin,
		selecao = "",
		gender = "None";
	
	debugger;
	
	//criando o objeto svg e passando seus atributos
	var svg = d3.select(".mapa")
				.append("svg")
				.attr("width",1000)
				.attr("height",600)
				.append("g").attr("class","map");
	
	//projeção -- passando latitude e longitude temos de converter para pixels
	//scale is similar to google maps zoom
	//translate is change the center of the map
	var projection = d3.geoMercator().scale(140).translate([width/3, height/3]);
	
	//Para criar o SVG precisamos das coordenadas dos polígonos -- passamos os polígonos por path e a projeção que queremos utilizar para desenhar os pixels no SVG
	//recebe os dados e aplica a projeção para renderizar a página
	//path é uma função
	var path = d3.geoPath().projection(projection);
	
	//agora juntamos os dados + polígonos com a projeção que escolhemos
	//geo.features é o vetor das coordenadas
	//.data(geo.features)é um vetor de espaços reservados
	//.enter() para selecionar todos os "paths" que não estão na página
	//.attr("d",path) -- acrescentamos o caminho ao nosso SVG que tem a propriedade d definida acima
	/*.data(geo.features).attr("class", function(d){return d.properties.name}) -- crio uma classe que representa cada país e na hora de mudar
		as cores vai ficar mais simples -- d3.selectAll(".Austria").style("fill",'red'); para selecionar as classes*/
	
	var map = svg.selectAll("path")
				.data(geo.features)
				.enter()
				.append("path")
				.attr("d",path)
				.attr("class", function(d){return d.properties.name})
				.style("fill",countries_startcolor)
				.style("stroke","steelblue")
				.style("stroke-width",1)
				.on("click", country_select);
				
				var legend = svg;
				legend.append("rect").attr("x",5).attr("y",1).attr("width",200).attr("height",60).attr("fill","none").style("stroke","white");
				
				legend.append("rect").attr("x",10).attr("y",15).attr("width",10).attr("height",10).attr("fill","rgb(200,200,200)");
				legend.append("text").html("PISA").attr("fill","darkblue").attr("dx","2em").attr("y",25);
				
				legend.append("rect").attr("x",10).attr("y",40).attr("width",10).attr("height",10).attr("fill","rgb(0,0,0)");
				legend.append("text").html("Non-PISA").attr("fill","darkblue").attr("dx","2em").attr("dy",50);
				
	function country_select(d,i){
		
		if(countries_pisa.includes(d.properties.name)){
			
			if(selecao!=""){
				d3.select("."+selecao.properties.name).style("fill",countries_startcolor(d));
			};
				
			selecao = d; //guardo a selecao
			
			d3.select(this).style("fill",'darkblue');
			d3.select(".country").text(selecao.properties.name).style("color","steelblue");
			//d3.selectAll(".svg_plot").remove();
			
			selected_country = d.properties.name;
			
			debugger;
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
		};
	};
	
	function countries_startcolor(d){
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