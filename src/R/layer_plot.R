draw_polygon <- function(x, y1, y2, col){
	polygon(x = c(x,rev(x)), y = c(y1, rev(y2)), col = col, border = NA)
}
# R_2 : an input matrix contained the data for plot [first row: all 0; last row: R^2; middle rows: accumulated explained R^2 by variables; if all R^2 is explained, set other to FALSE; else, set other to TRUE; a discription of variables can be sent to annotation]
# If use annotation, the number of rows in R_2 should be equal to annotation + other + 1

# Npredictor : number of predictors, x in the plot
# file_name : file name to be saved 
# title : plot title

layer_plot <- function(R_2, Npredictor, file_name, title, other = TRUE, annotation = c(), xlab = 'number of predictors', ylab = 'R^2'){
	png(file_name)
	total_R_2 <- R_2[dim(R_2)[1],]
	if(other){
		Nlayer <- dim(R_2)[1]-2
	}else{
		Nlayer <- dim(R_2)[1]-1
	}
	plot(Npredictor, total_R_2, main = title, xlab = xlab, ylab = ylab, ylim = c(0,1))
	lines(Npredictor, total_R_2)
	for(i in 1:Nlayer){
		draw_polygon(Npredictor, R_2[i,], R_2[i+1,], adjustcolor(rainbow(Nlayer)[i],alpha.f=0.5))
		lines(Npredictor, R_2[i+1,], col = rainbow(Nlayer)[i])
		points(Npredictor[-1], R_2[i+1,-1], col = rainbow(Nlayer)[i])
	}
	if(other){
		draw_polygon(Npredictor, R_2[Nlayer+1,], R_2[Nlayer+2,], adjustcolor('grey',alpha.f=0.5))
	}
	if(!is.null(annotation)){
		for(i in 1:Nlayer){
			text(tail(Npredictor, 1),mean(R_2[0:1+i,length(Npredictor)]),annotation[i], pos = 2)
		}
		if(other){
			text(tail(Npredictor, 1),mean(R_2[1:2+Nlayer,length(Npredictor)]),'other', pos = 2)
		}
	}
	dev.off()
}


Npredictor <- seq(0,30,by = 5)
total_R_2 <- c(0, 1-1/Npredictor[-1])
gene_R_2 <- total_R_2/2
metabolite_R_2 <- total_R_2/3
R_2 <- rbind(rep(0, length(Npredictor)), gene_R_2, gene_R_2 + metabolite_R_2, total_R_2)

layer_plot(R_2, Npredictor, file_name = 'lay_plot_example1.png', title = 'plot example', other = TRUE)

layer_plot(R_2, Npredictor, file_name = 'lay_plot_example2.png', title = 'plot example', other = FALSE)

layer_plot(R_2, Npredictor, file_name = 'lay_plot_example3.png', title = 'plot example', other = TRUE, annotation = c('gene', 'metabolites'))

layer_plot(R_2, Npredictor, file_name = 'lay_plot_example4.png', title = 'plot example', other = FALSE, annotation = c('gene', 'metabolites','hormones'))
