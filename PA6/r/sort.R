g8 <- read.csv("gpu_8.csv")
g16 <- read.csv("gpu_16.csv")
g32 <- read.csv("gpu_32.csv")

#read in the variables
sort8 <- g8$sorttime
sort16 <- g16$sorttime
sort32 <- g32$sorttime

images <- c(256,512,1024,2048,4096,8192,10240)^2  #All the images

#arrays to hold average magnitude, suppression, sorting, hysteresis, and edge linking times
stime <- c(0,0,0,0,0,0,0)

#Fill the averages
count <- 0
repeat 
{
	x <- 30*count+1
	y <- 30*count+30
	
	stime[count+1] <- mean(c(sort8[x:y],sort16[x:y],sort32[x:y]))
	
	count <- count+1

	if(count>=7) {
		break
	}
}

xlabels <- c(1024, 4096, 8192, 10240)

plot(images,stime,xlab="Image Size (px)",ylab="Sorting Time (ms)", axes=FALSE)
axis(1, at=c(-(15000^2), xlabels^2), labels=c(-(15000^2), parse(text=paste(xlabels, "^2\n"))), col.axis="black", las=0)
axis(2, at=seq(-10,60,by=10), labels=TRUE, col.axis="black", las=2)
sort_model <- lm(stime ~ images)