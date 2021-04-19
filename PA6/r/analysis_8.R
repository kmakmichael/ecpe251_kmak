A <- read.csv("gpu_8.csv")

#read in the variables
magtime <- A$magtime
supptime <- A$supptime
hysttime <- A$hysttime
edgetime <- A$edgetime

images <- c(256,512,1024,2048,4096,8192,10240)  #All the images

#arrays to hold average magnitude, suppression, hysteresis, and edge linking times
mtime <- c(0,0,0,0,0,0,0)
stime <- c(0,0,0,0,0,0,0)
htime <- c(0,0,0,0,0,0,0)
etime <- c(0,0,0,0,0,0,0)

#Fill the averages
count <- 0
repeat 
{
	x <- 30*count+1
	y <- 30*count+30
	
	mtime[count+1] <- mean(magtime[x:y])
	stime[count+1] <- mean(supptime[x:y])
	htime[count+1] <- mean(hysttime[x:y])
	etime[count+1] <- mean(edgetime[x:y])
	
	count <- count+1

	if(count>=7) {
		break
	}
}


xlabels <- c(256, 4096, 8192, 10240, 12000)

# mag & dir
plot(images,mtime,xlab="Image Width (px)",ylab="Magnitude & Direction Time (ms)", axes=FALSE)
axis(1, at=c(-1000, xlabels), labels=c(-1000, xlabels), col.axis="black", las=0)
axis(2, at=c(-1000, seq(0,50,by=5)), labels=TRUE, col.axis="black", las=2)
mag_model <- lm(log(mtime) ~ images)

# suppression
plot(images,stime,xlab="Image Width (px)",ylab="Suppression Time (ms)", axes=FALSE)
axis(1, at=c(-1000, xlabels), labels=c(-1000, xlabels), col.axis="black", las=0)
axis(2, at=c(-1000, seq(0,16,by=2)), labels=TRUE, col.axis="black", las=2)
supp_model <- lm(log(stime) ~ images)

# hysteresis
plot(images,htime,xlab="Image Width (px)",ylab="Hysteresis Time (ms)", axes=FALSE)
axis(1, at=c(-1000, xlabels), labels=c(-1000, xlabels), col.axis="black", las=0)
axis(2, at=c(-1000, seq(0,50,by=5)), labels=TRUE, col.axis="black", las=2)
hyst_model <- lm(log(htime) ~ images)

# edge linking
plot(images,etime,xlab="Image Width (px)",ylab="Edge Linking Time (ms)", axes=FALSE)
axis(1, at=c(-1000, xlabels), labels=c(-1000, xlabels), col.axis="black", las=0)
axis(2, at=c(-1000, seq(0,50,by=5)), labels=TRUE, col.axis="black", las=2)
edge_model <- lm(log(etime) ~ images)

