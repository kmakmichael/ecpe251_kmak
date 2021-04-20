g8 <- read.csv("gpu_8.csv")
g16 <- read.csv("gpu_16.csv")
g32 <- read.csv("gpu_32.csv")

#read in the variables
magtime8 <- g8$magtime
magtime16 <- g16$magtime
magtime32 <- g32$magtime
supptime8 <- g8$supptime
supptime16 <- g16$supptime
supptime32 <- g32$supptime
hysttime8 <- g8$hysttime
hysttime16 <- g16$hysttime
hysttime32 <- g32$hysttime
edgetime8 <- g8$edgetime
edgetime16 <- g16$edgetime
edgetime32 <- g32$edgetime

images <- c(256,512,1024,2048,4096,8192,10240)^2  #All the images

#arrays to hold average magnitude, suppression, hysteresis, and edge linking times
mtime8 <- c(0,0,0,0,0,0,0)
mtime16 <- c(0,0,0,0,0,0,0)
mtime32 <- c(0,0,0,0,0,0,0)
stime8 <- c(0,0,0,0,0,0,0)
stime16 <- c(0,0,0,0,0,0,0)
stime32 <- c(0,0,0,0,0,0,0)
htime8 <- c(0,0,0,0,0,0,0)
htime16 <- c(0,0,0,0,0,0,0)
htime32 <- c(0,0,0,0,0,0,0)
etime8 <- c(0,0,0,0,0,0,0)
etime16 <- c(0,0,0,0,0,0,0)
etime32 <- c(0,0,0,0,0,0,0)

#Fill the averages
count <- 0
repeat 
{
	x <- 30*count+1
	y <- 30*count+30
	
	mtime8[count+1] <- mean(magtime8[x:y])
	mtime16[count+1] <- mean(magtime16[x:y])
	mtime32[count+1] <- mean(magtime32[x:y])
	stime8[count+1] <- mean(supptime8[x:y])
	stime16[count+1] <- mean(supptime16[x:y])
	stime32[count+1] <- mean(supptime32[x:y])
	htime8[count+1] <- mean(hysttime8[x:y])
	htime16[count+1] <- mean(hysttime16[x:y])
	htime32[count+1] <- mean(hysttime32[x:y])
	etime8[count+1] <- mean(edgetime8[x:y])
	etime16[count+1] <- mean(edgetime16[x:y])
	etime32[count+1] <- mean(edgetime32[x:y])
	
	count <- count+1

	if(count>=7) {
		break
	}
}

# plotting vars
xlabels <- c(256, 4096, 8192, 10240, 12000)
tvals <- seq(0, 10240, by=128)
col8 <- "blueviolet"
col16 <- "firebrick2"
col32 <- "darkgreen"
mylty <- 5
mylwd <- 1.5

# mag & dir
plot(images,mtime8,pch=21,bg=col8,ylab="Magnitude & Direction Time (ms)",xlab="Image Width (px)",axes=FALSE)
points(images,mtime16, pch=22, col=col16, bg=col16)
points(images,mtime32, pch=23, col=col32, bg=col32)
axis(1, at=c(-(10000^2), xlabels^2), labels=c(-1000, xlabels), col.axis="black", las=0)
axis(2, at=c(-1000, seq(0,50,by=5)), labels=TRUE, col.axis="black", las=2)
mag_model8 <- lm(mtime8 ~ images)
mag_model16 <- lm(mtime16 ~ images)
mag_model32 <- lm(mtime32 ~ images)
lines(images, predict(mag_model8, list(images)), col=col8, lty=mylty, lwd=mylwd)
lines(images, predict(mag_model16, list(images)), col=col16, lty=mylty, lwd=mylwd)
lines(images, predict(mag_model32, list(images)), col=col32, lty=mylty, lwd=mylwd)
legend('topleft',legend=c("8x8","16x16","32x32"),col=c(col8,col16,col32),pt.bg=c(col8,col16,col32),lty=c(mylty,mylty,mylty),pch=c(19,22,23),cex=0.9, box.lty=0)

# suppression
#plot(images,stime8,pch=21,bg=col8,ylab="Suppression Time (ms)", xlab="Image Width (px)",axes=FALSE)
#points(images,stime16, pch=22, col=col16, bg=col16)
#points(images,stime32, pch=23, col=col32, bg=col32)
#axis(1, at=c(-(10000^2), xlabels^2), labels=c(-1000, xlabels), col.axis="black", las=0)
#axis(2, at=c(-1000, seq(0,50,by=5)), labels=TRUE, col.axis="black", las=2)
#supp_model8 <- lm(stime8 ~ images)
#supp_model16 <- lm(stime16 ~ images)
#supp_model32 <- lm(stime32 ~ images)
#lines(images, predict(supp_model8, list(images)), col=col8, lty=mylty, lwd=mylwd)
#lines(images, predict(supp_model16, list(images)), col=col16, lty=mylty, lwd=mylwd)
#lines(images, predict(supp_model32, list(images)), col=col32, lty=mylty, lwd=mylwd)
#legend('topleft',legend=c("8x8","16x16","32x32"),col=c(col8,col16,col32),pt.bg=c(col8,col16,col32),lty=c(mylty,mylty,mylty),pch=c(19,22,23),cex=0.9, box.lty=0)

# hysteresis
#plot(images,htime8,pch=21,bg=col8,ylab="Hysteresis Time (ms)", xlab="Image Width (px)",axes=FALSE)
#points(images,htime16, pch=22, col=col16, bg=col16)
#points(images,htime32, pch=23, col=col32, bg=col32)
#axis(1, at=c(-(10000^2), xlabels^2), labels=c(-1000, xlabels), col.axis="black", las=0)
#axis(2, at=c(-1000, seq(0,50,by=5)), labels=TRUE, col.axis="black", las=2)
#hyst_model8 <- lm(htime8 ~ images)
#hyst_model16 <- lm(htime16 ~ images)
#hyst_model32 <- lm(htime32 ~ images)
#lines(images, predict(hyst_model8, list(images)), col=col8, lty=mylty, lwd=mylwd)
#lines(images, predict(hyst_model16, list(images)), col=col16, lty=mylty, lwd=mylwd)
#lines(images, predict(hyst_model32, list(images)), col=col32, lty=mylty, lwd=mylwd)
#legend('topleft',legend=c("8x8","16x16","32x32"),col=c(col8,col16,col32),pt.bg=c(col8,col16,col32),lty=c(mylty,mylty,mylty),pch=c(19,22,23),cex=0.9, box.lty=0)

# edge linking
#plot(images,etime8,pch=21,bg=col8,ylab="Edge Linking Time (ms)", xlab="Image Width (px)",axes=FALSE)
#points(images,etime16, pch=22, col=col16, bg=col16)
#points(images,etime32, pch=23, col=col32, bg=col32)
#axis(1, at=c(-(10000^2), xlabels^2), labels=c(-1000, xlabels), col.axis="black", las=0)
#axis(2, at=c(-1000, seq(0,50,by=5)), labels=TRUE, col.axis="black", las=2)
#edge_model8 <- lm(etime8 ~ images)
#edge_model16 <- lm(etime16 ~ images)
#edge_model32 <- lm(etime32 ~ images)
#lines(images, predict(edge_model8, list(images)), col=col8, lty=mylty, lwd=mylwd)
#lines(images, predict(edge_model16, list(images)), col=col16, lty=mylty, lwd=mylwd)
#lines(images, predict(edge_model32, list(images)), col=col32, lty=mylty, lwd=mylwd)
#legend('topleft',legend=c("8x8","16x16","32x32"),col=c(col8,col16,col32),pt.bg=c(col8,col16,col32),lty=c(mylty,mylty,mylty),pch=c(19,22,23),cex=0.9, box.lty=0)

