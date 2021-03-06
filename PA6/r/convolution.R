g8 <- read.csv("gpu_8.csv")
g16 <- read.csv("gpu_16.csv")
g32 <- read.csv("gpu_32.csv")

convtime8 <- g8$convtime
convtime16 <- g16$convtime
convtime32 <- g32$convtime

images <- c(256,512,1024,2048,4096,8192,10240)^2

ctime <- c(0,0,0,0,0,0,0)

count <- 0
repeat 
{
	x <- 30*count+1
	y <- 30*count+30
	
	ctime[count+1] <- mean(c(convtime8[x:y],convtime16[x:y],convtime32[x:y]))
	
	count <- count+1

	if(count>=7) {
		break
	}
}

bytes <- c(0,0,0,0,0,0,0)
flops <- c(0,0,0,0,0,0,0)

count <- 1
bytes=(images) * 4
flops=(images) * 5

xlabels <- c(1024, 4096, 8192, 10240)
tvals <- seq(0, 10240, by=128)^2
blabels <- seq(0,600,by=100)

plot(bytes,ctime,pch=21,bg="blueviolet",col="blueviolet",xlab="Bytes (MB)",ylab="Convolution time (ms)", axes=FALSE)
axis(1, at=c(-(15000^2), blabels*2^20), labels=c(-(15000^2), blabels), col.axis="black", las=0)
axis(2, at=seq(-50,400,by=50), labels=TRUE, col.axis="black", las=2)
byte_model <- lm(ctime ~ bytes)
lines(bytes, predict(byte_model, list(bytes)), col="blueviolet", lty=5)

plot(flops,ctime,pch=21,bg="blueviolet",col="blueviolet",xlab="Flops (MF)",ylab="Convolution time (ms)", axes=FALSE)
axis(1, at=c(-(15000^2), blabels*2^20), labels=c(-(15000^2), blabels), col.axis="black", las=0)
axis(2, at=seq(-50,400,by=50), labels=TRUE, col.axis="black", las=2)
flop_model <- lm(ctime ~ bytes)
lines(flops, predict(byte_model, list(flops)), col="blueviolet", lty=5)

# time model for predictions
#plot(images,ctime,pch=21,bg="blueviolet",col="blueviolet",xlab="Image Size (px)",ylab="Convolution (ms)", axes=FALSE)
#axis(1, at=c(-(15000^2), xlabels^2), labels=c(-(15000^2), parse(text=paste(xlabels, "^2\n"))), col.axis="black", las=0)
#axis(2, at=seq(-50,400,by=50), labels=TRUE, col.axis="black", las=2)
conv_model <- lm(ctime ~ images)
#lines(images, predict(conv_model, list(images)), col="blueviolet", lty=5)

# prediction data
cv_predi <- predict(conv_model, data.frame(images = c(3072, 5120, 7680)^2))