A <- read.csv("memcpy.csv")

d2htime <- A$dtoh
h2dtime <- A$htod

dtime <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
htime <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

count <- 0
repeat 
{
	x <- 30*count+1
	y <- 30*count+30
	
	dtime[count+1] <- mean(d2htime[x:y])
	htime[count+1] <- mean(h2dtime[x:y])
	
	count <- count+1

	if(count>=20) {
		break
	}
}
sizes <- 2^(10:29)
d2h <- (sizes*4)/dtime
h2d <- (sizes*4)/htime

xlabels <- seq(0,1000,by=100)
ylabels <- seq(-1,4,by=0.5)


plot(sizes,h2d,pch=21,bg="blueviolet",col="blueviolet",xlab="Vector Size (MB)",ylab="Bandwidth (MB/s)",axes=FALSE)
axis(1, at=c(-(2^25), 2^20*xlabels, 2^30), labels=c(-(2^25), xlabels, 2^30), col.axis="black", las=0)
axis(2, at=ylabels*2^20, labels=ylabels*1000, col.axis="black", las=2)
h2d_model <- lm(h2d ~ log(sizes))
lines(sizes, predict(h2d_model, list(sizes)), col="blueviolet", lty=5)

plot(sizes,d2h,pch=21,bg="blueviolet",col="blueviolet",xlab="Vector Size (MB)",ylab="Bandwidth (MB/s)",axes=FALSE)
axis(1, at=c(-(2^25), 2^20*xlabels, 2^30), labels=c(-(2^25), xlabels, 2^30), col.axis="black", las=0)
axis(2, at=ylabels*2^20, labels=ylabels*1000, col.axis="black", las=2)
d2h_model <- lm(d2h ~ log(sizes))
lines(sizes, predict(d2h_model, list(sizes)), col="blueviolet", lty=5)

# times
#plot(sizes,htime,pch=21,bg="blueviolet",col="blueviolet",xlab="Vector Size (MB)",ylab="time (ms)",axes=FALSE)
#axis(1, at=c(-(2^25), 2^20*xlabels, 2^32), labels=c(-(2^25), xlabels, 2^30), col.axis="black", las=0)
#axis(2, at=c(-100, xlabels), labels=TRUE, col.axis="black", las=2)
htime_model <- lm(htime ~ sizes)
#lines(sizes, predict(htime_model, list(sizes)), col="blueviolet", lty=5)

#plot(sizes,dtime,pch=21,bg="blueviolet",col="blueviolet",xlab="Vector Size (MB)",ylab="time (ms)",axes=FALSE)
#axis(1, at=c(-(2^27), 2^20*xlabels, 2^32), labels=c(-(2^25), xlabels, 2^30), col.axis="black", las=0)
#axis(2, at=c(-100, xlabels), labels=TRUE, col.axis="black", las=2)
dtime_model <- lm(dtime ~ sizes)
#lines(sizes, predict(dtime_model, list(sizes)), col="blueviolet", lty=5)

h2d_predi <- predict(htime_model, data.frame(sizes = c(3072, 5120, 7680)^2))
d2h_predi <- predict(dtime_model, data.frame(sizes = c(3072, 5120, 7680)^2))
