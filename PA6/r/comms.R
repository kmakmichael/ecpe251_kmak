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

xlabels <- seq(0,600,by=100)
ylabels <- seq(-1,4,by=0.5)


plot(sizes,h2d,xlab="Vector Size (MB)",ylab="Bandwidth (MB/s)", axes=FALSE)
axis(1, at=c(-(2^25), 2^20*xlabels, 2^30), labels=c(-(2^25), xlabels, 2^30), col.axis="black", las=0)
axis(2, at=ylabels*2^20, labels=ylabels*1000, col.axis="black", las=2)
h2d_model <- lm(h2d ~ log(sizes))
lines(sizes, predict(h2d_model, list(sizes)), col="blue", lty=2)

plot(sizes,d2h,xlab="Vector Size (MB)",ylab="Bandwidth (MB/s)", axes=FALSE)
axis(1, at=c(-(2^25), 2^20*xlabels, 2^30), labels=c(-(2^25), xlabels, 2^30), col.axis="black", las=0)
axis(2, at=ylabels*2^20, labels=ylabels*1000, col.axis="black", las=2)
d2h_model <- lm(d2h ~ log(sizes))
lines(sizes, predict(d2h_model, list(sizes)), col="blue", lty=2)