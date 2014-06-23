dataset<-read.csv("E:\\MyPgms\\Dataminingalgos\\dataset.csv")




kmeans(dataset,3)

eucliddist<-function(a,b){
return (sqrt(sum((a-b)**2)))
}

randcentroids<-function(data,k){
centroids<-data.frame(replicate(ncol(data),numeric(0)))
i=1
while (i<=k){
centroids[i,]<-runif(ncol(data),min(dataset[,1:ncol(dataset)]),max(dataset[,1:ncol(dataset)]))
i=i+1
}
colnames(centroids)=colnames(data)
return (centroids)
}




kmeans<-function(dataset,k){
centroids<-randcentroids(dataset,k)
print(centroids)
clusterassignment<-c()
i=1
j=1
k=1
kval<-Inf
temp<-0
cluster<-1
newvals<-c()
temporary<-list()
while (i<=nrow(dataset)){
cluster<-1
j=1
kval<-Inf
temp<-0

	while (j<=nrow(centroids)){
print(clusterassignment)
		temp<-eucliddist(centroids[j,],dataset[i,])
		if(kval>temp){
			kval=temp
			cluster<-j
		}
	j = j + 1
	}
	clusterassignment<-c(clusterassignment,cluster)
	i= i + 1
}
newvals<-clusterassignment
temporary<-data.frame(t(sapply(unique(newvals),function(x) apply(dataset[which(newvals==x),],2,mean))))
print(temporary)


while (k<=length(temporary[1])){
centroids[((unique(newvals))[k]),]<-temporary[k,]
k=k+1
}
print(centroids)
return (clusterassignment)
}

