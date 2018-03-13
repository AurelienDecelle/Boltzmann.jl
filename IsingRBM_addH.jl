include("reporter/reporter.jl")

using Boltzmann
using MNIST
# using Plots, Images
# using PyPlot
using Distributions



# function plotDigits(samples;path="digits.jpg")
#   #f = squarify(10,10,samples)
#   c = 10
#   r = 10
#   f = zeros(r*28,c*28)
#   for i=1:r, j=1:c
#     f[(i-1)*28+1:i*28,(j-1)*28+1:j*28] = reshape(samples[:,(i-1)*c+j],28,28)
#   end
#   w_min = minimum(samples)
#   w_max = maximum(samples)
#   λ = x -> (x-w_min)/(w_max-w_min)
#   map!(λ,f,f)
#   img = colorview(Gray,f)
#   println(path)
#   Images.save(path,img)
#   #plot(img)
# end

X, y = traindata()  # test data is smaller, no need to downsample
X = X ./ (maximum(X) - minimum(X)) 
X = X[:,1:5000]
X = X .* 2 - 1

#X = X .- mean(X,2)
#X = (X-minimum(X)) ./ (maximum(X) - minimum(X))
#X = X .* 2 - 1 

##uu,ss,vv = svd(X/sqrt(5000))
##yM=1.
##xM=ss[1]
##p = log(yM)/log(xM)
##sss = copy(ss);
##sss[60:end] = 10e-5
# sss[1:80] = 1.5
##Xn = uu*diagm((sss.^0.5).*2)*vv'
##X = Xn*sqrt(5000)


#X = X .- mean(X,2)
#X = (X-minimum(X)) ./ (maximum(X) - minimum(X))
#X = X .* 2 - 1 


# rbm = IsingRBM(28*28, 100; TrainData=X)
rbm = IsingRBM(28*28,1) #; TrainData=X)
#rbm = BernoulliRBM(28*28, 500; TrainData=X)
# rbm.W = (rand(size(rbm.W))*2-1) * 4 * sqrt(6/(784*500))/4
# rbm.W = rand(Normal(0, 0.0001), 100, 784)


# params = Dict(:n_epochs => 5, 
#           :batch_size => 20, 
#           :n_gibbs => 5, 
#           :randomize => true, 
#           :sampler=>sampler_UpdMeanfield, 
#           :approx=>"nmf", 
#           :lr=>0.02
# )

ApproxSampler = Dict()
ApproxSampler["cd"] = Boltzmann.contdiv
ApproxSampler["pcd"] = Boltzmann.persistent_contdiv
ApproxSampler["nmf"] = Boltzmann.sampler_UpdMeanfield
ApproxSampler["nmf_pcd"] = Boltzmann.persistent_meanfield
ApproxSampler["tap2"] = Boltzmann.sampler_UpdMeanfield
ApproxSampler["tap2_pcd"] = Boltzmann.persistent_meanfield

# vr = default_reporter(rbm, 20, X)

params = Dict()
params[:n_epochs] = 50
params[:batch_size] = 20
params[:n_gibbs] = 5
params[:randomize] = true
# params[:sampler] = Boltzmann.sampler_UpdMeanfield
params[:lr] = 0.00005
params[:approx] = "pcd"
params[:dump] = 0.1
# params[:reporter] = vr
params[:sample_app] = ApproxSampler
params[:sampler] = ApproxSampler[params[:approx]]
params[:TAP_neg_upd] = false

nbH = 1
addH = 5
sampling = []
while(nbH<62)
	#if(nbH>20)
	#	params[:n_epochs] = 20
	# end
	fit(rbm, X, params)
	Boltzmann.addHidden(rbm,addH)
	vp,hp,vn,hn = Boltzmann.gibbs_training(rbm,X[:,1:100];n_times=1000)
	push!(sampling,[vn,hn])
	nbH += addH
	# addH *= 2
end
#fit(rbm, X, n_epochs=20, batch_size=20, n_gibbs=2, randomize=true, 
#    sampler=Boltzmann.sampler_UpdMeanfield, approx="tap2", lr=0.00001,
#    dump=0.1 ,reporter=vr)

u,s,v = svd(rbm.W)
X_new = generate(rbm,X[:,1:100],n_gibbs = 1000)

# X_new = iter_mag(rbm,X[:,1:100], approx="nmf"; n_times=20)
# plotDigits(X_new)
# plotDigits(rbm.W[1:100,:]';path="weigths.jpg")

# TODO 
# * reconn_error
# * ponderated fixed point

# ger gradient for u,v !

gu = zeros(size(rbm.saving_tmp,1),100,100);
gv = zeros(size(rbm.saving_tmp,1),100,100);
v_t = zeros(size(rbm.saving_tmp,1),100,100);

for i=1:size(rbm.saving_tmp,1)
	dW = rbm.saving_tmp[i][2];
	dW_S = 0.5*(dW+dW');
	dW_A = 0.5*(dW-dW');

	sumW = zeros(100,100);
	summW = zeros(100,100);
	for α=1:100, β=1:100
		sumW[α,β] = 1/(rbm.saving_tmp[i][4][α] + rbm.saving_tmp[i][4][β]);
		summW[α,β] = 1/(rbm.saving_tmp[i][4][α] - rbm.saving_tmp[i][4][β]);
	end
	for α=1:100
		sumW[α,α] = 0
		summW[α,α] = 0
	end
	gu[i,:,:] = -(sumW.*dW_A) .+ (summW.*dW_S);
	gv[i,:,:] = (sumW.*dW_A) .+ (summW.*dW_S);
	v_t[i,:,:] = rbm.saving_tmp[i][3]
end


nH = 76
v_t = zeros(size(rbm.saving_tmp,1),nH,nH);
for i=1:size(rbm.saving_tmp,1)
	a = zeros(nH,nH)
	# b = zeros(784,nH)
	si = size(rbm.saving_tmp[i][3])
	if(si==1) si = [1,1] end
	a[1:si[1],1:si[2]] = rbm.saving_tmp[i][3]
	v_t[i,:,:] = a
	# v_t[i,:,:] = rbm.saving_tmp[i][3]
end

k = zeros(size(rbm.saving_tmp,1),nH);
for i=1:size(rbm.saving_tmp,1), j=1:nH k[i,j]  =kurtosis(v_t[i,:,j]) end
k[find(x->isnan(x),k)] = 0;

ss = zeros(size(rbm.saving_tmp,1),nH);
for i=1:size(rbm.saving_tmp,1) ss[i,:]=vcat(rbm.saving_tmp[i][4],zeros(nH-size(rbm.saving_tmp[i][4],1))) end

u,s,v = svd(rbm.W)
kv = zeros(size(v,2))
for i=1:size(v,2) kv[i] = kurtosis(v[:,i]) end

function sample(rbm; X=rand(784,1), t=1000, Δt=10)
	vn = X
	nt = round(Int,t/Δt)
	vn_s = zeros(784,nt)
	for	i=1:nt
		vp,hp,vn,hn = Boltzmann.gibbs_training(rbm,vn; n_times=Δt)
		vn_s[:,i] = vn
	end
	return vn_s
end

function sample_save(rbm; X=rand(784,1), t=1000, Δt=10)
	vn = X
	nt = round(Int,t/Δt)
	for	i=1:nt
		vp,hp,vn,hn = Boltzmann.gibbs_training(rbm,vn; n_times=Δt)
		plt[:clf]()
		# plt[:imshow](reshape(vn,28,28))
		plt[:imshow](reshape(tanh.(rbm.W'*hn),28,28))
		plt[:savefig](string("s_",i,".png"))
	end
end