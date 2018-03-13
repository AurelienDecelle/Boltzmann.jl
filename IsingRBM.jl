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


rbm = IsingRBM(28*28, 100; TrainData=X)
# rbm = IsingRBM(28*28,100) #; TrainData=X)
#rbm = BernoulliRBM(28*28, 500; TrainData=X)
# rbm.W = (rand(size(rbm.W))*2-1) * 4 * sqrt(6/(784*500))/4
rbm.W = rand(Normal(0, 0.0001), 100, 784)


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

vr = default_reporter(rbm, 20, X)

params = Dict()
params[:n_epochs] = 40
params[:batch_size] = 20
params[:n_gibbs] = 3
params[:randomize] = true
params[:sampler] = Boltzmann.sampler_UpdMeanfield
params[:lr] = 0.0002
params[:approx] = "tap2"
params[:dump] = 0.1
params[:reporter] = vr
params[:sample_app] = ApproxSampler
params[:sampler] = ApproxSampler[params[:approx]]
params[:TAP_neg_upd] = true

fit(rbm, X, params)
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


