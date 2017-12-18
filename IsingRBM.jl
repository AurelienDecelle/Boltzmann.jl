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
X = X[:,1:10000]
X = X .* 2 - 1

rbm = IsingRBM(28*28, 100; TrainData=X)
#rbm = BernoulliRBM(28*28, 500; TrainData=X)
rbm.W = (rand(size(rbm.W))*2-1) * 4 * sqrt(6/(784*500))/4
#rbm.W = rand(Normal(0, 0.001), 100, 784)


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

vr = default_reporter(rbm, 100, X)

params = Dict()
params[:n_epochs] = 20
params[:batch_size] = 20
params[:n_gibbs] = 3
params[:randomize] = true
params[:sampler] = Boltzmann.sampler_UpdMeanfield
params[:lr] = 0.0001
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